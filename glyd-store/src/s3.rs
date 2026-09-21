//! S3 over HTTPS: PUT, GET, HEAD, DELETE and ListObjectsV2, signed with
//! Signature Version 4. Credentials come from the environment
//! (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`),
//! then the shared files (`~/.aws/credentials` and `~/.aws/config`, the
//! `AWS_PROFILE` or default profile), then the instance role (ECS
//! container credentials, or EC2's IMDSv2), refreshed before they
//! expire. The region comes from `AWS_REGION`, `AWS_DEFAULT_REGION`,
//! the profile, or the bucket itself (a request to the wrong region is
//! answered with the right one and retried). `AWS_ENDPOINT_URL_S3` or
//! `AWS_ENDPOINT_URL` points at any S3-compatible service (MinIO,
//! Cloudflare R2, Backblaze B2, Ceph), which is addressed path-style.
//! Not covered: SSO and assume-role profiles; run those through
//! `aws configure export-credentials --format env` or the environment.

use std::io::{Error, ErrorKind, Result};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::Backend;

const IMDS: &str = "http://169.254.169.254";
const ECS_CREDENTIALS: &str = "http://169.254.170.2";
/// A single PUT holds this much; larger objects need multipart upload.
const MAX_PUT: usize = 5 << 30;
const ATTEMPTS: u32 = 4;

#[derive(Clone)]
struct Credentials {
    key: String,
    secret: String,
    token: Option<String>,
    /// When the instance role's credentials lapse (`None`: static).
    expires: Option<Instant>,
}

pub struct S3Backend {
    bucket: String,
    /// No trailing slash; may be empty.
    prefix: String,
    region: Mutex<String>,
    /// `https://bucket.s3.region.amazonaws.com` or the custom endpoint.
    endpoint: Option<String>,
    agent: ureq::Agent,
    creds: Mutex<Option<Credentials>>,
}

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    /// `<Code>` of an S3 error body.
    fn code(&self) -> String {
        xml_text(&self.body, "Code").unwrap_or_default()
    }
}

impl S3Backend {
    /// `s3://bucket/prefix` (the prefix may be empty).
    pub fn new(url: &str) -> Result<S3Backend> {
        let rest = url.strip_prefix("s3://").ok_or_else(|| bad("an s3:// url"))?;
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        if bucket.is_empty() {
            return Err(bad("an s3:// url with a bucket"));
        }
        let endpoint = std::env::var("AWS_ENDPOINT_URL_S3").or_else(|_| std::env::var("AWS_ENDPOINT_URL")).ok().map(|e| e.trim_end_matches('/').to_string());
        let region = std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION")).ok().or_else(profile_region).unwrap_or_else(|| "us-east-1".to_string());
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder().http_status_as_error(false).max_redirects(0).timeout_connect(Some(Duration::from_secs(10))).timeout_await_100(Some(Duration::from_secs(3))).user_agent(concat!("glyd-store/", env!("CARGO_PKG_VERSION"))).build(),
        );
        Ok(S3Backend { bucket: bucket.to_string(), prefix: prefix.trim_end_matches('/').to_string(), region: Mutex::new(region), endpoint, agent, creds: Mutex::new(None) })
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The objects under the prefix: (key relative to the prefix, bytes),
    /// in key order.
    pub fn list(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        let under = if self.prefix.is_empty() { String::new() } else { format!("{}/", self.prefix) };
        loop {
            let mut query = vec![("list-type".to_string(), "2".to_string()), ("prefix".to_string(), under.clone())];
            if let Some(t) = &token {
                query.push(("continuation-token".to_string(), t.clone()));
            }
            let r = self.request("GET", "", &query, &[])?;
            if r.status != 200 {
                return Err(other(format!("S3 list {}: {} {}", self.bucket, r.status, r.code())));
            }
            let text = String::from_utf8_lossy(&r.body);
            let mut rest = text.as_ref();
            while let Some(i) = rest.find("<Contents>") {
                let end = rest[i..].find("</Contents>").map(|e| i + e).unwrap_or(rest.len());
                let item = &rest[i..end];
                if let (Some(key), Some(size)) = (xml_text(item.as_bytes(), "Key"), xml_text(item.as_bytes(), "Size")) {
                    if let Ok(n) = size.parse::<u64>() {
                        out.push((key[under.len().min(key.len())..].to_string(), n));
                    }
                }
                rest = &rest[end..];
            }
            token = if xml_text(&r.body, "IsTruncated").as_deref() == Some("true") { xml_text(&r.body, "NextContinuationToken") } else { None };
            if token.is_none() {
                break;
            }
        }
        out.sort();
        Ok(out)
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() { key.to_string() } else { format!("{}/{}", self.prefix, key) }
    }

    /// Host and path for `key` ("" for the bucket itself).
    fn locate(&self, key: &str) -> (String, String, String) {
        let region = self.region.lock().unwrap().clone();
        // Path-style for custom endpoints and for dotted bucket names,
        // which break the wildcard certificate of virtual hosting.
        match &self.endpoint {
            Some(e) => {
                let (scheme, host) = e.split_once("://").unwrap_or(("https", e));
                (scheme.to_string(), host.to_string(), format!("/{}/{}", self.bucket, uri_encode(key, true)))
            }
            None if self.bucket.contains('.') => ("https".into(), format!("s3.{region}.amazonaws.com"), format!("/{}/{}", self.bucket, uri_encode(key, true))),
            None => ("https".into(), format!("{}.s3.{region}.amazonaws.com", self.bucket), format!("/{}", uri_encode(key, true))),
        }
    }

    /// One signed request with retries; a wrong-region answer moves the
    /// region and retries once.
    fn request(&self, method: &str, key: &str, query: &[(String, String)], body: &[u8]) -> Result<Response> {
        let mut moved = false;
        let mut delay = Duration::from_millis(200);
        let mut last = String::new();
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(delay);
                delay *= 2;
            }
            // No credentials is not a transient condition.
            let creds = self.credentials()?;
            match self.send(method, key, query, body, &creds) {
                Ok(r) => {
                    if let Some(region) = r.header("x-amz-bucket-region").map(str::to_string).or_else(|| if r.status == 400 || r.status == 301 { xml_text(&r.body, "Region") } else { None }) {
                        if (r.status == 301 || r.status == 400) && !moved && region != *self.region.lock().unwrap() {
                            *self.region.lock().unwrap() = region;
                            moved = true;
                            continue;
                        }
                    }
                    if r.status == 403 && r.code() == "ExpiredToken" {
                        *self.creds.lock().unwrap() = None;
                        last = "ExpiredToken".into();
                        continue;
                    }
                    if matches!(r.status, 429 | 500 | 502 | 503 | 504) {
                        last = format!("{} {}", r.status, r.code());
                        continue;
                    }
                    return Ok(r);
                }
                Err(e) => last = e.to_string(),
            }
        }
        Err(other(format!("S3 {method} {}: gave up after {ATTEMPTS} attempts ({last})", self.full_key(key))))
    }

    fn send(&self, method: &str, key: &str, query: &[(String, String)], body: &[u8], creds: &Credentials) -> Result<Response> {
        let full = if key.is_empty() { String::new() } else { self.full_key(key) };
        let (scheme, host, path) = self.locate(&full);
        let mut query: Vec<(String, String)> = query.to_vec();
        query.sort();
        let canonical_query = query.iter().map(|(k, v)| format!("{}={}", uri_encode(k, false), uri_encode(v, false))).collect::<Vec<_>>().join("&");
        let payload_hash = hex(&Sha256::digest(body));
        let (date, datetime) = amz_date(SystemTime::now());
        let mut headers = vec![("host".to_string(), host.clone()), ("x-amz-content-sha256".to_string(), payload_hash.clone()), ("x-amz-date".to_string(), datetime.clone())];
        if let Some(t) = &creds.token {
            headers.push(("x-amz-security-token".to_string(), t.clone()));
        }
        headers.sort();
        let signed = headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");
        let canonical = format!("{method}\n{path}\n{canonical_query}\n{}\n{signed}\n{payload_hash}", headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect::<String>());
        let region = self.region.lock().unwrap().clone();
        let scope = format!("{date}/{region}/s3/aws4_request");
        let to_sign = format!("AWS4-HMAC-SHA256\n{datetime}\n{scope}\n{}", hex(&Sha256::digest(canonical.as_bytes())));
        let signature = hex(&sign(&signing_key(&creds.secret, &date, &region), to_sign.as_bytes()));
        let authorization = format!("AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}", creds.key);
        let url = if canonical_query.is_empty() { format!("{scheme}://{host}{path}") } else { format!("{scheme}://{host}{path}?{canonical_query}") };
        let apply = |mut req: ureq::RequestBuilder<ureq::typestate::WithoutBody>| {
            for (k, v) in &headers {
                if k != "host" {
                    req = req.header(k.as_str(), v.as_str());
                }
            }
            req.header("authorization", authorization.as_str())
        };
        let result = match method {
            "PUT" => {
                // S3 answers before the body when asked to; without this
                // the store's 200 MB puts ended in a broken pipe.
                let mut req = self.agent.put(&url).header("expect", "100-continue");
                for (k, v) in &headers {
                    if k != "host" {
                        req = req.header(k.as_str(), v.as_str());
                    }
                }
                req.header("authorization", authorization.as_str()).send(body)
            }
            "GET" => apply(self.agent.get(&url)).call(),
            "HEAD" => apply(self.agent.head(&url)).call(),
            "DELETE" => apply(self.agent.delete(&url)).call(),
            _ => unreachable!(),
        };
        let mut resp = result.map_err(|e| other(format!("S3 {method} {host}{path}: {e}")))?;
        let status = resp.status().as_u16();
        let headers = resp.headers().iter().map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())).collect();
        let body = if method == "HEAD" { Vec::new() } else { resp.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| other(format!("S3 {method} {host}{path}: {e}")))? };
        Ok(Response { status, headers, body })
    }

    fn credentials(&self) -> Result<Credentials> {
        let mut slot = self.creds.lock().unwrap();
        if let Some(c) = &*slot {
            if c.expires.map_or(true, |t| t > Instant::now() + Duration::from_secs(300)) {
                return Ok(c.clone());
            }
        }
        let c = env_credentials().or_else(profile_credentials).map(Ok).unwrap_or_else(role_credentials)?;
        *slot = Some(c.clone());
        Ok(c)
    }
}

impl Backend for S3Backend {
    fn read(&self, key: &str) -> Result<Vec<u8>> {
        let r = self.request("GET", key, &[], &[])?;
        match r.status {
            200 => Ok(r.body),
            404 => Err(Error::new(ErrorKind::NotFound, format!("S3 GET {}: not found", self.full_key(key)))),
            s => Err(other(format!("S3 GET {}: {s} {}", self.full_key(key), r.code()))),
        }
    }
    fn write(&self, key: &str, data: &[u8]) -> Result<()> {
        if data.len() > MAX_PUT {
            // ponytail: single PUT, multipart upload when an object beats 5 GB.
            return Err(other(format!("S3 PUT {}: {} bytes is over the 5 GB single-put limit", self.full_key(key), data.len())));
        }
        let r = self.request("PUT", key, &[], data)?;
        if r.status == 200 { Ok(()) } else { Err(other(format!("S3 PUT {}: {} {}", self.full_key(key), r.status, r.code()))) }
    }
    fn remove(&self, key: &str) -> Result<()> {
        let r = self.request("DELETE", key, &[], &[])?;
        if r.status == 204 || r.status == 404 { Ok(()) } else { Err(other(format!("S3 DELETE {}: {} {}", self.full_key(key), r.status, r.code()))) }
    }
    fn exists(&self, key: &str) -> bool {
        self.len(key).is_some()
    }
    fn len(&self, key: &str) -> Option<u64> {
        let r = self.request("HEAD", key, &[], &[]).ok()?;
        if r.status == 200 { r.header("content-length")?.parse().ok() } else { None }
    }
}

// Credentials.

fn env_credentials() -> Option<Credentials> {
    let key = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
    let secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
    Some(Credentials { key, secret, token: std::env::var("AWS_SESSION_TOKEN").ok().filter(|t| !t.is_empty()), expires: None })
}

fn profile_name() -> String {
    std::env::var("AWS_PROFILE").unwrap_or_else(|_| "default".to_string())
}

fn home_file(var: &str, rel: &str) -> Option<String> {
    std::env::var(var).ok().or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.aws/{rel}"))).and_then(|p| std::fs::read_to_string(p).ok())
}

/// `key` of `[section]` in an INI text.
fn ini_value(text: &str, section: &str, key: &str) -> Option<String> {
    let mut inside = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            inside = line.trim_matches(|c| c == '[' || c == ']').trim() == section;
        } else if inside {
            if let Some((k, v)) = line.split_once('=') {
                if k.trim() == key {
                    return Some(v.trim().to_string());
                }
            }
        }
    }
    None
}

fn profile_credentials() -> Option<Credentials> {
    let text = home_file("AWS_SHARED_CREDENTIALS_FILE", "credentials")?;
    let p = profile_name();
    let key = ini_value(&text, &p, "aws_access_key_id")?;
    let secret = ini_value(&text, &p, "aws_secret_access_key")?;
    Some(Credentials { key, secret, token: ini_value(&text, &p, "aws_session_token"), expires: None })
}

fn profile_region() -> Option<String> {
    let text = home_file("AWS_CONFIG_FILE", "config")?;
    let p = profile_name();
    let section = if p == "default" { p } else { format!("profile {p}") };
    ini_value(&text, &section, "region")
}

/// The container's or instance's role, through the metadata services.
fn role_credentials() -> Result<Credentials> {
    let quick = ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(2))).build());
    let text = if let Ok(uri) = std::env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
        quick.get(format!("{ECS_CREDENTIALS}{uri}")).call().and_then(|mut r| r.body_mut().read_to_string()).map_err(|e| other(format!("container credentials: {e}")))?
    } else {
        let token = quick.put(format!("{IMDS}/latest/api/token")).header("x-aws-ec2-metadata-token-ttl-seconds", "21600").send_empty().and_then(|mut r| r.body_mut().read_to_string()).map_err(|_| other("no AWS credentials: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, a profile in ~/.aws/credentials, or run on an instance with a role"))?;
        let role = quick.get(format!("{IMDS}/latest/meta-data/iam/security-credentials/")).header("x-aws-ec2-metadata-token", &token).call().and_then(|mut r| r.body_mut().read_to_string()).map_err(|e| other(format!("instance role: {e}")))?;
        let role = role.lines().next().unwrap_or("").trim().to_string();
        quick.get(format!("{IMDS}/latest/meta-data/iam/security-credentials/{role}")).header("x-aws-ec2-metadata-token", &token).call().and_then(|mut r| r.body_mut().read_to_string()).map_err(|e| other(format!("instance role {role}: {e}")))?
    };
    let field = |name: &str| json_string(&text, name);
    let (Some(key), Some(secret), Some(token)) = (field("AccessKeyId"), field("SecretAccessKey"), field("Token")) else {
        return Err(other("instance role: credentials document without keys"));
    };
    // Expiration is ISO 8601; the service hands out at least an hour,
    // and a refresh five minutes early on a six-hour lease is safe.
    let lease = field("Expiration").and_then(|e| seconds_until(&e)).unwrap_or(3600);
    Ok(Credentials { key, secret, token: Some(token), expires: Some(Instant::now() + Duration::from_secs(lease)) })
}

/// `"name": "value"` of a flat JSON object.
fn json_string(text: &str, name: &str) -> Option<String> {
    let i = text.find(&format!("\"{name}\""))?;
    let rest = &text[i + name.len() + 2..];
    let rest = &rest[rest.find(':')? + 1..];
    let rest = &rest[rest.find('"')? + 1..];
    Some(rest[..rest.find('"')?].to_string())
}

/// Seconds from now to an ISO 8601 UTC time like 2026-09-21T22:00:00Z.
fn seconds_until(iso: &str) -> Option<u64> {
    let b = iso.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |s: &str| s.parse::<i64>().ok();
    let (y, mo, d, h, mi, s) = (num(&iso[0..4])?, num(&iso[5..7])?, num(&iso[8..10])?, num(&iso[11..13])?, num(&iso[14..16])?, num(&iso[17..19])?);
    let at = days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    Some((at - now).max(0) as u64)
}

// Time and encoding.

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (`YYYYMMDD`, `YYYYMMDDTHHMMSSZ`) of a time.
fn amz_date(t: SystemTime) -> (String, String) {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86400));
    let rem = secs.rem_euclid(86400);
    let date = format!("{y:04}{m:02}{d:02}");
    (date.clone(), format!("{date}T{:02}{:02}{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60))
}

/// RFC 3986 encoding; `/` is kept in paths.
fn uri_encode(s: &str, path: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b'/' if path => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sign(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn signing_key(secret: &str, date: &str, region: &str) -> Vec<u8> {
    let k = sign(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k = sign(&k, region.as_bytes());
    let k = sign(&k, b"s3");
    sign(&k, b"aws4_request")
}

/// The text of the first `<tag>` in an XML body, unescaped.
fn xml_text(body: &[u8], tag: &str) -> Option<String> {
    let text = std::str::from_utf8(body).ok()?;
    let open = format!("<{tag}>");
    let i = text.find(&open)? + open.len();
    let j = i + text[i..].find(&format!("</{tag}>"))?;
    Some(text[i..j].replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&"))
}

fn bad(what: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, format!("expected {what}"))
}

fn other(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::Other, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GET Object example of the SigV4 chapter of the S3 API
    /// reference (bucket examplebucket, us-east-1, 24 May 2013).
    #[test]
    fn sigv4_matches_the_documented_example() {
        let secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        let canonical = "GET\n/test.txt\n\nhost:examplebucket.s3.amazonaws.com\nrange:bytes=0-9\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130524T000000Z\n\nhost;range;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let to_sign = format!("AWS4-HMAC-SHA256\n20130524T000000Z\n20130524/us-east-1/s3/aws4_request\n{}", hex(&Sha256::digest(canonical.as_bytes())));
        assert_eq!(hex(&sign(&signing_key(secret, "20130524", "us-east-1"), to_sign.as_bytes())), "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
    }

    #[test]
    fn dates_and_encoding() {
        assert_eq!(amz_date(UNIX_EPOCH + Duration::from_secs(1369353600)), ("20130524".to_string(), "20130524T000000Z".to_string()));
        assert_eq!(civil_from_days(days_from_civil(2026, 9, 21)), (2026, 9, 21));
        assert_eq!(uri_encode("a b/c+d~e", true), "a%20b/c%2Bd~e");
        assert_eq!(uri_encode("a/b", false), "a%2Fb");
        assert_eq!(xml_text(b"<r><Key>a&amp;b</Key></r>", "Key").as_deref(), Some("a&b"));
        assert_eq!(json_string(r#"{"AccessKeyId" : "AK", "Token":"t"}"#, "Token").as_deref(), Some("t"));
        assert_eq!(ini_value("[default]\nregion=x\n[profile p]\nregion = eu-west-1\n", "profile p", "region").as_deref(), Some("eu-west-1"));
    }

    /// Round trip against a real bucket when GLYD_S3_TEST=s3://bucket/prefix.
    #[test]
    fn live_round_trip() {
        let Ok(url) = std::env::var("GLYD_S3_TEST") else { return };
        let s3 = S3Backend::new(&url).unwrap();
        let key = format!("s3-test-{}/a b&c.bin", std::process::id());
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7919 % 251) as u8).collect();
        assert!(!s3.exists(&key));
        s3.write(&key, &data).unwrap();
        assert_eq!(s3.len(&key), Some(data.len() as u64));
        assert_eq!(s3.read(&key).unwrap(), data);
        assert!(s3.list().unwrap().iter().any(|(k, n)| *k == key && *n == data.len() as u64));
        s3.remove(&key).unwrap();
        assert!(!s3.exists(&key));
        assert_eq!(s3.read(&key).unwrap_err().kind(), ErrorKind::NotFound);
    }
}
