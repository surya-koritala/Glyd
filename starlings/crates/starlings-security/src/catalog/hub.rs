//! Explicit Hub access; offline catalog operations never construct this client.
use super::{Catalog, read_bounded, validate_repository};
use crate::{Error, Result};
use starlings_protocol::{Digest, ModelSource, parse_json};
use std::time::{Duration, Instant};
use ureq::http::Uri;

pub struct Hub {
    agent: ureq::Agent,
    // Never Debug/log provider credentials or signed redirect URLs.
    token: Option<String>,
}
impl Hub {
    pub fn new(token: Option<String>) -> Result<Self> {
        if token.as_ref().is_some_and(|t| {
            t.is_empty() || t.len() > 4096 || !t.bytes().all(|b| b.is_ascii_graphic())
        }) {
            return Err(Error::Invalid);
        }
        let config = ureq::Agent::config_builder()
            .https_only(true)
            .proxy(None)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(60)))
            .build();
        Ok(Self {
            agent: config.into(),
            token,
        })
    }
    /// The caller explicitly selects the repository and revision to resolve.
    /// The returned immutable source still requires manifest/license approval.
    pub fn resolve(&self, repository: &str, revision: &str) -> Result<ModelSource> {
        validate_repository(repository)?;
        if revision.is_empty()
            || revision.len() > 128
            || matches!(revision, "." | "..")
            || !revision
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err(Error::Invalid);
        }
        let url = format!("https://huggingface.co/api/models/{repository}/revision/{revision}");
        let mut response = self.get(url, None)?;
        if response.status().as_u16() != 200 {
            return Err(Error::Transport);
        }
        resolved_source(repository, &read_bounded(response.body_mut().as_reader())?)
    }
    pub fn fetch(&self, catalog: &mut Catalog, digest: &Digest) -> Result<()> {
        let manifest = catalog.authorized(digest, false)?;
        for asset in &manifest.assets {
            if catalog.open_asset(digest, &asset.path).is_ok() {
                continue;
            }
            // One bounded transfer per invocation. A later explicit call resumes it.
            let offset = catalog.resume_offset(digest, &asset.path)?;
            if offset != 0 && offset == asset.size_bytes.0 {
                catalog.import_asset(digest, &asset.path, offset, std::io::empty())?;
                continue;
            }
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                manifest.source.repository, manifest.source.revision, asset.path
            );
            let mut response = self.get(url, (offset != 0).then_some(offset))?;
            let status = response.status().as_u16();
            // Servers may ignore Range; an ordinary full response restarts safely.
            let start = if status == 200 {
                0
            } else if status == 206 && offset != 0 {
                let expected = format!(
                    "bytes {offset}-{}/{}",
                    asset.size_bytes.0 - 1,
                    asset.size_bytes.0
                );
                if response
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    != Some(expected.as_str())
                {
                    return Err(Error::Invalid);
                }
                offset
            } else {
                return Err(Error::Transport);
            };
            if response
                .body()
                .content_length()
                .is_some_and(|n| n != asset.size_bytes.0 - start)
            {
                return Err(Error::Invalid);
            }
            catalog.import_asset(digest, &asset.path, start, response.body_mut().as_reader())?;
        }
        catalog.ready(digest)?;
        Ok(())
    }
    fn get(
        &self,
        mut url: String,
        offset: Option<u64>,
    ) -> Result<ureq::http::Response<ureq::Body>> {
        let deadline = Instant::now() + Duration::from_secs(60);
        for _ in 0..=4 {
            let uri = approved_url(&url)?;
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or(Error::Expired)?;
            let mut request = self
                .agent
                .get(url.as_str())
                .header("Accept-Encoding", "identity");
            if uri.host() == Some("huggingface.co")
                && let Some(token) = &self.token
            {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            if let Some(offset) = offset {
                request = request.header("Range", format!("bytes={offset}-"));
            }
            let response = request
                .config()
                .timeout_global(Some(remaining))
                .build()
                .call()
                .map_err(|_| Error::Transport)?;
            if matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
                let target = response
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or(Error::Invalid)?;
                if target.len() > 16 * 1024 {
                    return Err(Error::Capacity);
                }
                url = if target.starts_with('/') && !target.starts_with("//") {
                    format!("https://{}{target}", uri.authority().ok_or(Error::Invalid)?)
                } else {
                    target.to_owned()
                };
                continue;
            }
            if response
                .headers()
                .get("content-encoding")
                .is_some_and(|v| v != "identity")
            {
                return Err(Error::Invalid);
            }
            return Ok(response);
        }
        Err(Error::Capacity)
    }
}

fn approved_url(url: &str) -> Result<Uri> {
    if !url.is_ascii() || url.contains(['#', '\\']) || url.len() > 16 * 1024 {
        return Err(Error::Denied);
    }
    let uri: Uri = url.parse().map_err(|_| Error::Invalid)?;
    let authority = uri.authority().ok_or(Error::Denied)?.as_str();
    let host = authority.strip_suffix(":443").unwrap_or(authority);
    if uri.scheme_str() != Some("https")
        || !matches!(
            host,
            "huggingface.co"
                | "cdn-lfs.huggingface.co"
                | "cdn-lfs-us-1.hf.co"
                | "cdn-lfs-eu-1.hf.co"
                | "cas-bridge.xethub.hf.co"
        )
    {
        return Err(Error::Denied);
    }
    Ok(uri)
}
fn resolved_source(repository: &str, bytes: &[u8]) -> Result<ModelSource> {
    let value = parse_json(bytes)?;
    if value["id"].as_str() != Some(repository) {
        return Err(Error::Denied);
    }
    let sha = value["sha"].as_str().ok_or(Error::Invalid)?;
    if sha.len() != 40
        || !sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Invalid);
    }
    let created = value["createdAt"].as_str().ok_or(Error::Invalid)?;
    time::OffsetDateTime::parse(created, &time::format_description::well_known::Rfc3339)
        .map_err(|_| Error::Invalid)?;
    Ok(ModelSource {
        repository: repository.into(),
        revision: sha.into(),
        created_at_evidence: created.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn redirects_strip_credentials_reject_other_hosts_and_stop_loops() {
        use ureq::{
            Body, SendBody,
            http::{Request, Response},
            middleware::MiddlewareNext,
        };
        fn response(
            request: Request<SendBody>,
            _: MiddlewareNext,
        ) -> std::result::Result<Response<Body>, ureq::Error> {
            assert_eq!(request.headers()["range"], "bytes=4-");
            let uri = request.uri();
            let mut response = Response::builder();
            match (uri.host().unwrap(), uri.path()) {
                ("huggingface.co", path) => {
                    assert_eq!(request.headers()["authorization"], "Bearer test-secret");
                    let target = match path {
                        "/redirect" => "https://cdn-lfs.huggingface.co/object",
                        "/reject" => "https://unapproved.invalid/object",
                        "/loop" => "/loop",
                        _ => panic!("unexpected request"),
                    };
                    response = response.status(302).header("location", target);
                }
                ("cdn-lfs.huggingface.co", "/object") => {
                    assert!(!request.headers().contains_key("authorization"));
                    response = response.status(200);
                }
                _ => panic!("request escaped the provider allowlist"),
            }
            Ok(response
                .body(Body::builder().data(b"tail".to_vec()))
                .unwrap())
        }
        let hub = Hub {
            agent: ureq::Agent::config_builder()
                .middleware(response)
                .build()
                .into(),
            token: Some("test-secret".into()),
        };
        let mut response = hub
            .get("https://huggingface.co/redirect".into(), Some(4))
            .unwrap();
        assert_eq!(
            read_bounded(response.body_mut().as_reader()).unwrap(),
            b"tail"
        );
        assert!(matches!(
            hub.get("https://huggingface.co/reject".into(), Some(4)),
            Err(Error::Denied)
        ));
        assert!(matches!(
            hub.get("https://huggingface.co/loop".into(), Some(4)),
            Err(Error::Capacity)
        ));
    }
    #[test]
    fn provider_boundaries_reject_credential_and_redirect_confusion() {
        for url in [
            "http://huggingface.co/a",
            "https://huggingface.co.evil.invalid/a",
            "https://127.0.0.1/a",
            "https://user@huggingface.co/a",
            "https://huggingface.co:444/a",
            "https://cas-bridge.xethub.hf.co/a#fragment",
        ] {
            assert!(approved_url(url).is_err());
        }
        assert!(approved_url("https://cas-bridge.xethub.hf.co/a?signature=opaque").is_ok());
        assert!(Hub::new(Some("secret\r\nInjected: header".into())).is_err());
        let record = serde_json::json!({"id":"test/model", "sha":"a".repeat(40), "createdAt":"2026-10-01T00:00:00Z"});
        let bytes = serde_json::to_vec(&record).unwrap();
        assert_eq!(
            resolved_source("test/model", &bytes).unwrap().revision,
            "a".repeat(40)
        );
        assert!(resolved_source("other/model", &bytes).is_err());
    }
}
