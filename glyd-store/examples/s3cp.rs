// A file to or from S3 through the store's client (parts and ranges on
// several connections), for baselines that store other codecs' files:
//   s3cp put FILE s3://bucket/key
//   s3cp get s3://bucket/key FILE
use glyd_store::{Backend, S3Backend};

fn main() -> std::io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let split = |url: &str| {
        let rest = url.strip_prefix("s3://").expect("an s3:// url");
        let (bucket, key) = rest.split_once('/').expect("s3://bucket/key");
        let (prefix, name) = key.rsplit_once('/').unwrap_or(("", key));
        (format!("s3://{bucket}/{prefix}"), name.to_string())
    };
    match a.get(1).map(String::as_str) {
        Some("put") => {
            let (url, name) = split(&a[3]);
            S3Backend::new(&url)?.write(&name, &std::fs::read(&a[2])?)
        }
        Some("get") => {
            let (url, name) = split(&a[2]);
            std::fs::write(&a[3], S3Backend::new(&url)?.read(&name)?)
        }
        _ => {
            eprintln!("s3cp put FILE s3://bucket/key | s3cp get s3://bucket/key FILE");
            std::process::exit(1)
        }
    }
}
