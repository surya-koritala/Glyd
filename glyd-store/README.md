# glyd-store

A store that compresses across the objects it holds: each object is
kept as a delta against the stored object it most resembles (found by
fingerprints) when that pays, small objects go into packs, and any
object comes back byte-exact. On a 1.2 TB bucket of releases, dumps
and events: 3.1× fewer bytes than zstd -3 per object (24× against
raw), every object back byte-exact; 4.6× on a 39 GB one. Built on the
[glyd](https://github.com/surya-koritala/Glyd) codec (BSD-3-Clause OR GPL-2.0); the
store is under the Business Source License 1.1 (LICENSE).

    glyd-store bucket/ --put mon.tar tue.tar wed.tar
    glyd-store bucket/ --get 2 -o wed.tar
    glyd-store bucket/ --stats | --verify | --compact | --delete ID | --rebase ID | --find NAME
    glyd-store meta/ --s3 s3://bucket/prefix --put wed.tar     # objects in S3 (or any S3-compatible service)
    glyd-store --audit s3://bucket/prefix                      # what it would save, in dollars a year
    glyd-store meta/ --s3 s3://bucket/prefix --rebuild         # meta/ made anew from the objects

S3 is spoken directly over HTTPS (Signature V4; `glyd-store/src/s3.rs`);
objects over 64 MB go up as parts on eight connections.
Credentials: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (and
`AWS_SESSION_TOKEN`), else `~/.aws/credentials` for `AWS_PROFILE` or the
default profile, else the instance or container role. Region:
`AWS_REGION`, the profile, or the bucket's own (a wrong one is
corrected on the first request). `AWS_ENDPOINT_URL` points at any
S3-compatible service (MinIO, Cloudflare R2, Backblaze B2, Ceph). SSO
and assume-role profiles are not read; export them to the environment
(`aws configure export-credentials --format env`).

```rust
let mut store = glyd_store::Store::open("bucket/")?;
let id = store.put("wed.tar", &data)?;
let back = store.get(id)?;
```

The numbers, the design and the measurements are in the repository's
README and `docs/`.
