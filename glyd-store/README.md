# glyd-store

A store that compresses across the objects it holds: each object is
kept as a delta against the stored object it most resembles (found by
fingerprints) when that pays, small objects go into packs, and any
object comes back byte-exact. On a 39 GB bucket of images, releases,
dumps and events: 4.6× fewer bytes than zstd -3 per object. Built on the
[glyd](https://github.com/surya-koritala/Glyd) codec (Apache-2.0 OR GPL-2.0); the
store is under the Business Source License 1.1 (LICENSE).

    glyd-store bucket/ --put mon.tar tue.tar wed.tar
    glyd-store bucket/ --get 2 -o wed.tar
    glyd-store bucket/ --stats | --verify | --compact | --delete ID | --rebase ID | --find NAME
    glyd-store meta/ --s3 s3://bucket/prefix --put wed.tar     # objects in S3 through the AWS CLI
    glyd-store --audit s3://bucket/prefix                      # what it would save, in dollars a year

```rust
let mut store = glyd_store::Store::open("bucket/")?;
let id = store.put("wed.tar", &data)?;
let back = store.get(id)?;
```

The numbers, the design and the measurements are in the repository's
README and `docs/`.
