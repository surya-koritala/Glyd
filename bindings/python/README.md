# Glyd for Python

    bindings/python/build.sh && pip install bindings/python

```python
import glyd
c = glyd.compress(data)                       # --max; level="ultra" / "cold" / "default"
c = glyd.compress(log_bytes, records=True)    # logs, dumps, CSV, JSON lines as typed columns
data = glyd.decompress(c)

p = glyd.pack(events)                         # many small objects as one stream
event = glyd.unpack(p, 7)

with glyd.Store("bucket/") as s:              # objects compressed across each other
    i = s.put("wed.tar", data)                # a delta against the object it most resembles
    data = s.get(i)
with glyd.Store("meta/", s3="s3://bucket/prefix") as s:   # objects in S3, through the AWS CLI
    ...
```

A thin ctypes layer over `include/glyd.h` (Apache-2.0 OR GPL-2.0); no build step
beyond placing the shared library. `build.sh` places `libglyd_store`,
which carries the codec and the store; with `libglyd` alone (the codec
crate, Apache-2.0 OR GPL-2.0) everything but `Store` works.
