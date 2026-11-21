# OSS-Fuzz

The files Google's [OSS-Fuzz](https://github.com/google/oss-fuzz) builds Glyd's fuzz targets with (`projects/glyd/` there): `Dockerfile`,
`build.sh` (every target in `fuzz/`, each with the seeds `fuzz/seed.py` makes) and `project.yaml`. A change here goes to OSS-Fuzz
as a pull request to google/oss-fuzz. To build them as OSS-Fuzz does, from a checkout of google/oss-fuzz with these files in
`projects/glyd/`:

```bash
python3 infra/helper.py build_fuzzers glyd && python3 infra/helper.py check_build glyd
```
