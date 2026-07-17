# Films regression fixture asset

The frame payload is attached to GitHub release `issue-86-films-v1` as
`films-v1-frames.tar.gz`.

- Size: 9,696,293 bytes
- SHA-256: `ff2cd232783969d49a1d1f814e6dabdc4fb5c862b7d3a2fdaf3352f57e8e712a`
- Contents: 60 checksummed 640×360 JPEGs for each of Films 1 and 2 under
  `film-1/frames/` and `film-2/frames/`.

Materialize the fixtures from the repository root:

```bash
gh release download issue-86-films-v1 \
  --pattern films-v1-frames.tar.gz \
  --dir /tmp/issue86-films-fixture

cd /tmp/issue86-films-fixture
shasum -a 256 -c \
  "$OLDPWD/benchmarks/pose-resolution/films-v1/assets.sha256"

tar -C "$OLDPWD/benchmarks/pose-resolution/films-v1" \
  -xzf films-v1-frames.tar.gz
```

Both manifests independently verify every materialized frame hash before CUDA
loads.
