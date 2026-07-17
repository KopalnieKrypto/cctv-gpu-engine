# Fixture asset

The full-resolution payload is attached to GitHub release
`issue-86-bending-pilot-v1` as `bending-pilot-v1-assets.tar.gz`.

- Size: 192,774,857 bytes
- SHA-256: `39a5939c59e668baa4f87b67ad3730d807b62419a0d87c94bd2d220cc5d5d16f`
- Contents: 60 checksummed 3840×2160 JPEGs and three checksummed one-minute
  source clips under `frames/` and `clips/`.

Materialize the fixture from the repository root:

```bash
gh release download issue-86-bending-pilot-v1 \
  --pattern bending-pilot-v1-assets.tar.gz \
  --dir /tmp/issue86-fixture

cd /tmp/issue86-fixture
shasum -a 256 -c \
  "$OLDPWD/benchmarks/pose-resolution/bending-pilot-v1/assets.sha256"

tar -C "$OLDPWD/benchmarks/pose-resolution/bending-pilot-v1" \
  -xzf bending-pilot-v1-assets.tar.gz
```

`manifest.json` independently verifies every materialized frame hash before
CUDA loads. The three clip hashes are also recorded in the manifest and
methodology.
