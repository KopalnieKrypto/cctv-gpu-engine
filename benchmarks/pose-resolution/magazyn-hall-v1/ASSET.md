# Fixture asset

The full-resolution payload is attached to GitHub release
`issue-99-magazyn-hall-v1` as `magazyn-hall-v1-assets.tar.gz`.

- Size: 195,404,088 bytes
- SHA-256: `034e901b664057658ca82c41185457a058b0001e9ae48247cca61b54e30364d2`
- Contents: 60 checksummed 3840×2160 JPEGs and three checksummed one-minute
  source clips under `frames/` and `clips/`.

Materialize the fixture from the repository root:

```bash
gh release download issue-99-magazyn-hall-v1 \
  --pattern magazyn-hall-v1-assets.tar.gz \
  --dir /tmp/issue99-fixture

cd /tmp/issue99-fixture
shasum -a 256 -c \
  "$OLDPWD/benchmarks/pose-resolution/magazyn-hall-v1/assets.sha256"

tar -C "$OLDPWD/benchmarks/pose-resolution/magazyn-hall-v1" \
  -xzf magazyn-hall-v1-assets.tar.gz
```

`manifest.json` independently verifies every materialized frame hash before
CUDA loads. The three clip hashes are also recorded in the manifest and
methodology.

## Source object

The 2.4 GB source recording these clips were cut from is **not** part of the
asset — it is the platform's own upload and stays there:

- R2 key: `tenants/ryjUroVDl3H0xVhlh4pjqXZfcRuo8dTd/appliance-uploads/ce475156-f85a-4264-a23d-052105a50ec1/chunk_000.mp4`
- 2,396,924,949 bytes, HEVC 3840×2160, 3540.436 s

The three 60-second clips in this asset are byte-exact stream copies from it, so
the fixture is reproducible from the source without re-encoding.

## Status

The asset ships **unannotated** — every `persons` array in `manifest.json` is
empty. See `METHODOLOGY.md`.
