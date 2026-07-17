# Private dataset artifacts

The restricted frame payloads are present in this workspace under `releases/` and are intentionally
ignored by git. Do not publish them: the controlled and factory footage is authorized only for
internal model development and evaluation.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `releases/activity-classifier-issue33-20260717-v1.tgz` | 155,517,687 | `c475e7de8b0f3ab9591da1a91d57763af593451f809a24c93e859b8fcb44d147` |
| `releases/activity-classifier-issue33-20260717-v1-review.tgz` | 39,310,115 | `63a305a9ef8144fbc70bbd2fe587b25c60adc06c521f0837d6b84f3b42b271ca` |

The first archive contains the complete materialized corpus. It is also extracted in this directory:
1,000 ignored JPEG files under the six `<camera_geometry_id>/frames/` directories, with the tracked
`labels.jsonl`, metadata, quotas, decisions, summary, and review record at the dataset root. The
second archive contains all 40 review sheets plus their sample index. `review-record.json` pins each
sheet checksum and the exact reviewer-decision file.

Validate the extracted corpus, including every JPEG checksum and dimension, with:

```bash
uv run python -c "from pipeline.activity_dataset import validate_dataset; validate_dataset('datasets/activity-classifier')"
```
