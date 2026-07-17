# Activity classifier dataset

Issue #33 evaluation/training corpus for the four-class pose activity classifier. A sample is one
reviewed person in one unique full source frame; images are resized without cropping to at most
1280 px wide before pose labeling.

The materialized corpus is generated as a checksummed release payload because the operational
camera frames cannot be redistributed in the public source repository. `labels.jsonl`, source
checksums, camera metadata, split assignments, and the public-source rights remain versioned here.

## Camera geometries

The six geometries are defined in `geometries.json`. Mounting height, tilt, and horizontal FOV are
visual estimates rather than installation measurements, and each record states its uncertainty.
The set spans estimated heights 1.8–3.5 m, tilts 15–75°, and five horizontal FOV values.

Development geometries: `controlled-patio`, `factory-bending`, `factory-floor`, and
`pexels-track`. Test-only geometries: `controlled-garden` and `pexels-marathon`. Contact-sheet
review found that the provisional patio standing interval was continued walking, so the controlled
geometry roles were swapped instead of accepting an ambiguous label.

## Sources and licenses

- `controlled-film-1` and `controlled-film-2`: project-owner supplied controlled recordings;
  authorized for internal model development/evaluation, not redistribution.
- `camera-buffer-20260717-a7b76f41` and `camera-buffer-20260717-c88d18d9`: operator-authorized
  factory footage from the hardlink snapshot
  `/home/cameraboy/.local/state/cctv-client/issue33-source-20260717` on `cctv-vps-camera`;
  internal model development/evaluation only, not redistribution.
- `pexels-video-3943396`: “Sport Time Marathon Runners” by Nino Souza,
  <https://www.pexels.com/video/sport-time-marathon-runners-3943396/>.
- `pexels-video-11769251`: “Runners at Marathon” by Mesud Khalaf,
  <https://www.pexels.com/video/runners-at-marathon-11769251/>.
- The two public videos are used under the Pexels License:
  <https://www.pexels.com/license/>. They may be used and modified without attribution; this
  dataset stores derived labeled frames rather than redistributing the original stock videos.

Every input video is pinned by SHA-256 in `source-manifest.json`. No camera credentials are stored
in this directory.

## Split strategy

The fixed split is 700 train / 150 validation / 150 test. Train and validation are stratified over
the same four development geometries. The 150 test samples use only `controlled-garden` and
`pexels-marathon`, neither of which appears in train. Source-frame SHA-256 values are checked to
prevent a frame from crossing split boundaries. Every split contains all four activities.

The exact class target is 250 each of `sitting`, `standing`, `walking`, and `running`. Public
marathon footage supplies the running long pole; it is not synthetic. The other three activities
come from controlled and real factory footage.

## Labeling tool decision

We use the repository's CUDA-only YOLO-pose path through
`python -m pipeline.activity_dataset_collection`, followed by contact-sheet review and deterministic
quota selection. This was chosen over CVAT for this bounded one-person-per-frame corpus because it
reuses the exact COCO-17 inference contract consumed by the MLP, verifies source hashes before
labeling, retains source timestamps, and emits a visible heartbeat every 100 decoded frames or 60
seconds. Human-readable review artifacts remain separate from final labels.

Automatic activity hints are accepted only for the controlled intervals and clearly running public
clips. Factory hints from the geometric classifier are review candidates, not ground truth. Final
rows are not accepted until their activity, bbox, and pose overlay have been visually reviewed.
The reproducible confidence thresholds, transition ranges, and rejected corrupt frame are recorded
in `review-decisions.json`; `review-record.json` binds the final labels to checksums of every sheet.

## Label schema

`labels.jsonl` contains exactly 1,000 JSON objects. Core fields are:

- `bbox`: `[x, y, width, height]` in full-frame pixels.
- `activity`: `sitting`, `standing`, `walking`, or `running`.
- `keypoints`: exactly 17 COCO-order `{x, y, vis}` objects; `vis` is confidence in `[0, 1]`.
- `camera_geometry_id`, `source_id`, source timestamp/video checksum, split, image checksum, and
  image dimensions.

Run metadata-only validation with `verify_assets=False`; release verification checks every frame
file, SHA-256, and pixel dimension.
