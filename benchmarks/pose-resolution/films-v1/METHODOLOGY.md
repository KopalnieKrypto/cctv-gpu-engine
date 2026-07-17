# Films 1+2 detection-recall regression methodology

## Purpose and provenance

These fixtures pin the issue #86 requirement that neither candidate arm may
reduce detection recall relative to the same-run fixed-640 letterboxed
baseline on the two legacy validation films. They are a detection-only guard;
the activity labels from issue #32 are intentionally not reused or rescored.

| Fixture | Source SHA-256 | Sample | Visible person instances |
|---|---|---|---:|
| Film 1 | `2f6ef8a0eaa1b1c96f3171ea48f5e25e6008ca7af4c04d634945220717dbceb8` | first 60 frames at 1 fps | 34 |
| Film 2 | `652f133d155eab6fd681d74158fc86e8b39bd8ce655fc383950573340c552c95` | first 60 frames at 1 fps | 30 |

Every frame is 640×360. Per-frame JPEG hashes and boxes are locked in the two
manifests. The exact same materialized frames are read by every arm.

## Annotation and review

- Every visible person is annotated, including partially visible entry/exit
  frames. Empty frames were visually checked rather than inferred from a
  missing model detection.
- Film 1 is empty in frames 1–26 and contains one person in frames 27–60.
- Film 2 is empty in frames 1–25 and 47–51. It contains one person in frames
  26–46 and 52–60.
- Boxes use full-frame `xyxy` coordinates and include only visible body extent.
  There is no semantic zone filter for these regression fixtures.

Fixed-640 and fixed-1280 outputs from independently exported, same-weight
YOLO11s-pose models were used as drawing aids. For ordinary one-person frames,
the reviewed box is the coordinate midpoint of the two aid boxes, avoiding
making either benchmark candidate its own truth. All 120 frames and the
resulting overlays were then visually reviewed in numbered contact sheets.

Manual boundary/exception decisions:

- Film 1 frame 27 includes the entering head/upper body at bottom-left.
- Film 2 frame 26 contains one person; the nested second fixed-640 detection
  was a duplicate and was not annotated as a second person.
- Film 2 frame 46 includes the exiting head at bottom-right.
- Film 2 frames 47–51 are genuinely empty.
- Film 2 frame 52 includes the re-entering head/shoulder at bottom-right. Both
  aids missed it, so its visible-extent box was drawn manually.

The benchmark reports each arm's recall on each film. Eligibility compares
each candidate to the baseline values from that same isolated benchmark run;
it does not compare against the historical issue #83 detection count.

## Distribution

The 120 JPEGs are distributed as an immutable GitHub release asset. Git tracks
this methodology, both manifests, and the release checksum/instructions; the
materialized `frames/` directories are ignored.
