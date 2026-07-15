"""Per-frame detections JSONL dump (issue #35).

Archives the raw per-frame detections of a full-video run so post-hoc "why did
the system score X?" questions never again force an expensive (and possibly
non-reproducing) pipeline re-run — the round-7 bench false-positive pain. The
:class:`~pipeline.aggregator.Aggregator` deliberately keeps only a *bounded*
buffer of candidate keyframes (issue #49), so the full per-frame stream cannot
be reconstructed from :class:`~pipeline.aggregator.ReportData` after the fact —
it must be written line-by-line while the pipeline runs.

Opt-in via ``analyze --dump-detections <path.jsonl>`` and wired through the
gpu-service R2 worker so the artifact lands as ``detections.jsonl`` alongside
``result.json`` in the job's R2 prefix.

Schema — one JSON object per line (JSONL), one line per processed frame::

    {
      "timestamp_s": float,   # continuous across concatenated chunks
      "frame_idx": int,       # 0-based, monotonic across the whole video
      "person_count": int,
      "persons": [
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": float,
          "activity": str,
          "track_id": int | None,  # person identity (issue #32); None = untracked
          "keypoints": [{"x": float, "y": float, "vis": float}]  # 17 COCO points
        }
      ]
    }

The dump is deliberately *unfiltered*: it records every detection the model
made, including tracks the min-track-length filter later rejected and boxes the
tracker refused to identify (``track_id: null``). That is the point — it is the
audit trail for "why was this counted?", and a filtered dump could not answer
the question. Aggregated numbers come from ``result.json``, not from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType

from pipeline.postprocessing import Detection


def detection_to_dict(det: Detection) -> dict:
    """Serialize one :class:`Detection` to the ``persons[]`` element schema.

    Shared with the single-frame CLI payload so both paths emit an identical
    per-person shape.
    """
    return {
        "bbox": [float(v) for v in det.bbox],
        "confidence": float(det.confidence),
        "activity": det.activity,
        # None when tracking is off, or when the tracker refused this box an
        # identity — either way, the aggregator did not count it.
        "track_id": det.track_id,
        "keypoints": [
            {"x": float(kp.x), "y": float(kp.y), "vis": float(kp.vis)} for kp in det.keypoints
        ],
    }


def frame_to_jsonl_line(
    timestamp_s: float,
    frame_idx: int,
    detections: list[Detection],
) -> str:
    """Serialize one processed frame to a single-line JSON string (no newline)."""
    return json.dumps(
        {
            "timestamp_s": float(timestamp_s),
            "frame_idx": int(frame_idx),
            "person_count": len(detections),
            "persons": [detection_to_dict(d) for d in detections],
        }
    )


class DetectionsDumpWriter:
    """Streaming JSONL sink for per-frame detections, or a no-op when disabled.

    Constructed with a target ``path`` (or ``None`` to disable). Used as a
    context manager around the analysis loop so the file is always closed:

        with DetectionsDumpWriter(path) as dump:
            for ts, frame in frames:
                dump.write_frame(ts, detections)

    ``write_frame`` owns the ``frame_idx`` counter, so it stays 0-based and
    monotonic across every chunk of a concatenated video regardless of which
    classifier branch drives the loop. When ``path is None`` the writer opens
    no file and ``write_frame`` still advances the counter but writes nothing —
    callers need no ``if dump is None`` guards.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._fh = None
        self._frame_idx = 0

    def __enter__(self) -> DetectionsDumpWriter:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("w", encoding="utf-8")
        return self

    def write_frame(self, timestamp_s: float, detections: list[Detection]) -> None:
        """Append one JSONL line for a processed frame and advance ``frame_idx``."""
        if self._fh is not None:
            self._fh.write(frame_to_jsonl_line(timestamp_s, self._frame_idx, detections) + "\n")
        self._frame_idx += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
