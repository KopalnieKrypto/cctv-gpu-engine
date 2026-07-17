"""Render auditable full-frame pose review sheets for issue #33."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ACTIVITY_COLORS = {
    "sitting": (255, 180, 0),
    "standing": (0, 200, 0),
    "walking": (0, 165, 255),
    "running": (0, 0, 255),
}


def _annotated_thumbnail(
    dataset_dir: Path, sample: dict[str, Any], *, width: int, cell_height: int
) -> np.ndarray:
    frame = cv2.imread(str(dataset_dir / sample["frame_path"]))
    if frame is None:
        raise RuntimeError(f"failed to read review frame {sample['frame_path']}")
    color = _ACTIVITY_COLORS[sample["activity"]]
    x, y, bbox_width, bbox_height = sample["bbox"]
    cv2.rectangle(
        frame,
        (round(x), round(y)),
        (round(x + bbox_width), round(y + bbox_height)),
        color,
        thickness=max(2, frame.shape[1] // 640),
    )
    keypoint_radius = max(2, frame.shape[1] // 640)
    for keypoint in sample["keypoints"]:
        if keypoint["vis"] > 0:
            cv2.circle(
                frame,
                (round(keypoint["x"]), round(keypoint["y"])),
                keypoint_radius,
                (255, 0, 255),
                thickness=-1,
            )

    image_height = cell_height - 42
    scale = min(width / frame.shape[1], image_height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = np.zeros((cell_height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    cell[42 : 42 + resized_height, x_offset : x_offset + resized_width] = resized

    font_scale = max(0.32, width / 900)
    line_1 = f"{sample['activity']} | {sample['split']} | {sample['sample_id'][-20:]}"
    line_2 = (
        f"{sample['camera_geometry_id']} t={sample['source_timestamp_s']:.3f} "
        f"conf={sample['pose_confidence']:.3f}"
    )
    cv2.putText(
        cell,
        line_1,
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cell,
        line_2,
        (4, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return cell


def render_review_sheets(
    *,
    dataset_dir: Path,
    output_dir: Path,
    columns: int = 5,
    rows: int = 5,
    thumbnail_width: int = 320,
) -> dict[str, int]:
    """Render labeled tiles and a sheet-to-sample index for every dataset row."""
    with (dataset_dir / "labels.jsonl").open(encoding="utf-8") as labels_file:
        samples = [json.loads(line) for line in labels_file if line.strip()]
    split_order = {"train": 0, "validation": 1, "test": 2}
    samples.sort(
        key=lambda sample: (
            split_order[sample["split"]],
            sample["camera_geometry_id"],
            sample["activity"],
            sample["source_video_sha256"],
            sample["source_timestamp_s"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = columns * rows
    cell_height = round(thumbnail_width * 9 / 16) + 42
    sheet_count = math.ceil(len(samples) / per_sheet)
    review_index: dict[str, list[str]] = {}
    for sheet_index in range(sheet_count):
        page_samples = samples[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = np.zeros((rows * cell_height, columns * thumbnail_width, 3), dtype=np.uint8)
        for cell_index, sample in enumerate(page_samples):
            row, column = divmod(cell_index, columns)
            thumbnail = _annotated_thumbnail(
                dataset_dir,
                sample,
                width=thumbnail_width,
                cell_height=cell_height,
            )
            top = row * cell_height
            left = column * thumbnail_width
            sheet[top : top + cell_height, left : left + thumbnail_width] = thumbnail
        sheet_name = f"review-{sheet_index + 1:04d}.jpg"
        if not cv2.imwrite(str(output_dir / sheet_name), sheet):
            raise RuntimeError(f"failed to write review sheet {sheet_name}")
        review_index[sheet_name] = [sample["sample_id"] for sample in page_samples]
        print(
            f"review-heartbeat sheet={sheet_index + 1}/{sheet_count} "
            f"samples={min((sheet_index + 1) * per_sheet, len(samples))}/{len(samples)}",
            flush=True,
        )

    (output_dir / "review-index.json").write_text(
        json.dumps(review_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {"samples": len(samples), "sheets": sheet_count}
    (output_dir / "review-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=320)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = render_review_sheets(
        dataset_dir=args.dataset,
        output_dir=args.output,
        columns=args.columns,
        rows=args.rows,
        thumbnail_width=args.thumbnail_width,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on materialized data
    raise SystemExit(main())
