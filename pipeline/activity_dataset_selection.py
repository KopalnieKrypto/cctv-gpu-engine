"""Deterministic quota selection and materialization for issue #33."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def apply_review_decisions(
    candidates: list[dict[str, Any]], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply reviewer-authored confidence, interval, and sample exclusions."""
    minimum_confidence = decisions.get("minimum_pose_confidence", {})
    excluded_sample_ids = set(decisions.get("exclude_sample_ids", ()))
    excluded_intervals = decisions.get("exclude_intervals", ())
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        geometry_id = candidate["camera_geometry_id"]
        if candidate["sample_id"] in excluded_sample_ids:
            continue
        if float(candidate["pose_confidence"]) < float(minimum_confidence.get(geometry_id, 0)):
            continue

        timestamp = float(candidate["source_timestamp_s"])
        rejected_by_interval = False
        for interval in excluded_intervals:
            if geometry_id != interval["camera_geometry_id"]:
                continue
            if interval.get("activity") not in (None, candidate["activity"]):
                continue
            if interval.get("source_video_sha256") not in (
                None,
                candidate["source_video_sha256"],
            ):
                continue
            start_s = float(interval.get("start_s", "-inf"))
            end_s = float(interval.get("end_s", "inf"))
            if start_s <= timestamp < end_s:
                rejected_by_interval = True
                break
        if not rejected_by_interval:
            accepted.append(candidate)
    return accepted


def select_from_quota_plan(
    candidates: list[dict[str, Any]], quota_plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply geometry/activity/split quotas with global frame deduplication."""
    selected: list[dict[str, Any]] = []
    used_frame_hashes: set[str] = set()
    used_sample_ids: set[str] = set()
    for quota in quota_plan:
        group = [
            candidate
            for candidate in candidates
            if candidate["camera_geometry_id"] == quota["camera_geometry_id"]
            and candidate["activity"] == quota["activity"]
        ]
        picks = select_evenly_spaced(
            group,
            quota=int(quota["count"]),
            used_frame_hashes=used_frame_hashes,
            used_sample_ids=used_sample_ids,
        )
        for candidate in picks:
            selected_candidate = dict(candidate)
            selected_candidate["split"] = quota["split"]
            selected_candidate["review_status"] = "pending"
            selected.append(selected_candidate)
            used_frame_hashes.add(candidate["frame_sha256"])
            used_sample_ids.add(candidate["sample_id"])
    return selected


def select_evenly_spaced(
    candidates: list[dict[str, Any]],
    *,
    quota: int,
    used_frame_hashes: set[str],
    used_sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select ``quota`` unique frames spread across source video and time."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate["source_video_sha256"],
            candidate["source_timestamp_s"],
            candidate["sample_id"],
        ),
    )
    eligible: list[dict[str, Any]] = []
    seen_frame_hashes = set(used_frame_hashes)
    seen_sample_ids = set(used_sample_ids or ())
    for candidate in ordered:
        frame_hash = candidate["frame_sha256"]
        sample_id = candidate["sample_id"]
        if frame_hash in seen_frame_hashes or sample_id in seen_sample_ids:
            continue
        seen_frame_hashes.add(frame_hash)
        seen_sample_ids.add(sample_id)
        eligible.append(candidate)

    if len(eligible) < quota:
        raise ValueError(
            f"quota requires {quota} unique frames but only {len(eligible)} are available"
        )
    if quota == 0:
        return []
    if quota == 1:
        return [eligible[len(eligible) // 2]]

    indices = [round(index * (len(eligible) - 1) / (quota - 1)) for index in range(quota)]
    return [eligible[index] for index in indices]


def load_candidates(candidate_roots: list[Path]) -> list[dict[str, Any]]:
    """Load candidate JSONL rows and retain the root of each frame asset."""
    candidates: list[dict[str, Any]] = []
    for candidate_root in candidate_roots:
        labels_path = candidate_root / "candidates.jsonl"
        with labels_path.open(encoding="utf-8") as labels_file:
            for line in labels_file:
                if not line.strip():
                    continue
                candidate = json.loads(line)
                candidate["_asset_root"] = str(candidate_root)
                candidates.append(candidate)
    return candidates


def materialize_selection(
    *,
    candidate_roots: list[Path],
    metadata_dir: Path,
    output_dir: Path,
    quota_plan_path: Path,
    review_decisions_path: Path,
) -> dict[str, int]:
    """Select exact quotas and copy their immutable full-frame assets."""
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory {output_dir}")
    quota_plan = json.loads(quota_plan_path.read_text(encoding="utf-8"))["quotas"]
    candidates = load_candidates(candidate_roots)
    decisions = json.loads(review_decisions_path.read_text(encoding="utf-8"))
    candidates = apply_review_decisions(candidates, decisions)
    selected = select_from_quota_plan(candidates, quota_plan)

    output_dir.mkdir(parents=True)
    finalized: list[dict[str, Any]] = []
    for candidate in selected:
        source_path = Path(candidate["_asset_root"]) / candidate["frame_path"]
        final_relative_path = (
            Path(candidate["camera_geometry_id"]) / "frames" / f"{candidate['sample_id']}.jpg"
        )
        destination = output_dir / final_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        finalized_candidate = {
            key: value for key, value in candidate.items() if not key.startswith("_")
        }
        finalized_candidate["frame_path"] = final_relative_path.as_posix()
        finalized.append(finalized_candidate)

    with (output_dir / "labels.jsonl").open("w", encoding="utf-8") as labels_file:
        for candidate in finalized:
            labels_file.write(json.dumps(candidate, sort_keys=True) + "\n")

    for metadata_name in (
        "README.md",
        "geometries.json",
        "quota-plan.json",
        "review-decisions.json",
        "source-manifest.json",
    ):
        shutil.copy2(metadata_dir / metadata_name, output_dir / metadata_name)

    activity_counts = Counter(candidate["activity"] for candidate in finalized)
    split_counts = Counter(candidate["split"] for candidate in finalized)
    summary = {
        "samples": len(finalized),
        **{f"activity:{activity}": count for activity, count in sorted(activity_counts.items())},
        **{f"split:{split}": count for split, count in sorted(split_counts.items())},
    }
    (output_dir / "selection-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", action="append", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota-plan", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    materialize_selection(
        candidate_roots=args.candidate_root,
        metadata_dir=args.metadata_dir,
        output_dir=args.output,
        quota_plan_path=args.quota_plan,
        review_decisions_path=args.review_decisions,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on the GPU VPS
    raise SystemExit(main())
