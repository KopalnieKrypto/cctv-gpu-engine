"""Rebuild `zone-annotator-package/`, the package `station_package_test.py` trains on (#301).

Two windows of zone-annotator's own `tests/fixtures/tiny.mp4` (copies that hash
differently and decode the same), one zone, every sample answered, folds
`fold-1` = W1 and `fold-2` = W2, exported and passed through `check_export`.

Run with the Python of a zone-annotator checkout at the commit in
`client-appliance/zone-annotator.pin`, on Linux: on Windows the export writes `\\`
into image paths and its own `check_export` refuses the result.

    python build_zone_annotator_package.py <zone-annotator checkout> <work dir>

The package lands in `<work dir>/paczka`; copy it over `zone-annotator-package/`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ANSWERS = {
    "W1.mp4": ["spawanie", "spawanie", "postoj", "nierozpoznane"],
    "W2.mp4": ["postoj", "spawanie", "spawanie", "postoj"],
}


def run(za: Path, *args: str) -> None:
    result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=za)
    if result.returncode:
        raise SystemExit(f"{args[0]} exited {result.returncode}: {result.stdout}{result.stderr}")


def main() -> None:
    za, work = Path(sys.argv[1]), Path(sys.argv[2])
    from zone_annotator.project import Project

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    tiny = (za / "tests" / "fixtures" / "tiny.mp4").read_bytes()
    project = work / "hala"
    for i, name in enumerate(ANSWERS, start=1):
        clip = work / name
        clip.write_bytes(tiny + b"\x00\x00\x00\x08free" * i)
        run(za, "zone_annotator.ingest", str(clip), "--project", str(project),
            "--zone", "stanowisko-1=40,30,120,60", "--stride", "2",
            "--label", "spawanie", "--label", "postoj", "--label", "nierozpoznane",
            "--cannot-tell", "nierozpoznane",
            "--recorded-at", f"2026-11-2{i}T08:00:00Z")  # fmt: skip
    opened = Project.open(project)
    for clip in opened.clips():
        samples = opened.samples(zone=clip.zones[0])
        for sample, value in zip(samples, ANSWERS[clip.name], strict=True):
            opened.label(sample.id, value, author="fixture-301")
    run(za, "zone_annotator.split", str(project), "declare",
        "--fold", "fold-1=W1.mp4", "--fold", "fold-2=W2.mp4")  # fmt: skip
    run(za, "zone_annotator.export", str(project), "--out", str(work / "paczka"),
        "--bound-to", "station-head-test/v1")  # fmt: skip
    run(za, "zone_annotator.check_export", str(work / "paczka"))


if __name__ == "__main__":
    main()
