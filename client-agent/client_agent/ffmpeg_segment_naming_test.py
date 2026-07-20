"""Real-ffmpeg check that buffer-mode segment names survive a respawn (issue #90).

The unit tests in ``recorder_test.py`` assert the *argv* ``build_ffmpeg_cmd``
emits. That cannot catch the failure mode #90 was actually about, because the
overwriting happened inside ffmpeg: the argv was well-formed, and
``chunk_%03d.mp4`` looked perfectly reasonable right up until the second
recorder run reset the counter to 000 and clobbered live footage.

So this file forks real ffmpeg and looks at the filenames on disk. It needs
ffmpeg but **not** Docker, unlike ``mediamtx_integration_test.py`` — the
naming contract doesn't need a real camera, only a realtime source, which
lavfi provides. That keeps it runnable on a macOS dev box.

Marked ``integration`` (it forks a real encoder and spends a few seconds of
wallclock); opt in with ``pytest -m integration``.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_agent.recorder import BUFFER_CHUNK_TEMPLATE, build_ffmpeg_cmd

pytestmark = pytest.mark.integration

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="segment-naming test needs a real ffmpeg/ffprobe on PATH",
)

# Short enough to keep the test a few seconds, long enough to span several
# segments. Production records 60s segments off a live camera; the naming
# mechanism is identical, only the interval differs.
SEGMENT_SECONDS = 1
SOURCE_SECONDS = 3


def _make_realtime_source(path: Path) -> None:
    """Encode a short clip with one keyframe per second.

    ``-c copy`` can only cut a segment at a keyframe, so a clip with the
    x264 default GOP (~250 frames) would yield a single segment no matter
    what ``-segment_time`` says — and the test would silently stop testing
    anything. Forcing 1s keyframes mirrors an IP camera, which keeps a short
    GOP so clients can join the stream quickly."""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10",
            "-t",
            str(SOURCE_SECONDS),
            "-c:v",
            "libx264",
            "-force_key_frames",
            "expr:gte(t,n_forced*1)",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _record_once(source: Path, out_dir: Path) -> None:
    """Run one recording into ``out_dir`` using the production segment flags.

    Built from :func:`build_ffmpeg_cmd` so the flags under test are the ones
    production actually emits, with three substitutions on the *input* side,
    all forced by reading a file instead of a camera:

    * ``-rtsp_transport tcp`` is dropped — it belongs to the RTSP demuxer, and
      ffmpeg aborts with "Option rtsp_transport not found" on a file input.
    * ``-re`` is added, throttling the read to realtime, which is what an RTSP
      stream inherently is. This one is load-bearing: ffmpeg stamps segment
      names with *wallclock* time, so a full-speed file read closes every
      segment inside a single second and they all overwrite each other.
    * ``-segment_time`` drops from 60s to 1s so the test spans several
      segments in seconds rather than minutes.

    The output side — ``-f segment``, ``-strftime 1``, ``-reset_timestamps 1``
    and the filename template, i.e. everything #90 changed — passes through
    untouched."""
    cmd = build_ffmpeg_cmd(
        url=str(source),
        duration_s=SOURCE_SECONDS,
        output_dir=str(out_dir),
        buffer_mode=True,
    )
    transport = cmd.index("-rtsp_transport")
    del cmd[transport : transport + 2]
    cmd.insert(cmd.index("-i"), "-re")
    cmd[cmd.index("-segment_time") + 1] = str(SEGMENT_SECONDS)
    subprocess.run(cmd, check=True, capture_output=True)


@requires_ffmpeg
def test_ffmpeg_segment_names_are_unique_within_and_across_runs(tmp_path: Path) -> None:
    """Two recordings into one camera dir must accumulate, never overwrite.

    This is #90's acceptance criterion "a recorder restart no longer destroys
    in-retention segments", checked the only way that really settles it: run
    the recorder twice into the same directory and count the files.

    Under the old ``chunk_%03d.mp4`` the second run restarts at 000 and the
    final count equals the per-run count instead of the sum — which is exactly
    how a box configured for ``buffer_hours=5`` ended up holding ~1 h."""
    source = tmp_path / "src.mp4"
    _make_realtime_source(source)

    cam_dir = tmp_path / "cam-1"
    cam_dir.mkdir()

    _record_once(source, cam_dir)
    after_first = sorted(p.name for p in cam_dir.glob("chunk_*.mp4"))
    assert len(after_first) > 1, (
        "expected several segments; a single file means the source had no "
        "intermediate keyframes and this test is not exercising naming"
    )

    _record_once(source, cam_dir)
    after_second = sorted(p.name for p in cam_dir.glob("chunk_*.mp4"))

    # THE criterion: every file the first run wrote is still there.
    assert set(after_first) <= set(after_second), "a respawn overwrote earlier segments"
    # ...and the second run added to history rather than replacing it. Deliberately
    # not an exact count: how many segments a run yields depends on where the
    # keyframes fall relative to wallclock second boundaries, so it varies between
    # otherwise identical runs (observed 2 then 3). Pinning it would buy a flaky
    # test and no extra confidence.
    assert len(after_second) > len(after_first), "a respawn produced no new segments"
    assert all((cam_dir / name).stat().st_size > 0 for name in after_second)


@requires_ffmpeg
def test_ffmpeg_segment_names_match_the_rolling_buffer_glob(tmp_path: Path) -> None:
    """Real ffmpeg output is discoverable by :class:`RollingBuffer`.

    ``recorder_test.py`` asserts this against names *expanded in Python*.
    Here the names come from ffmpeg itself, so a divergence between how
    Python's ``strftime`` and ffmpeg's read the template — the kind of thing
    that turns the buffer's glob into a permanent zero-match — has somewhere
    to surface."""
    from client_agent.buffer import RollingBuffer

    source = tmp_path / "src.mp4"
    _make_realtime_source(source)
    cam_dir = tmp_path / "cam-1"
    cam_dir.mkdir()
    _record_once(source, cam_dir)

    produced = sorted(p.name for p in cam_dir.glob("chunk_*.mp4"))
    assert produced, "ffmpeg wrote nothing matching the buffer's chunk_*.mp4 glob"
    # The template was interpreted, not taken literally: no stray '%' survived.
    assert all("%" not in name for name in produced), (
        f"ffmpeg wrote the template literally — is '-strftime 1' missing? {produced}"
    )
    assert produced != [BUFFER_CHUNK_TEMPLATE]

    buffer = RollingBuffer(base_dir=tmp_path, buffer_hours=1, segment_seconds=SEGMENT_SECONDS)
    assert buffer.has_recorded("cam-1") is True

    # A window wide enough to cover everything just written, so the assertion
    # is about discovery and ordering rather than about overlap arithmetic
    # (``recorder_test.py`` pins the boundary rules).
    now = datetime.now(UTC)
    found = buffer.chunks_in_range(
        "cam-1", start=now - timedelta(hours=1), end=now + timedelta(minutes=1)
    )
    # The buffer sorts by mtime; ``produced`` is sorted by filename. They must
    # agree — that equality is the lexical-order-is-chronological-order property.
    assert [c.path.name for c in found] == produced
