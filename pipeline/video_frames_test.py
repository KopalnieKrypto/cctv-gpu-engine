"""Tests for streaming-frame extraction (1 fps) via ffmpeg rawvideo pipe.

ffmpeg is mocked at the subprocess boundary. The fake stdout pipes a sequence
of synthetic BGR frames so we can verify ``iter_frames`` decodes them, yields
the right number with ascending timestamps, and exits cleanly.
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.video_frames import iter_frames

# A fake ffmpeg that emits `n_frames` raw BGR frames to stdout while spraying
# `stderr_kib` KiB to stderr *interleaved* — reproducing a corrupt-MP4 decode
# storm. Under the pre-fix code (stderr=PIPE, never drained) the first stderr
# write blocks once the ~64 KiB OS pipe buffer fills, before any frame reaches
# stdout → classic subprocess deadlock.
_FAKE_FFMPEG = """
import sys
frame_size = int(sys.argv[1])
n_frames = int(sys.argv[2])
stderr_kib = int(sys.argv[3])
exit_code = int(sys.argv[4])
marker = sys.argv[5] if len(sys.argv) > 5 else ""

filler = b"E" * 1024
per_frame = max(1, stderr_kib // max(1, n_frames)) if n_frames else stderr_kib
for i in range(max(1, n_frames)):
    for _ in range(per_frame):
        sys.stderr.buffer.write(filler)
    sys.stderr.buffer.flush()
    if i < n_frames:
        sys.stdout.buffer.write(bytes([(i * 40) % 256]) * frame_size)
        sys.stdout.buffer.flush()
if marker:
    sys.stderr.buffer.write(marker.encode())
    sys.stderr.buffer.flush()
sys.exit(exit_code)
"""


def _write_fake_ffmpeg(tmp_path: Path) -> Path:
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(_FAKE_FFMPEG)
    return script


def _run_with_timeout(fn, timeout_s: float):
    """Run ``fn`` in a daemon thread; return (finished, result_or_exc).

    Used instead of pytest-timeout (not a dependency) to bound the pre-fix
    deadlock so the suite can't hang forever.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surface to caller
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    return (not t.is_alive()), box


def _fake_proc(frames_bytes: bytes, width: int, height: int):
    """Mimic the subset of subprocess.Popen that iter_frames touches."""
    proc = MagicMock()
    proc.stdout = io.BytesIO(frames_bytes)
    proc.stderr = io.BytesIO(b"")
    proc.returncode = 0
    proc.wait.return_value = 0
    proc.poll.return_value = 0
    return proc


def _bgr_frame(h: int, w: int, fill: int) -> bytes:
    return np.full((h, w, 3), fill, dtype=np.uint8).tobytes()


class TestIterFrames:
    def test_yields_decoded_frames_with_ascending_timestamps_at_1fps(self, mocker):
        w, h = 8, 6
        frames_blob = _bgr_frame(h, w, 11) + _bgr_frame(h, w, 22) + _bgr_frame(h, w, 33)
        proc = _fake_proc(frames_blob, w, h)

        # ffprobe returns the (width, height) tuple before ffmpeg is launched
        mocker.patch(
            "pipeline.video_frames._probe_dimensions",
            return_value=(w, h),
        )
        popen_mock = mocker.patch(
            "pipeline.video_frames.subprocess.Popen",
            return_value=proc,
        )

        results = list(iter_frames("video.mp4", fps=1))

        assert len(results) == 3
        ts = [t for t, _ in results]
        assert ts == pytest.approx([0.0, 1.0, 2.0])
        for _, frame in results:
            assert frame.shape == (h, w, 3)
            assert frame.dtype == np.uint8

        # ffmpeg invoked with -vf fps=1 -pix_fmt bgr24 -f rawvideo to stdout
        popen_args = popen_mock.call_args[0][0]
        assert popen_args[0] == "ffmpeg"
        assert "-vf" in popen_args
        assert popen_args[popen_args.index("-vf") + 1] == "fps=1"
        assert "-pix_fmt" in popen_args
        assert popen_args[popen_args.index("-pix_fmt") + 1] == "bgr24"
        assert popen_args[-1] == "-"
        # The video path is forwarded after -i
        assert popen_args[popen_args.index("-i") + 1] == "video.mp4"

    def test_raises_runtime_error_if_ffmpeg_exits_nonzero(self, mocker):
        w, h = 4, 4
        proc = _fake_proc(b"", w, h)
        proc.returncode = 1
        proc.wait.return_value = 1
        proc.stderr = io.BytesIO(b"ffmpeg: bad input")
        mocker.patch("pipeline.video_frames._probe_dimensions", return_value=(w, h))
        mocker.patch("pipeline.video_frames.subprocess.Popen", return_value=proc)

        with pytest.raises(RuntimeError, match="ffmpeg"):
            list(iter_frames("video.mp4", fps=1))


@pytest.mark.integration
class TestIterFramesStderrBackpressure:
    """Issue #60 — a chatty ffmpeg stderr must never deadlock the stream.

    These fork a real fake-ffmpeg process (mock pipes can't reproduce the OS
    pipe-buffer deadlock), so they carry the ``integration`` marker and are
    skipped by the default ``-m 'not integration'`` run.
    """

    def test_iter_frames_survives_chatty_stderr(self, tmp_path, mocker):
        w, h = 4, 4
        frame_size = w * h * 3
        script = _write_fake_ffmpeg(tmp_path)
        # 300 KiB of stderr — ~5x the ~64 KiB pipe buffer — interleaved with 3
        # frames. Pre-fix this deadlocks; post-fix all 3 frames arrive.
        mocker.patch("pipeline.video_frames._probe_dimensions", return_value=(w, h))
        mocker.patch(
            "pipeline.video_frames._build_ffmpeg_cmd",
            return_value=[sys.executable, str(script), str(frame_size), "3", "300", "0"],
        )

        finished, box = _run_with_timeout(lambda: list(iter_frames("video.mp4", fps=1)), 20.0)

        assert finished, "iter_frames deadlocked draining ffmpeg stdout vs stderr"
        assert "error" not in box, box.get("error")
        frames = box["result"]
        assert len(frames) == 3
        for _, frame in frames:
            assert frame.shape == (h, w, 3)

    def test_iter_frames_reports_stderr_on_failure(self, tmp_path, mocker):
        w, h = 4, 4
        frame_size = w * h * 3
        script = _write_fake_ffmpeg(tmp_path)
        marker = "FAKE_DECODE_ERROR_9f3a"
        # Heavy stderr (300 KiB) then a non-zero exit — the diagnostic marker is
        # the last thing written, so it must survive in the RuntimeError tail
        # even though the sink can't block the writer.
        mocker.patch("pipeline.video_frames._probe_dimensions", return_value=(w, h))
        mocker.patch(
            "pipeline.video_frames._build_ffmpeg_cmd",
            return_value=[sys.executable, str(script), str(frame_size), "1", "300", "3", marker],
        )

        finished, box = _run_with_timeout(lambda: list(iter_frames("video.mp4", fps=1)), 20.0)

        assert finished, "iter_frames hung instead of raising on ffmpeg failure"
        err = box.get("error")
        assert isinstance(err, RuntimeError)
        assert "exit 3" in str(err)
        assert marker in str(err)
