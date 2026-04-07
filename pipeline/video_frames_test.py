"""Tests for streaming-frame extraction (1 fps) via ffmpeg rawvideo pipe.

ffmpeg is mocked at the subprocess boundary. The fake stdout pipes a sequence
of synthetic BGR frames so we can verify ``iter_frames`` decodes them, yields
the right number with ascending timestamps, and exits cleanly.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.video_frames import iter_frames


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
