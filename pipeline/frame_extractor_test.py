"""Tests for single-frame extraction from MP4 via ffmpeg.

ffmpeg is mocked at the subprocess boundary so these tests don't need a real
ffmpeg binary or video file. A real-ffmpeg integration test lives under the
``integration`` marker and is skipped by default.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from pipeline.frame_extractor import extract_frame_at


def _png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestExtractFrameAt:
    def test_invokes_ffmpeg_with_seek_before_input_for_fast_seek(self, mocker):
        # Use a real-looking PNG so cv2.imdecode succeeds
        png = _png_bytes(1280, 720, color=(40, 80, 160))
        completed = MagicMock(returncode=0, stdout=png, stderr=b"")
        run_mock = mocker.patch(
            "pipeline.frame_extractor.subprocess.run",
            return_value=completed,
        )

        frame = extract_frame_at(Path("/videos/sample.mp4"), timestamp_s=12.5)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8

        # Verify ffmpeg was called with -ss BEFORE -i (fast seek pattern)
        cmd = run_mock.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx, "-ss must come before -i for fast seek"
        assert cmd[ss_idx + 1] == "12.5"
        assert cmd[i_idx + 1] == "/videos/sample.mp4"
        # Single frame, image2pipe to stdout as PNG
        assert "-frames:v" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "1"
        assert cmd[-1] == "-"  # stdout sink

    def test_raises_runtime_error_when_ffmpeg_fails(self, mocker):
        completed = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"ffmpeg: invalid timestamp",
        )
        mocker.patch(
            "pipeline.frame_extractor.subprocess.run",
            return_value=completed,
        )

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            extract_frame_at(Path("/videos/sample.mp4"), timestamp_s=999.0)

    def test_raises_runtime_error_when_ffmpeg_returns_empty_stdout(self, mocker):
        completed = MagicMock(returncode=0, stdout=b"", stderr=b"")
        mocker.patch(
            "pipeline.frame_extractor.subprocess.run",
            return_value=completed,
        )

        with pytest.raises(RuntimeError, match="no frame data"):
            extract_frame_at(Path("/videos/sample.mp4"), timestamp_s=5.0)

    def test_raises_runtime_error_when_ffmpeg_binary_missing(self, mocker):
        mocker.patch(
            "pipeline.frame_extractor.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg"),
        )

        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            extract_frame_at(Path("/videos/sample.mp4"), timestamp_s=5.0)
