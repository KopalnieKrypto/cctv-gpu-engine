"""Tests for the multi-chunk ffmpeg-concat helper (issue #25 AC #4).

Stream copy only — no re-encoding. Tested by capturing the subprocess
argv; an integration test under ``-m integration`` exercises real ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

from gpu_service.ffmpeg_concat import ffmpeg_concat


class TestFfmpegConcat:
    def test_runs_ffmpeg_concat_with_stream_copy(self, tmp_path) -> None:
        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for p in inputs:
            p.write_bytes(b"FAKE")
        out = tmp_path / "out.mp4"

        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["cwd"] = kwargs.get("cwd")
            # Simulate ffmpeg producing the output so the runner can read it.
            out.write_bytes(b"CONCATENATED")

            class _Result:
                returncode = 0
                stderr = b""

            return _Result()

        ffmpeg_concat(inputs, out, runner=fake_run)

        argv = captured["argv"]
        # Stream-copy: no -c:v / -c:a libx264 / aac (re-encoding) — `-c copy` only.
        assert argv[0] == "ffmpeg"
        assert "-f" in argv and "concat" in argv
        assert "-safe" in argv and "0" in argv
        assert "-c" in argv and "copy" in argv
        assert str(out) in argv
        # No re-encoding flags slipped in.
        forbidden = {"libx264", "libx265", "aac", "-crf"}
        assert not (set(argv) & forbidden)

    def test_writes_listfile_with_one_line_per_input(self, tmp_path) -> None:
        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
        for p in inputs:
            p.write_bytes(b"FAKE")
        out = tmp_path / "out.mp4"

        captured_listfile: dict = {}

        def fake_run(argv, **kwargs):
            # The -i argument points at a listfile we wrote. Read it.
            list_path = Path(argv[argv.index("-i") + 1])
            captured_listfile["content"] = list_path.read_text()
            out.write_bytes(b"x")

            class _Result:
                returncode = 0
                stderr = b""

            return _Result()

        ffmpeg_concat(inputs, out, runner=fake_run)

        # ffmpeg concat demuxer wants `file '<path>'` lines, one per input.
        lines = [ln for ln in captured_listfile["content"].splitlines() if ln.strip()]
        assert len(lines) == 3
        for ln, src in zip(lines, inputs, strict=True):
            assert ln == f"file '{src}'"

    def test_raises_when_ffmpeg_exits_non_zero(self, tmp_path) -> None:
        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for p in inputs:
            p.write_bytes(b"FAKE")
        out = tmp_path / "out.mp4"

        def fake_run(argv, **kwargs):
            class _Result:
                returncode = 1
                stderr = b"Invalid data found"

            return _Result()

        import pytest

        with pytest.raises(RuntimeError) as exc:
            ffmpeg_concat(inputs, out, runner=fake_run)

        assert "Invalid data found" in str(exc.value)
