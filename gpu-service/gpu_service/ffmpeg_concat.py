"""Multi-chunk MP4 concat via ffmpeg demuxer (issue #25 AC #4).

The gpu-agent splits a long surveillance recording into chunks for parallel
R2 upload. We stitch them back together before pose inference. Stream
copy (``-c copy``) is mandatory — re-encoding would destroy frame
timestamps and silently change pose-detection inputs vs. what the operator
recorded, on top of being 20x slower than copy.

The ``runner`` parameter is injected so tests can capture the argv without
spawning ffmpeg. The production default is :func:`subprocess.run`.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class _ResultLike(Protocol):
    returncode: int
    stderr: bytes


Runner = Callable[..., _ResultLike]


def _default_runner(argv: list[str], **kwargs: Any) -> _ResultLike:
    return subprocess.run(argv, capture_output=True, check=False, **kwargs)


def ffmpeg_concat(
    inputs: list[Path],
    output: Path,
    *,
    runner: Runner = _default_runner,
) -> None:
    """Concatenate ``inputs`` (in order) into ``output`` via ffmpeg.

    The concat demuxer requires a listfile of ``file '<path>'`` lines. We
    write it next to the output so a failed run leaves both the listfile
    and ffmpeg stderr in place for debugging.
    """
    listfile = output.parent / f"{output.stem}.concat-list.txt"
    listfile.write_text("".join(f"file '{p}'\n" for p in inputs))

    argv = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listfile),
        "-c",
        "copy",
        str(output),
    ]

    result = runner(argv)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg concat exited {result.returncode}: {stderr}")
