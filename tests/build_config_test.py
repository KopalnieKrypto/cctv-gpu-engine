"""Repo-level meta tests for build configuration.

These tests assert the shape of `pyproject.toml` and `Makefile` so the project's
install story stays predictable: a default `uv sync` must remain lightweight
(no ~1.5GB CUDA wheels), GPU runtime must be opt-in via `--extra gpu`, and a
CPU-only stub must exist for unit tests on macOS / dev boxes without NVIDIA.

See GitHub issue #9 for the rationale.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
MAKEFILE = REPO_ROOT / "Makefile"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _dep_names(specifiers: list[str]) -> list[str]:
    """Extract bare package names from PEP 508 specifier strings.

    "onnxruntime-gpu[cuda,cudnn]>=1.17 ; sys_platform == 'linux'" -> "onnxruntime-gpu"
    """
    names = []
    for spec in specifiers:
        # strip environment marker
        head = spec.split(";", 1)[0].strip()
        # strip extras
        head = head.split("[", 1)[0].strip()
        # strip version specifier
        for op in (">=", "<=", "==", "!=", "~=", ">", "<"):
            if op in head:
                head = head.split(op, 1)[0].strip()
                break
        names.append(head)
    return names


def _has_linux_marker(specifier: str) -> bool:
    """True iff the PEP 508 spec is gated on `sys_platform == 'linux'`."""
    if ";" not in specifier:
        return False
    marker = specifier.split(";", 1)[1].strip()
    # accept both single and double quotes; tolerate whitespace variations
    normalized = marker.replace('"', "'").replace(" ", "")
    return "sys_platform=='linux'" in normalized


def _makefile_target_recipe(target: str) -> str | None:
    """Return the recipe lines for a given Makefile target, or None if absent.

    Parses simple Makefiles only — recognizes `target:` at column 0, then any
    indented (tab) lines until the next blank line / next target.
    """
    if not MAKEFILE.exists():
        return None
    text = MAKEFILE.read_text()
    lines = text.splitlines()
    in_target = False
    recipe: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        # Match target line: `name:` or `name: deps`. Variable assignments
        # `FOO := bar` are not targets — they don't start with whitespace and
        # contain `=`, but a target line never has `=` before the colon.
        if not line.startswith((" ", "\t")) and ":" in line and "=" not in line.split(":", 1)[0]:
            head = line.split(":", 1)[0].strip()
            if in_target:
                break  # next target ends our match
            if head == target:
                in_target = True
                continue
        elif in_target:
            if line.startswith(("\t", " ")) and stripped:
                recipe.append(stripped)
            elif stripped == "":
                # blank line ends the recipe
                break
    return "\n".join(recipe) if in_target else None


class TestMainDependenciesAreLightweight:
    """A bare `uv sync` must not pull onnxruntime / CUDA wheels.

    Acceptance criterion from issue #9: default install on macOS < 100MB.
    """

    def test_no_onnxruntime_in_main_dependencies(self):
        pyproject = _load_pyproject()
        deps = _dep_names(pyproject["project"]["dependencies"])

        forbidden = {"onnxruntime", "onnxruntime-gpu", "nvidia-cublas-cu12"}
        leaked = forbidden & set(deps)

        assert not leaked, (
            f"Heavy GPU/runtime packages must not live in [project].dependencies; "
            f"move them to [project.optional-dependencies]. Leaked: {sorted(leaked)}"
        )


class TestGpuExtra:
    """`uv sync --extra gpu` must install the real GPU runtime, Linux-only."""

    def test_gpu_extra_contains_required_packages_with_linux_marker(self):
        pyproject = _load_pyproject()
        gpu_specs: list[str] = pyproject["project"].get("optional-dependencies", {}).get("gpu", [])

        assert gpu_specs, (
            "[project.optional-dependencies].gpu is missing or empty — "
            "this is the opt-in GPU install path documented in issue #9"
        )

        names = set(_dep_names(gpu_specs))
        required = {"onnxruntime-gpu", "nvidia-cublas-cu12"}
        missing = required - names
        assert not missing, (
            f"gpu extra is missing required packages: {sorted(missing)}. "
            f"nvidia-cublas-cu12 is mandatory because the [cuda,cudnn] extras "
            f"do NOT include cublas — without it, onnxruntime silently falls "
            f"back to CPU at runtime."
        )

        # Both packages must be Linux-gated: macOS users running `uv sync --extra
        # gpu` should resolve to a no-op for these wheels, not crash on missing
        # platform-specific binaries.
        for spec in gpu_specs:
            name = _dep_names([spec])[0]
            if name in required:
                assert _has_linux_marker(spec), (
                    f"gpu extra entry {spec!r} must carry `; sys_platform == 'linux'` marker"
                )


class TestCpuStubExtra:
    """`uv sync --extra cpu-stub` must install plain onnxruntime so unit tests
    on macOS / dev boxes can `import onnxruntime` without pulling CUDA wheels.
    """

    def test_cpu_stub_extra_contains_onnxruntime(self):
        pyproject = _load_pyproject()
        cpu_stub_specs: list[str] = (
            pyproject["project"].get("optional-dependencies", {}).get("cpu-stub", [])
        )

        assert cpu_stub_specs, (
            "[project.optional-dependencies].cpu-stub is missing or empty — "
            "this is the macOS/dev install path documented in issue #9"
        )

        names = set(_dep_names(cpu_stub_specs))
        assert "onnxruntime" in names, (
            f"cpu-stub extra must contain plain `onnxruntime`, got: {sorted(names)}"
        )

    def test_cpu_stub_does_not_pull_gpu_packages(self):
        """cpu-stub must NOT contain onnxruntime-gpu or NVIDIA libs — that
        would defeat the purpose of having a separate lightweight extra."""
        pyproject = _load_pyproject()
        cpu_stub_specs: list[str] = (
            pyproject["project"].get("optional-dependencies", {}).get("cpu-stub", [])
        )

        names = set(_dep_names(cpu_stub_specs))
        forbidden = {"onnxruntime-gpu", "nvidia-cublas-cu12"}
        leaked = forbidden & names
        assert not leaked, (
            f"cpu-stub extra leaked GPU packages: {sorted(leaked)}. Move them to the `gpu` extra."
        )


class TestMakefileTargets:
    """Issue #9 acceptance criterion: Makefile must expose
    sync-dev / sync-gpu / test / test-gpu so contributors don't have to
    memorize the underlying `uv` invocations."""

    def test_makefile_exists(self):
        assert MAKEFILE.exists(), (
            f"Makefile is missing at {MAKEFILE}. Issue #9 requires a Makefile "
            f"with sync-dev/sync-gpu/test/test-gpu targets."
        )

    def test_sync_dev_target_runs_uv_sync_with_cpu_stub_extra(self):
        recipe = _makefile_target_recipe("sync-dev")
        assert recipe is not None, "Makefile is missing the `sync-dev` target"
        assert "uv sync" in recipe, f"sync-dev recipe must invoke `uv sync`, got:\n{recipe}"
        assert "--extra cpu-stub" in recipe, (
            f"sync-dev recipe must pass `--extra cpu-stub`, got:\n{recipe}"
        )

    def test_sync_gpu_target_runs_uv_sync_with_gpu_extra(self):
        recipe = _makefile_target_recipe("sync-gpu")
        assert recipe is not None, "Makefile is missing the `sync-gpu` target"
        assert "uv sync" in recipe, f"sync-gpu recipe must invoke `uv sync`, got:\n{recipe}"
        assert "--extra gpu" in recipe, f"sync-gpu recipe must pass `--extra gpu`, got:\n{recipe}"

    def test_test_target_runs_pytest(self):
        recipe = _makefile_target_recipe("test")
        assert recipe is not None, "Makefile is missing the `test` target"
        assert "pytest" in recipe, f"test recipe must invoke pytest, got:\n{recipe}"

    def test_test_gpu_target_runs_pipeline_analyze(self):
        """`test-gpu` is the end-to-end GPU smoke test invoked on cctv-vps after
        `make sync-gpu`. It must exercise the real CUDA inference path, so the
        recipe should call `pipeline.analyze` (the only entry point that loads
        the ONNX model with CUDAExecutionProvider)."""
        recipe = _makefile_target_recipe("test-gpu")
        assert recipe is not None, "Makefile is missing the `test-gpu` target"
        assert "pipeline.analyze" in recipe, (
            f"test-gpu recipe must run `pipeline.analyze` to actually exercise "
            f"the GPU code path, got:\n{recipe}"
        )
