"""Repo-level meta tests for build configuration.

These tests assert the shape of `pyproject.toml` and `Makefile` so the project's
install story stays predictable: a default `uv sync` must remain lightweight
(no ~1.5GB CUDA wheels), GPU runtime must be opt-in via `--extra gpu`, and a
CPU-only stub must exist for unit tests on macOS / dev boxes without NVIDIA.

See GitHub issue #9 for the rationale.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
MAKEFILE = REPO_ROOT / "Makefile"
CLIENT_AGENT_DOCKERFILE = REPO_ROOT / "client-agent" / "Dockerfile"
GPU_SERVICE_DOCKERFILE = REPO_ROOT / "gpu-service" / "Dockerfile"
SETUP_MODELS_SCRIPT = REPO_ROOT / "setup-models.sh"
TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"


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


class TestClientAgentDockerfile:
    """Static smoke test for the client-agent container image (issue #7).

    Asserts the Dockerfile still:

    * exists at the expected path,
    * declares ``EXPOSE 8080`` so ``docker compose -f docker-compose.client.yml
      up`` actually publishes the Flask UI port,
    * launches the real entrypoint module (``client_agent.agent``) — the
      module that wires R2 creds → :class:`R2Client` → Flask ``app.run``.

    A real "does it boot" check would need a docker daemon, which CI on the
    macOS box doesn't have. These static assertions are the next best thing:
    they fail fast in unit tests if anyone (a) drops the EXPOSE directive,
    (b) reverts the entrypoint to the old SIGTERM-blocking placeholder, or
    (c) renames the Python module without updating the image config.
    """

    def test_client_agent_dockerfile_exists(self):
        assert CLIENT_AGENT_DOCKERFILE.exists(), (
            f"client-agent Dockerfile is missing at {CLIENT_AGENT_DOCKERFILE}. "
            f"Issue #16 requires this file; issue #7 depends on it shipping the Flask UI."
        )

    def test_client_agent_dockerfile_exposes_port_8080(self):
        """Acceptance criterion (#7): UI reachable on :8080. Without an
        ``EXPOSE 8080`` directive the docker-compose port mapping has no
        target to bind to and the container appears 'healthy' but unreachable."""
        text = CLIENT_AGENT_DOCKERFILE.read_text()
        # Match `EXPOSE 8080` on its own line, tolerating extra ports listed
        # alongside it (e.g. `EXPOSE 8080 9090`) and trailing whitespace.
        directives = [
            line.strip() for line in text.splitlines() if line.strip().upper().startswith("EXPOSE")
        ]
        assert directives, (
            "client-agent Dockerfile has no EXPOSE directive — port 8080 "
            "(Flask UI from issue #7) must be published."
        )
        ports: set[str] = set()
        for d in directives:
            ports.update(d.split()[1:])
        assert "8080" in ports, (
            f"client-agent Dockerfile must EXPOSE 8080 (Flask UI port from "
            f"issue #7), got EXPOSE directives: {directives}"
        )

    def test_client_agent_dockerfile_entrypoint_runs_real_module(self):
        """The ENTRYPOINT must invoke ``client_agent.agent`` — not a shell
        sleep, not a SIGTERM blocker. If anyone reverts to a placeholder,
        the GHCR image would build cleanly but never serve Flask, and that
        regression should fail this unit test in CI rather than only being
        caught by a human pulling the image and running it."""
        text = CLIENT_AGENT_DOCKERFILE.read_text()
        entrypoint_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().upper().startswith("ENTRYPOINT")
        ]
        assert entrypoint_lines, "client-agent Dockerfile has no ENTRYPOINT"
        joined = " ".join(entrypoint_lines)
        assert "client_agent.agent" in joined, (
            f"client-agent ENTRYPOINT must launch the real `client_agent.agent` "
            f"module (Flask UI on :8080, issue #7), got: {entrypoint_lines}"
        )


class TestApplianceDockerCompose:
    """Static smoke test for ``docker-compose.appliance.yml`` (issue #30).

    Two compose flavours coexist by design (issue #29 retires the legacy
    one later):

    * ``docker-compose.client.yml`` — legacy Phase 1-3 Docker UI mode,
      entrypoint ``client_agent.agent`` (Flask UI :8080, manual recordings).
    * ``docker-compose.appliance.yml`` — Phase 4 platform mode, entrypoint
      ``client_agent.appliance`` (TaskPoller + heartbeat + buffer trim).

    These assertions guard the *appliance* file's shape — they fail if the
    entrypoint reverts to the legacy module, if a required env var
    disappears, or if the buffer volume is dropped (would lose every chunk
    on container restart). Booting the image would catch the regressions
    too, but only after the operator wonders why the appliance won't pick
    up tasks.
    """

    COMPOSE_FILE = REPO_ROOT / "docker-compose.appliance.yml"

    def _load(self) -> dict:
        import yaml  # PyYAML ships transitively via wsdiscovery / other deps

        with self.COMPOSE_FILE.open("r") as f:
            return yaml.safe_load(f)

    def test_compose_file_exists(self):
        assert self.COMPOSE_FILE.exists(), (
            f"docker-compose.appliance.yml missing at {self.COMPOSE_FILE}. "
            f"Issue #30 requires this file to support Phase 4 platform mode."
        )

    def test_compose_file_parses_as_yaml(self):
        data = self._load()
        assert isinstance(data, dict), "compose file must parse to a top-level mapping"
        assert "services" in data, "compose file has no `services:` section"

    def test_compose_service_uses_appliance_entrypoint(self):
        """Entrypoint must invoke ``client_agent.appliance`` — not the
        legacy ``client_agent.agent``. Reverting to the legacy module
        would silently skip the platform integration and the task poller,
        so the appliance container would look healthy but never consume
        a single task from the queue."""
        data = self._load()
        services = data["services"]
        assert services, "no services defined"
        # We accept any service name (operator preference); contract is
        # *at least one* runs the appliance module.
        bodies = list(services.values())
        joined = " ".join(
            " ".join(map(str, body.get("command", []) or []))
            + " "
            + " ".join(map(str, body.get("entrypoint", []) or []))
            for body in bodies
        )
        assert "client_agent.appliance" in joined, (
            f"no service launches `client_agent.appliance` "
            f"(found: {[(b.get('entrypoint'), b.get('command')) for b in bodies]})"
        )

    def test_compose_requires_platform_url_and_token(self):
        """Operator must supply ``PLATFORM_URL`` + ``APPLIANCE_TOKEN`` via
        env interpolation. Without the ``${VAR:?error}`` form (or env_file)
        the compose stack would boot with empty strings and silently fall
        back to legacy mode — defeating the whole point of the appliance
        compose file."""
        text = self.COMPOSE_FILE.read_text()
        for key in ("PLATFORM_URL", "APPLIANCE_TOKEN"):
            assert key in text, f"compose file does not reference {key}"

    def test_compose_declares_buffer_hours(self):
        """``BUFFER_HOURS`` must surface in the compose env so the operator
        can override the 1h default without rebuilding the image."""
        text = self.COMPOSE_FILE.read_text()
        assert "BUFFER_HOURS" in text, "compose file does not reference BUFFER_HOURS"

    def test_compose_persists_buffer_volume(self):
        """Per-camera chunks land in ``BUFFER_DIR`` (defaults under the
        recordings root in :mod:`client_agent.appliance`). The compose
        file must bind a volume there or every restart wipes the rolling
        buffer — every task would fail with "buffer empty" until each
        camera repopulates."""
        data = self._load()
        services = data["services"]
        has_volume = False
        for body in services.values():
            vols = body.get("volumes") or []
            for v in vols:
                if isinstance(v, str) and (
                    "buffer" in v.lower() or "cctv-buffer" in v or "/var/lib/cctv" in v
                ):
                    has_volume = True
                    break
                if isinstance(v, dict) and "target" in v:
                    if "buffer" in v["target"].lower():
                        has_volume = True
                        break
        assert has_volume, (
            "compose file does not bind a volume for the rolling buffer — "
            "every container restart would lose buffered footage"
        )

    def test_compose_defines_healthcheck(self):
        """A healthcheck on :8080 lets ``docker compose ps`` surface the
        appliance state (matches issue #30 AC: 'healthcheck OK after 30s').
        Operators rely on this to know whether the appliance crashed
        between heartbeats."""
        data = self._load()
        services = data["services"]
        has_healthcheck = any("healthcheck" in body for body in services.values())
        assert has_healthcheck, "no service defines a healthcheck"


class TestGpuServiceDockerfileBundlesModel:
    """Static smoke test for the gpu-service Dockerfile bundling the
    default YOLO pose ONNX model (issue #31).

    Before the fix, ``gpu_service.rest_server`` crashed on boot when run
    without the host bind-mount because ``/app/models/yolo11s-pose.onnx``
    did not exist inside the image. The legacy compose worker only "worked"
    because ``docker-compose.yml`` mounts ``./models:/app/models:ro``; the
    REST contract (#25) used by gpu-agent has no such bind-mount.

    These assertions guard the Dockerfile shape — they would catch a
    revert that drops the model-fetching RUN step, removes the sha256
    pin (silent model swap risk), or changes ``ENV MODEL_PATH`` to a
    path the Dockerfile does not materialize. The real "image actually
    boots in REST mode" check happens on cctv-vps after push.
    """

    DEFAULT_MODEL_FILENAME = "yolo11s-pose.onnx"
    DEFAULT_MODEL_PATH_IN_IMAGE = f"/app/models/{DEFAULT_MODEL_FILENAME}"

    @staticmethod
    def _dockerfile_text() -> str:
        return GPU_SERVICE_DOCKERFILE.read_text()

    def test_dockerfile_materializes_default_model_at_model_path(self):
        """The image must ship ``/app/models/yolo11s-pose.onnx`` so that
        REST mode (``python -m gpu_service.rest_server``) can boot
        without any host bind-mount. Either ``COPY`` or ``RUN curl``
        is acceptable — both produce the same end-state inside the image."""
        text = self._dockerfile_text()

        # Pattern 1: explicit COPY of the model file (preflight approach).
        copy_pattern = re.compile(
            r"^\s*COPY\s+\S*" + re.escape(self.DEFAULT_MODEL_FILENAME),
            re.MULTILINE,
        )
        # Pattern 2: RUN step that downloads the model into /app/models/.
        # Match a RUN line (possibly multi-line via backslash continuation)
        # that mentions the model filename — that is the build-time fetch.
        run_pattern = re.compile(
            r"RUN[^\n]*(?:\\\n[^\n]*)*" + re.escape(self.DEFAULT_MODEL_FILENAME),
            re.DOTALL,
        )

        has_copy = bool(copy_pattern.search(text))
        has_run = bool(run_pattern.search(text))

        assert has_copy or has_run, (
            f"gpu-service/Dockerfile does not materialize "
            f"{self.DEFAULT_MODEL_PATH_IN_IMAGE} during build. Issue #31: REST "
            f"mode boots without the docker-compose bind-mount, so the model "
            f"must be baked into the image (either `COPY models/{self.DEFAULT_MODEL_FILENAME} ...` "
            f"or `RUN curl -fL <url> -o {self.DEFAULT_MODEL_PATH_IN_IMAGE}`)."
        )

    def test_dockerfile_sha256_matches_setup_models_script(self):
        """Local dev (``./setup-models.sh``) and Docker build must pin the
        SAME sha256, otherwise the gpu-service image and a local
        ``make test-gpu`` would run inference with different weights — a
        nightmare to debug when accuracy regressions appear only in one
        environment. The two pins are physically separate (Dockerfile ARG
        and shell variable), so we cross-check them here."""
        dockerfile = self._dockerfile_text()
        script = SETUP_MODELS_SCRIPT.read_text()

        # Pull the sha256 hex out of the Dockerfile ARG. Matches:
        #   ARG YOLO_MODEL_SHA256=469beac503fdc788ea3980331bc4bfbd2bd00de3772eb0984f4c53032740583f
        dockerfile_pin = re.search(r"ARG\s+YOLO_MODEL_SHA256\s*=\s*([0-9a-fA-F]{64})", dockerfile)
        assert dockerfile_pin, (
            "gpu-service/Dockerfile has no `ARG YOLO_MODEL_SHA256=<hex>` line — "
            "the build_config_test relies on this ARG to cross-check the pin "
            "against setup-models.sh. Either add the ARG or update the test."
        )

        # Pull the default sha256 from setup-models.sh. Matches lines like:
        #   MODEL_SHA256="${MODEL_SHA256:-<64-hex>}"
        script_pin = re.search(r'MODEL_SHA256="\$\{MODEL_SHA256:-([0-9a-fA-F]{64})\}"', script)
        assert script_pin, (
            "setup-models.sh has no parseable `MODEL_SHA256:-<hex>` default. "
            "If you changed the variable convention, update this test."
        )

        assert dockerfile_pin.group(1).lower() == script_pin.group(1).lower(), (
            f"sha256 drift between Dockerfile and setup-models.sh:\n"
            f"  Dockerfile ARG: {dockerfile_pin.group(1)}\n"
            f"  setup-models.sh: {script_pin.group(1)}\n"
            f"Both pins must point at the same canonical weights — bump them "
            f"together when releasing a new yolo11s-pose-vN.0 tag."
        )

    def test_dockerfile_verifies_sha256_during_build(self):
        """The model-fetch RUN step must verify a sha256 checksum, so a
        compromised / accidentally re-uploaded GH release asset fails the
        build instead of silently substituting weights. Same defense-in-depth
        rationale as ``setup-models.sh`` for local installs.

        Allows either ``sha256sum -c`` (GNU coreutils, present on Ubuntu base)
        or ``shasum -a 256 -c`` (BSD, present on macOS — irrelevant inside the
        nvidia/cuda Ubuntu image but cheap to accept here)."""
        text = self._dockerfile_text()

        sha256_pattern = re.compile(
            r"(sha256sum|shasum\s+-a\s+256)\s+-c\b",
        )
        assert sha256_pattern.search(text), (
            "gpu-service/Dockerfile materializes the model but does not "
            "verify its sha256. Add `... | sha256sum -c -` (or `shasum -a "
            "256 -c -`) to the RUN step so a silent model swap on the GH "
            "release fails the build instead of shipping different weights "
            "than setup-models.sh produces."
        )

    def test_model_path_env_var_targets_bundled_file(self):
        """``ENV MODEL_PATH`` is what ``gpu_service.rest_server`` reads at
        boot to find the ONNX model (rest_server.py:111). If someone moves
        the bundled model to a different path inside the image but forgets
        to update the ENV (or vice versa), the REST entrypoint resurrects
        exactly the NoSuchFile crash this issue was filed to prevent.

        Asserts the ``MODEL_PATH`` value equals the canonical bundled path."""
        text = self._dockerfile_text()

        # Match `ENV MODEL_PATH=...` whether it's on its own line or part of
        # a multi-line ENV block (`ENV FOO=bar \` continuation).
        env_match = re.search(
            r"\bMODEL_PATH\s*=\s*(\S+)",
            text,
        )
        assert env_match, (
            "gpu-service/Dockerfile has no `MODEL_PATH=...` env assignment — "
            "rest_server.py:111 reads this to find the ONNX model. Without "
            "it, REST mode falls back to the hardcoded default which may "
            "diverge from the path the Dockerfile materializes."
        )
        configured_path = env_match.group(1).rstrip("\\").strip()
        assert configured_path == self.DEFAULT_MODEL_PATH_IN_IMAGE, (
            f"ENV MODEL_PATH={configured_path!r} does not match the path the "
            f"Dockerfile materializes ({self.DEFAULT_MODEL_PATH_IN_IMAGE!r}). "
            f"Either fix the RUN step to download to MODEL_PATH or update "
            f"MODEL_PATH to point at the bundled file. Mismatch reintroduces "
            f"issue #31's NoSuchFile crash."
        )


class TestRuffJobScope:
    """Issue #15: the CI ruff job must lint the same Python source tree that
    the local pre-commit hook covers, otherwise a contributor who bypasses
    the hook (or doesn't have it installed) can land brokenness in
    ``gpu-service/`` / ``client-agent/`` / ``tests/`` / ``test/`` and CI
    won't catch it. Pre-commit runs unrestricted on every tracked file;
    the CI job historically only checked ``pipeline/``.

    These assertions parse ``.github/workflows/tests.yml`` textually rather
    than pulling in PyYAML — the workflow file is small, the assertions
    only care about the ``run:`` lines that invoke ruff, and the meta-test
    suite is otherwise dependency-free.
    """

    # Every directory that contains first-party Python the pre-commit hook
    # would lint. Keep in sync with the project layout — if a new top-level
    # Python package lands, add it here AND to tests.yml.
    EXPECTED_SCOPE = {
        "pipeline/",
        "tests/",
        "test/",
        "gpu-service/",
        "client-agent/",
    }

    @staticmethod
    def _ruff_invocations() -> list[str]:
        """Return the full command strings of every `uv run ruff ...` step
        in the workflow. Order preserved so callers can distinguish the
        ``ruff check`` step from the ``ruff format --check`` step."""
        text = TESTS_WORKFLOW.read_text()
        cmds: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # Step recipes look like:  `run: uv run ruff check pipeline/`
            if stripped.startswith("run:") and "uv run ruff" in stripped:
                cmds.append(stripped[len("run:") :].strip())
        return cmds

    def test_workflow_file_exists(self):
        assert TESTS_WORKFLOW.exists(), (
            f"GitHub Actions workflow missing at {TESTS_WORKFLOW}. "
            f"Issue #15 requires a `ruff` job in this file."
        )

    def test_ruff_check_step_covers_full_python_scope(self):
        cmds = self._ruff_invocations()
        check_cmds = [c for c in cmds if "ruff check" in c and "format" not in c]
        assert check_cmds, (
            "tests.yml has no `uv run ruff check` step — issue #15 requires "
            "the CI ruff job to run lint."
        )
        # If multiple `ruff check` steps exist (e.g. one per dir), the union
        # of their args must cover the full scope.
        covered = " ".join(check_cmds)
        missing = sorted(d for d in self.EXPECTED_SCOPE if d not in covered)
        assert not missing, (
            f"`uv run ruff check` in tests.yml does not cover all first-party "
            f"Python directories. Missing: {missing}. Pre-commit lints these "
            f"locally; CI must too (issue #15). Got command(s): {check_cmds}"
        )

    def test_ruff_format_check_step_covers_full_python_scope(self):
        cmds = self._ruff_invocations()
        format_cmds = [c for c in cmds if "ruff format" in c]
        assert format_cmds, (
            "tests.yml has no `uv run ruff format --check` step — issue #15 "
            "requires the CI ruff job to enforce formatting."
        )
        for cmd in format_cmds:
            assert "--check" in cmd, (
                f"ruff format step must use --check (read-only) in CI, "
                f"never rewrite files on the runner. Got: {cmd!r}"
            )
        covered = " ".join(format_cmds)
        missing = sorted(d for d in self.EXPECTED_SCOPE if d not in covered)
        assert not missing, (
            f"`uv run ruff format --check` in tests.yml does not cover all "
            f"first-party Python directories. Missing: {missing}. Pre-commit "
            f"formats these locally; CI must verify (issue #15). Got "
            f"command(s): {format_cmds}"
        )
