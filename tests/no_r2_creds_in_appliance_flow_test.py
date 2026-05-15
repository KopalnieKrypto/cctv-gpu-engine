"""Meta-AC for issue #28: appliance-flow code never touches R2 credentials.

Per DD-09 the appliance-task flow (platform → poller → uploader → R2) must
hold zero direct R2 credentials — every byte to R2 goes through a presigned
PUT URL issued on demand by the platform. This test grep's the four files
that own that flow and asserts none of the obvious credential identifiers
appear in source (comments, docstrings, env-var reads, anything).

**Scope is intentionally narrow.** The legacy Docker UI flow (``agent.py``,
``web.py``, ``recorder.py``) still uses direct R2 creds for one-shot
uploads — cleaning that up is tracked separately (#29 follow-up, deferred
until after the Phase-4 demo so we don't break the current Docker users).
If a future slice consolidates all uploads through the platform, the file
list below should grow accordingly."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that participate in the platform-mediated upload flow. These must
# not mention R2 credentials in any form — token-only auth against the
# platform, presigned URLs for the actual R2 byte transfer.
APPLIANCE_FLOW_FILES = [
    REPO_ROOT / "client-agent" / "client_agent" / "platform.py",
    REPO_ROOT / "client-agent" / "client_agent" / "poller.py",
    REPO_ROOT / "client-agent" / "client_agent" / "uploader.py",
    REPO_ROOT / "client-agent" / "client_agent" / "buffer.py",
]

# Patterns from the issue's grep AC, plus the boto3 client-constructor
# kwargs that would smuggle the same secrets under different names.
FORBIDDEN_PATTERN = re.compile(
    r"R2_ACCESS_KEY|R2_SECRET|access_key_id|secret_access_key",
    re.IGNORECASE,
)


def test_appliance_flow_files_contain_no_r2_credentials() -> None:
    """The four appliance-flow modules must be credential-free.

    Failure here means a future edit re-introduced direct R2 creds into
    the platform-mediated path — likely an accidental import from the
    legacy Docker flow. The fix is to route through ``PresignedUploader``,
    not to add the credential to this allowlist."""
    offenders: list[tuple[Path, int, str]] = []
    for path in APPLIANCE_FLOW_FILES:
        assert path.exists(), f"appliance-flow file missing: {path}"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN_PATTERN.search(line):
                offenders.append((path.relative_to(REPO_ROOT), lineno, line))

    assert offenders == [], "R2 credential strings found in appliance-flow files:\n" + "\n".join(
        f"  {p}:{ln}: {text.strip()}" for p, ln, text in offenders
    )
