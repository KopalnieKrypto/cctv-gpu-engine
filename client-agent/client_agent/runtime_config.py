"""Platform-delivered runtime-config applier (issue #85).

The ``gpu-exchange`` platform lets an admin edit per-appliance runtime config
and delivers it in the ``settings`` block of the register and heartbeat
responses (see ``docs/design/09-client-appliance.md §9.9`` in that repo).
This module is the box side: it holds the four settings as the single source
of truth and applies platform deltas **on change** so a value that arrives
unchanged on every heartbeat costs nothing.

    buffer_hours              → rolling-buffer retention window (trim cron)
    polling_interval_seconds  → task-poller inter-poll sleep
    heartbeat_interval_seconds→ heartbeat-loop sleep (read live off this obj)
    upload_chunk_bytes        → per-chunk upload size (store-only for now)

Precedence (issue #85): the four values seed from env as a cold-start
fallback; a platform value overrides once received. Three of them push into a
sub-object that keeps its own copy (buffer / poller / uploader) via injected
setter callbacks; ``heartbeat_interval_seconds`` has no sub-object — the
heartbeat loop reads :attr:`RuntimeConfig.heartbeat_interval_seconds` live
each tick.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

# Wire keys, in the order the platform documents them. Attribute names on
# RuntimeConfig match these exactly, so ``apply`` can ``getattr``/``setattr``
# by key without a translation table.
_WIRE_KEYS: tuple[str, ...] = (
    "buffer_hours",
    "polling_interval_seconds",
    "heartbeat_interval_seconds",
    "upload_chunk_bytes",
)


def _coerce_positive_int(value: object) -> int | None:
    """Return ``value`` as a positive ``int``, or ``None`` if it is not one.

    All four settings are positive integers (hours, seconds, bytes). ``bool``
    is rejected even though ``isinstance(True, int)`` holds — a boolean here is
    schema drift, not a count. Floats and numeric strings are rejected rather
    than truncated/parsed: the platform ships JSON ints, so anything else is a
    signal something is wrong upstream, and silently coercing it would hide the
    bug."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


class RuntimeConfig:
    """Single source of truth for the four platform-delivered settings."""

    def __init__(
        self,
        *,
        buffer_hours: int,
        polling_interval_seconds: int,
        heartbeat_interval_seconds: int,
        upload_chunk_bytes: int,
        set_buffer_hours: Callable[[int], None] | None = None,
        set_polling_interval_seconds: Callable[[int], None] | None = None,
        set_upload_chunk_bytes: Callable[[int], None] | None = None,
    ) -> None:
        self.buffer_hours = buffer_hours
        self.polling_interval_seconds = polling_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.upload_chunk_bytes = upload_chunk_bytes
        # heartbeat_interval_seconds has no setter: the heartbeat loop reads
        # the attribute live, so there is no sub-object copy to push into.
        self._setters: dict[str, Callable[[int], None] | None] = {
            "buffer_hours": set_buffer_hours,
            "polling_interval_seconds": set_polling_interval_seconds,
            "upload_chunk_bytes": set_upload_chunk_bytes,
        }

    def wire(
        self,
        *,
        set_buffer_hours: Callable[[int], None] | None = None,
        set_polling_interval_seconds: Callable[[int], None] | None = None,
        set_upload_chunk_bytes: Callable[[int], None] | None = None,
    ) -> None:
        """Late-bind consumer setters after the sub-objects exist (issue #85).

        The buffer / poller / uploader are built *after* the boot-time apply
        (they take the post-apply values as their construction seed), so their
        setters can't be passed to ``__init__``. The maintenance and poller
        threads each call ``wire`` for the consumers they own, in either order.
        Only the named setters are (re)bound; unnamed ones keep whatever was
        registered before, so two partial ``wire`` calls compose."""
        if set_buffer_hours is not None:
            self._setters["buffer_hours"] = set_buffer_hours
        if set_polling_interval_seconds is not None:
            self._setters["polling_interval_seconds"] = set_polling_interval_seconds
        if set_upload_chunk_bytes is not None:
            self._setters["upload_chunk_bytes"] = set_upload_chunk_bytes

    def apply(self, settings: Mapping[str, object] | None) -> dict[str, int]:
        """Apply a platform ``settings`` block on-change; return the deltas.

        For each known key present in ``settings`` whose value differs from
        the currently-applied one, update the live attribute and push it
        through the wired setter. Returns ``{key: new_value}`` for exactly
        the keys that changed (empty when nothing did) so the caller can log
        the delta."""
        applied: dict[str, int] = {}
        for key in _WIRE_KEYS:
            if settings is None or key not in settings:
                continue
            value = _coerce_positive_int(settings[key])
            if value is None:
                # A malformed platform value (admin typo, schema drift) must
                # not raise: apply() runs on the heartbeat daemon thread and a
                # single bad field would otherwise wedge every future beat.
                # Drop it, warn, keep the current value.
                logger.warning(
                    "runtime_config: ignoring invalid %s=%r (want positive int)",
                    key,
                    settings[key],
                )
                continue
            if getattr(self, key) == value:
                continue
            setattr(self, key, value)
            setter = self._setters.get(key)
            if setter is not None:
                setter(value)
            applied[key] = value
        return applied
