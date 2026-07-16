"""Tests for the platform-delivered runtime-config applier (issue #85).

The platform lets an admin edit per-appliance runtime config and ships it in
the ``settings`` block of both the register and heartbeat responses. This
module owns the "apply on-change" logic: it diffs the incoming block against
the values it last applied and pushes only the changed ones into the box's
live consumers (buffer retention, poll interval, heartbeat interval, upload
chunk size). The four values seed from env as a cold-start fallback; a
platform value wins once received.

The applier is a pure unit — its collaborators (buffer / poller / uploader)
are injected as plain setter callbacks, so every test asserts behavior
through the public :meth:`RuntimeConfig.apply` surface without touching the
real sub-objects or any HTTP.
"""

from __future__ import annotations

from client_agent.runtime_config import RuntimeConfig


def _config(**setters):  # noqa: ANN002, ANN202
    """A RuntimeConfig seeded with distinct cold-start values per field so a
    test can tell which key an ``apply`` touched. Setter callbacks default to
    no-ops; a test passes recorders for the keys it cares about."""
    return RuntimeConfig(
        buffer_hours=1,
        polling_interval_seconds=5,
        heartbeat_interval_seconds=30,
        upload_chunk_bytes=52_428_800,
        **setters,
    )


# ----- 1. tracer bullet: a changed value updates state + pushes the setter -----


def test_apply_changed_buffer_hours_updates_value_and_calls_setter() -> None:
    """The core contract: a platform ``settings`` block with a new
    ``buffer_hours`` updates the live attribute (read by the heartbeat/loops)
    AND pushes the new value through the wired setter (the buffer keeps its
    own copy). The returned delta names what changed."""
    pushed: list[int] = []
    config = _config(set_buffer_hours=pushed.append)

    applied = config.apply({"buffer_hours": 8})

    assert config.buffer_hours == 8
    assert pushed == [8]
    assert applied == {"buffer_hours": 8}


# ----- 2. all four wire keys map to their attribute + setter -----


def test_apply_maps_all_four_keys() -> None:
    """Each of the four settings lands on its own attribute and, where a
    sub-object keeps a copy, its setter. ``heartbeat_interval_seconds`` has
    no sub-object — the heartbeat loop reads the attribute live — so it
    updates state without a setter call."""
    calls: dict[str, list[int]] = {"buffer": [], "poll": [], "upload": []}
    config = RuntimeConfig(
        buffer_hours=1,
        polling_interval_seconds=5,
        heartbeat_interval_seconds=30,
        upload_chunk_bytes=52_428_800,
        set_buffer_hours=calls["buffer"].append,
        set_polling_interval_seconds=calls["poll"].append,
        set_upload_chunk_bytes=calls["upload"].append,
    )

    applied = config.apply(
        {
            "buffer_hours": 8,
            "polling_interval_seconds": 7,
            "heartbeat_interval_seconds": 45,
            "upload_chunk_bytes": 10_485_760,
        }
    )

    assert config.buffer_hours == 8
    assert config.polling_interval_seconds == 7
    assert config.heartbeat_interval_seconds == 45
    assert config.upload_chunk_bytes == 10_485_760
    assert calls == {"buffer": [8], "poll": [7], "upload": [10_485_760]}
    assert applied == {
        "buffer_hours": 8,
        "polling_interval_seconds": 7,
        "heartbeat_interval_seconds": 45,
        "upload_chunk_bytes": 10_485_760,
    }


# ----- 3. on-change only: an unchanged block re-pushes nothing -----


def test_apply_same_block_twice_is_noop_second_time() -> None:
    """``settings`` arrives on every heartbeat, so the applier must diff
    against what it last applied. The first apply pushes; a byte-identical
    second apply pushes nothing and reports an empty delta — no churn (#85)."""
    pushed: list[int] = []
    config = _config(set_buffer_hours=pushed.append)
    block = {"buffer_hours": 8, "polling_interval_seconds": 5}

    first = config.apply(block)
    second = config.apply(block)

    assert first == {"buffer_hours": 8}  # poll unchanged from seed → not in delta
    assert second == {}
    assert pushed == [8]  # setter fired exactly once


def test_apply_only_pushes_the_key_that_changed() -> None:
    """A heartbeat that changes one setting (admin bumped buffer_hours) must
    not disturb the other three: only the changed key's setter fires."""
    pushed: dict[str, list[int]] = {"buffer": [], "poll": []}
    config = RuntimeConfig(
        buffer_hours=1,
        polling_interval_seconds=5,
        heartbeat_interval_seconds=30,
        upload_chunk_bytes=52_428_800,
        set_buffer_hours=pushed["buffer"].append,
        set_polling_interval_seconds=pushed["poll"].append,
    )
    config.apply({"buffer_hours": 8, "polling_interval_seconds": 5})

    applied = config.apply({"buffer_hours": 8, "polling_interval_seconds": 9})

    assert applied == {"polling_interval_seconds": 9}
    assert pushed == {"buffer": [8], "poll": [9]}


# ----- 4. garbage values are ignored, never crash the heartbeat thread -----


def test_apply_ignores_invalid_values() -> None:
    """``apply`` runs inside the heartbeat loop's ``try`` on the daemon
    thread; a malformed platform value must be dropped, not raised, or one
    bad edit wedges every future beat. Non-positive / non-int values leave
    state untouched and fire no setter (#85)."""
    pushed: list[int] = []
    config = _config(set_buffer_hours=pushed.append)

    for bad in ("eight", 0, -3, None, True, 1.5, ""):
        applied = config.apply({"buffer_hours": bad})
        assert applied == {}, f"expected {bad!r} to be rejected"

    assert config.buffer_hours == 1  # seed untouched
    assert pushed == []


def test_apply_valid_key_survives_invalid_sibling() -> None:
    """A block that mixes a good and a bad value applies the good one — a
    single garbage field must not poison the whole block (#85)."""
    config = _config()

    applied = config.apply({"buffer_hours": 8, "polling_interval_seconds": -1})

    assert applied == {"buffer_hours": 8}
    assert config.buffer_hours == 8
    assert config.polling_interval_seconds == 5  # bad value rejected, seed kept


# ----- 5. absent settings (None / empty) is a clean no-op -----


def test_apply_none_or_empty_is_noop() -> None:
    """``PlatformClient`` returns ``settings=None`` when the platform predates
    the feature or a beat carries only camera reconciliation. The applier must
    treat that (and an empty block) as "keep current values" — the env
    cold-start seeds stand (#85)."""
    pushed: list[int] = []
    config = _config(set_buffer_hours=pushed.append)

    assert config.apply(None) == {}
    assert config.apply({}) == {}
    assert config.buffer_hours == 1
    assert pushed == []


# ----- 6. wire(): late-bind setters after the sub-objects are built -----


def test_wire_late_binds_setters() -> None:
    """The buffer / poller / uploader are constructed *after* the boot-time
    register/heartbeat apply (they read the post-apply values as their seed),
    so their setters can't be passed at RuntimeConfig construction. ``wire``
    registers them afterwards; a subsequent apply pushes through them (#85)."""
    config = _config()  # constructed with no setters
    poll_pushed: list[int] = []
    upload_pushed: list[int] = []

    config.wire(
        set_polling_interval_seconds=poll_pushed.append,
        set_upload_chunk_bytes=upload_pushed.append,
    )
    config.apply({"polling_interval_seconds": 9, "upload_chunk_bytes": 10_485_760})

    assert poll_pushed == [9]
    assert upload_pushed == [10_485_760]


def test_wire_only_replaces_named_setters() -> None:
    """A caller that wires just one consumer must not clear the others — the
    poller thread and maintenance thread wire separately, in either order."""
    buf_pushed: list[int] = []
    poll_pushed: list[int] = []
    config = _config(set_buffer_hours=buf_pushed.append)

    config.wire(set_polling_interval_seconds=poll_pushed.append)
    config.apply({"buffer_hours": 8, "polling_interval_seconds": 9})

    assert buf_pushed == [8]  # earlier-wired setter still fires
    assert poll_pushed == [9]
