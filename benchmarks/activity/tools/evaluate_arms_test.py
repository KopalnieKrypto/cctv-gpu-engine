"""Contract tests for the C.0 scoring harness.

These pin the rules that decide whether a C.0 number means anything: an
unanswered sample is an error rather than a smaller denominator, the bar is
per-activity and never an average, and an arm that will not fit one card is
disqualified regardless of accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import evaluate_arms as ev  # noqa: E402


def _annotation(window: str, labels: list[str], stride: int = 2) -> dict:
    duration = len(labels) * stride
    samples = [{"t_s": i * stride, "activity_id": a} for i, a in enumerate(labels)]
    grid = {s["t_s"]: s["activity_id"] for s in samples}
    return {
        "window": window,
        "stride_s": stride,
        "duration_s": duration,
        "grid": grid,
        "intervals": ev._fold_to_intervals(grid, stride, duration),
        "count": len(samples),
    }


ACTIVITY_IDS = [
    "spawanie",
    "ukladanie_pretow",
    "sciaganie_elementu",
    "inna_czynnosc",
    "postoj",
    "brak_na_stanowisku",
    "nierozpoznane",
]


# The vocabulary the client accepted (#117): `spawanie`, `ukladanie_pretow`,
# and everything else on one line.
DELIVERY_MEMBERS = [
    "sciaganie_elementu",
    "inna_czynnosc",
    "postoj",
    "brak_na_stanowisku",
]


def _manifest(members: list[str], bucket: str = "pozostale") -> dict:
    return {
        "activities": [{"id": a} for a in ACTIVITY_IDS],
        "delivery_vocabulary": {"bucket": bucket, "members": members},
    }


class TestPerActivityScores:
    def test_bar_is_per_activity_not_an_average(self):
        # 18 correct `spawanie` and 2 of 4 `postoj`: average clears 85%, postoj does not.
        pairs = [("spawanie", "spawanie")] * 18 + [("postoj", "postoj")] * 2
        pairs += [("postoj", "spawanie")] * 2
        scores = ev.per_activity_scores(pairs, ["spawanie", "postoj"])
        assert scores["spawanie"]["passes"] is True
        assert scores["postoj"]["recall"] == 0.5
        assert scores["postoj"]["passes"] is False

    def test_eighty_four_percent_fails(self):
        pairs = [("spawanie", "spawanie")] * 84 + [("spawanie", "postoj")] * 16
        scores = ev.per_activity_scores(pairs, ["spawanie"])
        assert scores["spawanie"]["recall"] == pytest.approx(0.84)
        assert scores["spawanie"]["passes"] is False

    def test_granularity_is_reported_for_thin_classes(self):
        pairs = [("postoj", "postoj")] * 2
        scores = ev.per_activity_scores(pairs, ["postoj"])
        # n=2 means one error moves the score 50 points - the figure cannot resolve 85%.
        assert scores["postoj"]["granularity_pp"] == pytest.approx(50.0)

    def test_precision_is_reported_separately_from_the_bar(self):
        # Over-calls spawanie: perfect recall, poor precision. The bar still passes,
        # which is exactly why precision has to be visible next to it.
        pairs = [("spawanie", "spawanie")] * 10 + [("postoj", "spawanie")] * 10
        scores = ev.per_activity_scores(pairs, ["spawanie", "postoj"])
        assert scores["spawanie"]["recall"] == 1.0
        assert scores["spawanie"]["precision"] == pytest.approx(0.5)
        assert scores["spawanie"]["passes"] is True

    def test_recall_bought_by_over_calling_is_not_usable(self):
        # The arc baseline's real failure mode: 100% recall by calling twice as
        # much time `spawanie` as happened. It clears the client's bar and must
        # still not be promoted.
        pairs = [("spawanie", "spawanie")] * 10 + [("postoj", "spawanie")] * 10
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["time_ratio"] == pytest.approx(2.0)
        assert s["inflated"] is True
        assert s["passes"] is True  # the client's bar, reported unchanged
        assert s["usable"] is False  # but not a measurement

    def test_honest_arm_is_usable(self):
        pairs = [("spawanie", "spawanie")] * 18 + [("spawanie", "postoj")] * 2
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["time_ratio"] == pytest.approx(0.9)
        assert s["inflated"] is False
        assert s["usable"] is True

    def test_under_reporting_is_not_flagged_as_gamed(self):
        # Under-reporting fails the bar on recall, which is the correct and
        # sufficient failure - it must not also be labelled as inflation.
        pairs = [("spawanie", "spawanie")] * 5 + [("spawanie", "postoj")] * 15
        s = ev.per_activity_scores(pairs, ["spawanie"])["spawanie"]
        assert s["inflated"] is False
        assert s["passes"] is False


class TestHardwareVerdict:
    def test_over_budget_is_disqualified(self):
        v = ev.hardware_verdict({"peak_vram_mib": 13000, "gpus_used": 1})
        assert v["verdict"] == "DISQUALIFIED"

    def test_two_cards_is_disqualified_even_when_small(self):
        v = ev.hardware_verdict({"peak_vram_mib": 4000, "gpus_used": 2})
        assert v["verdict"] == "DISQUALIFIED"

    def test_missing_gpu_block_is_unmeasured_not_ok(self):
        assert ev.hardware_verdict(None)["verdict"] == "UNMEASURED"
        assert ev.hardware_verdict({"box": "cctv-vps"})["verdict"] == "UNMEASURED"

    def test_within_budget_passes(self):
        v = ev.hardware_verdict({"peak_vram_mib": 7866, "gpus_used": 1})
        assert v["verdict"] == "OK"

    def test_cost_is_normalised_per_video_hour(self):
        gpu = {"gpu_seconds": 600, "video_seconds": 1200}
        assert ev.gpu_seconds_per_video_hour(gpu) == pytest.approx(1800.0)


class TestBoundaryErrors:
    def test_exact_boundaries_score_zero(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        b = ev.boundary_errors(truth, truth)
        assert b["median_s"] == 0
        assert b["spurious"] == 0

    def test_invented_boundaries_are_counted_as_spurious(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        pred = truth + [{"activity_id": "a", "start_s": 15, "end_s": 20}]
        b = ev.boundary_errors(truth, pred)
        assert b["spurious"] == 1

    def test_an_arm_emitting_no_boundaries_is_flagged(self):
        truth = [
            {"activity_id": "a", "start_s": 0, "end_s": 10},
            {"activity_id": "b", "start_s": 10, "end_s": 20},
        ]
        b = ev.boundary_errors(truth, [{"activity_id": "a", "start_s": 0, "end_s": 20}])
        assert b["matched"] == 0


class TestPredictionLoading:
    def test_intervals_and_samples_agree(self, tmp_path: Path):
        truth = _annotation("W1", ["spawanie", "spawanie", "postoj"])
        as_intervals = tmp_path / "a.json"
        as_intervals.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    "intervals": [
                        {"activity_id": "spawanie", "start_s": 0, "end_s": 3},
                        {"activity_id": "postoj", "start_s": 3, "end_s": 6},
                    ],
                }
            )
        )
        as_samples = tmp_path / "b.json"
        as_samples.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    "samples": [
                        {"t_s": 0, "activity_id": "spawanie"},
                        {"t_s": 2, "activity_id": "spawanie"},
                        {"t_s": 4, "activity_id": "postoj"},
                    ],
                }
            )
        )
        assert (
            ev.load_prediction(as_intervals, truth)["grid"]
            == (ev.load_prediction(as_samples, truth)["grid"])
        )

    def test_missing_prediction_is_an_error_not_a_smaller_denominator(self, tmp_path: Path):
        truth = _annotation("W1", ["spawanie", "spawanie", "postoj"])
        p = tmp_path / "partial.json"
        p.write_text(
            json.dumps(
                {
                    "arm": "x",
                    "window": "W1",
                    # Answers only the first sample.
                    "intervals": [{"activity_id": "spawanie", "start_s": 0, "end_s": 1}],
                }
            )
        )
        pred = ev.load_prediction(p, truth)
        pairs = [(gt, pred["grid"].get(t, "__brak_predykcji__")) for t, gt in truth["grid"].items()]
        scores = ev.per_activity_scores(pairs, ["spawanie"])
        # Support stays 2 - the unanswered sample counts against recall.
        assert scores["spawanie"]["support"] == 2
        assert scores["spawanie"]["recall"] == 0.5


class TestManifestSplit:
    def test_missing_split_block_refuses_to_score(self, tmp_path: Path):
        m = tmp_path / "manifest.json"
        m.write_text(json.dumps({"activities": [], "clips": []}))
        with pytest.raises(SystemExit):
            ev.load_manifest(m)


class TestArcBaseline:
    def test_threshold_is_tuned_per_clip(self):
        # Two clips with the same shape at different brightness: one fixed
        # threshold cannot serve both, which is the whole clip-relative caveat.
        dim = {t: (5.0 if t < 10 else 0.1) for t in range(20)}
        bright = {t: (50.0 if t < 10 else 1.0) for t in range(20)}
        truth = _annotation("W", ["spawanie"] * 5 + ["postoj"] * 5)
        t_dim, f1_dim = ev.tune_arc_threshold(dim, truth)
        t_bright, f1_bright = ev.tune_arc_threshold(bright, truth)
        assert f1_dim == pytest.approx(1.0)
        assert f1_bright == pytest.approx(1.0)
        assert t_bright > t_dim


class TestCollapse:
    def test_members_score_as_one_bucket(self):
        # The delivery vocabulary the client accepted: two named activities and
        # everything else as one line. A model that confused `postoj` with
        # `inna_czynnosc` was wrong under seven categories and is right under three.
        collapse = {"bucket": "pozostale", "members": ["inna_czynnosc", "postoj"]}
        pairs = [("postoj", "inna_czynnosc")] * 8 + [("inna_czynnosc", "postoj")] * 2
        merged = ev.collapse_pairs(pairs, collapse)
        s = ev.per_activity_scores(merged, ["pozostale"])["pozostale"]
        assert s["support"] == 10
        assert s["recall"] == 1.0

    def test_nierozpoznane_is_refused_as_a_bucket_member(self):
        # The manifest defines it as never work and never downtime. Folding the
        # honest "cannot tell" into a work bucket would quietly convert unknown
        # time into measured time - the one error a chronometraz cannot survive.
        manifest = _manifest(members=["postoj", ev.NON_ACTIVITY])
        with pytest.raises(SystemExit):
            ev.resolve_collapse(manifest, None)

    def test_nierozpoznane_keeps_its_own_row(self):
        # Three delivered categories plus the abstention, which is not one of them.
        collapse = {"bucket": "pozostale", "members": DELIVERY_MEMBERS}
        assert ev.collapse_classes(ACTIVITY_IDS, collapse) == [
            "spawanie",
            "ukladanie_pretow",
            "pozostale",
            "nierozpoznane",
        ]
        assert ev.collapse_pairs([("nierozpoznane", "nierozpoznane")], collapse) == [
            ("nierozpoznane", "nierozpoznane")
        ]

    def test_unknown_member_is_refused(self):
        # A typo would silently build a smaller bucket and still print a
        # confident three-category number.
        manifest = _manifest(members=["postoj", "sciaganie_elemetu"])
        with pytest.raises(SystemExit):
            ev.resolve_collapse(manifest, None)

    def test_bucket_named_after_a_real_activity_is_refused(self):
        # `postoj` would then mean both the activity and the bucket, and no
        # reader of the table could tell which figure they were looking at.
        manifest = _manifest(members=["inna_czynnosc", "postoj"], bucket="postoj")
        with pytest.raises(SystemExit):
            ev.resolve_collapse(manifest, None)

    def test_declared_vocabulary_names_the_manifest_as_its_source(self):
        # The report header prints this, so a reader can tell a fixture-declared
        # vocabulary from one someone typed at the prompt.
        collapse = ev.resolve_collapse(_manifest(members=DELIVERY_MEMBERS), None)
        assert collapse["bucket"] == "pozostale"
        assert collapse["members"] == DELIVERY_MEMBERS
        assert collapse["source"] == "manifest.source.json"

    def test_flag_overrides_the_declared_vocabulary(self):
        manifest = _manifest(members=DELIVERY_MEMBERS)
        collapse = ev.resolve_collapse(manifest, "inne=postoj,inna_czynnosc")
        assert collapse["bucket"] == "inne"
        assert collapse["members"] == ["postoj", "inna_czynnosc"]
        assert collapse["source"] == "--collapse"

    def test_flag_is_validated_like_the_declared_block(self):
        # An ad-hoc vocabulary is still a vocabulary; the guards are not a
        # property of where the mapping was written down.
        manifest = _manifest(members=DELIVERY_MEMBERS)
        with pytest.raises(SystemExit):
            ev.resolve_collapse(manifest, f"inne=postoj,{ev.NON_ACTIVITY}")


def _write_fixture(tmp_path: Path, *, declare_vocabulary: bool) -> Path:
    """A two-window fixture whose arm confuses exactly the classes the bucket merges.

    Under seven categories the arm scores 0% on `postoj` and `inna_czynnosc`;
    under three they are the same line and it scores 100%. That is the whole
    hypothesis of #121, so the fixture has to be able to show it.
    """
    labels = [
        "spawanie",
        "spawanie",
        "spawanie",
        "spawanie",
        "ukladanie_pretow",
        "ukladanie_pretow",
        "postoj",
        "postoj",
        "inna_czynnosc",
        "nierozpoznane",
    ]
    # Swaps `postoj` for `inna_czynnosc` and back - wrong under seven, right under three.
    swap = {"postoj": "inna_czynnosc", "inna_czynnosc": "postoj"}
    predicted = [swap.get(a, a) for a in labels]

    manifest = {
        "activities": [{"id": a} for a in ACTIVITY_IDS],
        "clips": [
            {
                "slot": w,
                "annotated": True,
                "annotation_file": f"{w}.intervals.json",
                "window_local": f"{w} 07:00",
                "shift_position": "pre-break",
            }
            for w in ("W1", "W2")
        ],
        "split": {
            "declared": "2026-09-02",
            "protocol": "2-fold cross-validation",
            "folds": [
                {"id": "A", "train_dev": ["W1"], "held_out": ["W2"]},
                {"id": "B", "train_dev": ["W2"], "held_out": ["W1"]},
            ],
            "reporting": "union of held-out folds",
            "caveat": "toy fixture",
        },
    }
    if declare_vocabulary:
        manifest["delivery_vocabulary"] = {
            "declared": "2026-09-02",
            "bucket": "pozostale",
            "members": DELIVERY_MEMBERS,
            "why": "the three-category scope the client accepted (#117)",
        }

    for w in ("W1", "W2"):
        truth = _annotation(w, labels)
        (tmp_path / f"{w}.intervals.json").write_text(
            json.dumps(
                {
                    "window": w,
                    "stride_s": truth["stride_s"],
                    "duration_s": truth["duration_s"],
                    "samples": [
                        {"t_s": t, "activity_id": a} for t, a in sorted(truth["grid"].items())
                    ],
                    "intervals": truth["intervals"],
                }
            )
        )
        (tmp_path / f"pred-{w}.json").write_text(
            json.dumps(
                {
                    "arm": "toy",
                    "window": w,
                    "samples": [{"t_s": i * 2, "activity_id": a} for i, a in enumerate(predicted)],
                    "gpu": {
                        "box": "cctv-vps",
                        "gpus_used": 1,
                        "gpu_seconds": 10.0,
                        "video_seconds": 20,
                        "peak_vram_mib": 1000,
                    },
                }
            )
        )

    path = tmp_path / "manifest.source.json"
    path.write_text(json.dumps(manifest))
    return path


def _render_fixture(tmp_path: Path, monkeypatch, *, declare_vocabulary=True, flag=None) -> str:
    manifest = _write_fixture(tmp_path, declare_vocabulary=declare_vocabulary)
    out = tmp_path / "report.md"
    argv = [
        "evaluate_arms.py",
        "--manifest",
        str(manifest),
        "--predictions",
        str(tmp_path / "pred-W1.json"),
        str(tmp_path / "pred-W2.json"),
        "--out",
        str(out),
    ]
    if flag:
        argv += ["--collapse", flag]
    monkeypatch.setattr(sys, "argv", argv)
    ev.main()
    return out.read_text()


def _slice(text: str, heading: str, level: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    end = next(
        (i for i, ln in enumerate(lines[start + 1 :], start + 1) if ln.startswith(level)),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _tables(report: str) -> list[list[str]]:
    """Every markdown table in the report, as runs of consecutive `|` lines."""
    tables, current = [], []
    for ln in report.splitlines():
        if ln.startswith("|"):
            current.append(ln)
        elif current:
            tables.append(current)
            current = []
    return tables + ([current] if current else [])


def _row_labels(table: list[str]) -> set[str]:
    return {row.split("|")[1].strip().strip("`") for row in table if row.count("|") > 1}


def _section(report: str, heading: str) -> str:
    return _slice(report, heading, "## ")


def _subsection(section: str, heading: str) -> str:
    return _slice(section, heading, "### ")


class TestCollapsedReport:
    def test_header_states_which_activities_were_merged(self, tmp_path: Path, monkeypatch):
        # A three-category figure must never be mistakable for a seven-category
        # one, so the merge is stated before any number is shown.
        report = _render_fixture(tmp_path, monkeypatch)
        lines = report.splitlines()
        stated = [
            i
            for i, ln in enumerate(lines)
            if "`pozostale`" in ln and all(f"`{m}`" in ln for m in DELIVERY_MEMBERS)
        ]
        assert stated, "no line states which activities were merged"
        first_arm = next(i for i, ln in enumerate(lines) if ln.startswith("## Arm:"))
        assert stated[0] < first_arm, "the merge is stated after the first arm's numbers"

    def test_three_categories_carry_recall_and_time_ratio_in_both_tables(
        self, tmp_path: Path, monkeypatch
    ):
        # The folds disagree on this fixture, so the union alone describes
        # neither of them - both views have to be present.
        delivery = _section(_render_fixture(tmp_path, monkeypatch), "## Delivery vocabulary")
        union = _subsection(delivery, "### Held-out union")
        per_window = _subsection(delivery, "### Per held-out window")

        union_row = next(ln for ln in union.splitlines() if ln.startswith("| `pozostale`"))
        assert "100.0%" in union_row
        assert "1.00×" in union_row

        assert "W1" in per_window and "W2" in per_window
        window_row = next(ln for ln in per_window.splitlines() if ln.startswith("| `pozostale`"))
        assert window_row.count("100.0%") == 2
        assert window_row.count("1.00×") == 2

    def test_no_table_mixes_the_two_vocabularies(self, tmp_path: Path, monkeypatch):
        # A merge cannot be undone by reading harder. If one table ever carried
        # both `pozostale` and one of its members, a reader would have no way to
        # tell a three-category figure from a seven-category one.
        report = _render_fixture(tmp_path, monkeypatch)
        members = set(DELIVERY_MEMBERS)
        for table in _tables(report):
            labels = _row_labels(table)
            assert not ("pozostale" in labels and labels & members), table[0]

    def test_seven_category_sections_are_untouched_by_the_collapse(
        self, tmp_path: Path, monkeypatch
    ):
        # The arm swaps two activities the bucket merges. Under seven categories
        # that is still an error, and the collapse must not launder it.
        report = _render_fixture(tmp_path, monkeypatch)
        arm = _section(report, "## Arm: `toy`")
        postoj = next(ln for ln in arm.splitlines() if ln.startswith("| `postoj`"))
        assert "0.0%" in postoj

    def test_no_declared_vocabulary_means_no_delivery_section(self, tmp_path: Path, monkeypatch):
        report = _render_fixture(tmp_path, monkeypatch, declare_vocabulary=False)
        assert "Delivery vocabulary" not in report
        assert "pozostale" not in report

    def test_unanswered_samples_are_not_swallowed_by_the_bucket(self):
        # The harness's central rule - an unanswered sample is an error, not a
        # smaller denominator - has to survive the collapse. It would not if the
        # bucket were defined as "everything except the named categories".
        collapse = {"bucket": "pozostale", "members": DELIVERY_MEMBERS}
        merged = ev.collapse_pairs([("postoj", "__brak_predykcji__")], collapse)
        assert merged == [("pozostale", "__brak_predykcji__")]
        s = ev.per_activity_scores(merged, ["pozostale"])["pozostale"]
        assert s["support"] == 1
        assert s["recall"] == 0.0
