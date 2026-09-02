"""Fill the client report's @@PLACEHOLDER@@ slots from measured artifacts.

Every number a client reads comes from `C0-report.json` or
`empty-station-rule.json`, never from a hand-typed table. The previous version
of this report carried 98.5% for `brak_na_stanowisku` that no arm had ever
scored: the pose detector's own hit rate on one window, in-sample, sitting in a
column of held-out classifier accuracies. A substitution script cannot make that
mistake.

    uv run fill_report.py C0-report.json empty-station-rule.json body.part out.html

Two groups, because the folds disagree. `Z` (znana) is the union of the two
held-out morning windows, whose folds trained on evening material as well. `N`
(nowa) is the evening window, held out by the fold that saw only mornings. The
overall union of all three is not used anywhere in the client body: averaging
97.8%, 91.2% and 27.5% into 72% would describe none of them.
"""

import json
import sys
from pathlib import Path

C0 = Path(sys.argv[1])
RULE = Path(sys.argv[2])
TEMPLATE = Path(sys.argv[3])
OUT = Path(sys.argv[4])

ARM = "tcn-pose"
VLM = "vlm-qwen2.5-vl-3b-v1"
RICH = "tcn-rich"
KNOWN = ["W1", "W2"]
NEW = ["W3"]
FOLD_WINDOW = {"A": "W3", "B": "W2", "C": "W1"}

KEYS = {
    "SPAW": "spawanie",
    "UKL": "ukladanie_pretow",
    "SCI": "sciaganie_elementu",
    "INNA": "inna_czynnosc",
    "POST": "postoj",
    "BRAK": "brak_na_stanowisku",
}
BAR = 0.85
INFLATION = 1.25


def pct(v):
    return f"{v * 100:.1f}".replace(".", ",") + "%"


def times(v):
    return f"{v:.2f}".replace(".", ",") + "×"


def merge(arm, windows, cls):
    """Recall and time ratio over a group of held-out windows, weighted by support."""
    pw = arm["per_window_scores"]
    n = hit = predicted = 0
    for w in windows:
        s = pw.get(w, {}).get(cls)
        if not s or not s["support"]:
            continue
        n += s["support"]
        hit += s["recall"] * s["support"]
        if s["time_ratio"] is not None:
            predicted += s["time_ratio"] * s["support"]
    if not n:
        return None
    return {"recall": hit / n, "time_ratio": predicted / n, "support": n}


def main() -> int:
    rep = json.loads(C0.read_text())
    rule = json.loads(RULE.read_text())
    arms = {a["name"]: a for a in rep["arms"]}
    for need in (ARM, VLM, RICH):
        if need not in arms:
            sys.exit(f"{need} missing from {C0}; have {sorted(arms)}")
    arm = arms[ARM]
    for w in KNOWN + NEW:
        if w not in arm["per_window_scores"]:
            sys.exit(f"{ARM} has no scores for {w} - the 3-fold run is incomplete")

    sub = {}
    for short, cls in KEYS.items():
        z = merge(arm, KNOWN, cls)
        n = merge(arm, NEW, cls)
        if z is None:
            sys.exit(f"no held-out support for {cls} on {KNOWN}")
        sub[f"REC_{short}_Z"] = pct(z["recall"])
        sub[f"RAT_{short}_Z"] = times(z["time_ratio"])
        sub[f"REC_{short}_N"] = "n/d" if n is None else pct(n["recall"])
        if z["recall"] >= BAR and z["time_ratio"] <= INFLATION:
            pill = '<span class="pill ok">gotowe</span>'
        elif z["recall"] >= BAR:
            pill = '<span class="pill wait">zawyżony czas</span>'
        elif z["recall"] >= 0.5:
            pill = '<span class="pill wait">za mało</span>'
        else:
            pill = '<span class="pill pause">nie działa</span>'
        sub[f"PILL_{short}"] = pill

    for short in ("SPAW", "UKL"):
        r = merge(arm, KNOWN, KEYS[short])["time_ratio"]
        sub[f"ERR_{short}"] = f"{abs(r - 1) * 100:.0f}%"

    for fold, w in FOLD_WINDOW.items():
        for short in ("SPAW", "UKL"):
            s = arm["per_window_scores"][w][KEYS[short]]
            sub[f"F_{fold}_{short}"] = pct(s["recall"])

    for prefix, name in (("V", VLM), ("P", ARM), ("R", RICH)):
        for short in ("SPAW", "UKL", "SCI", "INNA", "POST"):
            m = merge(arms[name], KNOWN, KEYS[short])
            sub[f"{prefix}_{short}"] = "n/d" if m is None else pct(m["recall"])

    vw3 = arms[VLM]["per_window_scores"].get("W3", {}).get("spawanie")
    if not vw3 or not vw3["support"]:
        sys.exit("the VLM arm has no W3 score - the cross-check paragraph has no source")
    sub["VLM_W3_SPAW"] = pct(vw3["recall"])
    sub["VLM_W3_RAT"] = f"{vw3['time_ratio']:.1f}".replace(".", ",") + " raza"

    # Boundary timing on the recommended configuration only, same basis as the
    # accuracy rows beside it.
    pwb = arm["per_window_boundaries"]
    tot = sum(pwb[w]["n"] for w in KNOWN)
    within = sum(pwb[w]["within_2s_frac"] * pwb[w]["n"] for w in KNOWN) / tot
    sub["BOUND"] = f"{within * 100:.0f}%"

    mins = arm["gpu_seconds_per_video_hour"] / 60
    sub["GPUMIN"] = f"{mins:.1f}".replace(".", ",") + " minuty"
    hw = arm["hardware"]
    if hw["verdict"] != "OK":
        sys.exit(f"{ARM} hardware verdict is {hw['verdict']}: {hw['reason']}")
    sub["VRAM"] = hw["reason"].split(" peak")[0]

    u = rule["held_out_union"]
    sub["BRAK_RULE_REC"] = pct(u["recall"])
    # Stated as "how much too much", which is what a chronometraż reader feels,
    # rather than as a multiplier they have to convert in their head.
    sub["BRAK_RULE_OVER"] = f"{(u['time_ratio'] - 1) * 100:.0f}%"
    sub["BRAK_NAIVE_OVER"] = f"{(rule['naive']['union']['time_ratio'] - 1) * 100:.0f}%"
    sub["BRAK_LIMIT"] = f"{(INFLATION - 1) * 100:.0f}%"

    text = TEMPLATE.read_text()
    for k, v in sub.items():
        text = text.replace(f"@@{k}@@", v)
    left = text.split("@@")[1::2]
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")
    if "—" in text:
        sys.exit("em dash found in the body; the client asked for none")
    OUT.write_text(text)
    for k in sorted(sub):
        print(f"{k:16} {sub[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
