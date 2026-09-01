# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build a per-track INTERVAL annotation package for a process-activity fixture.

The third annotation shape in this repo, and the one `hala-prawe-v1` needs. The
existing tools produce per-person *per-frame* posture labels
(`build_activity_annotation.py`, `magazyn-hall-v1`); a chronometraż records
*intervals*, so the unit here is `(zone_id, activity_id, start_s, end_s)` and a
20-minute clip is tens of rows rather than 1200 frames x N people.

## Why the unit is the zone, not a person track

Assumption A5, confirmed by the client 2026-08-28: the unit of measurement is
the **station**, never an identified person. `brak na stanowisku` means the zone
is empty, whoever they are. So this tool samples one station ROI over time and
asks "what is happening at this station right now" - no tracking, no re-ID, no
faces. `track_id` in the emitted schema is the zone, which is what the product
actually measures.

That is what makes hand-annotation tractable. Tracking every person through a
4K fisheye hall and labelling each track separately is a different, much larger
job, and the work-study does not need it.

## What it does, and does not

Three things:

1. Samples the station ROI out of the clip at a fixed stride, cropped from the
   NATIVE frame (see `station_roi` in the fixture manifest) - never from a
   downscale, which would throw away the pixels that make station framing work.
2. Writes a scaffold with `activity: null` on every sample, optionally seeded
   with `spawanie` SUGGESTIONS from the arc-flash timeline.
3. Renders a keyboard-driven `*.timeline.html` for the human pass, which folds
   consecutive same-label samples into intervals and exports them.

DELIBERATELY NOT A CLASSIFIER. Every `activity` starts null. The arc-flash
suggestion is a brightness threshold, offered as a hint the annotator confirms
or rejects - it detects an arc, not welding *work*, and cannot tell
`ukladanie_pretow` from `postoj`. A suggestion that is never confirmed never
becomes a label.

## Timing

Sampling uses ffmpeg's `fps` filter, which derives from presentation timestamps.
That matters here: W1's container reports `r_frame_rate=120/1` while the true
rate is ~20 fps, so anything that indexes by frame number mis-maps every W1
timestamp. Sample `i` is at `t = i * stride` seconds of PTS, and the boundary
between two differently-labelled samples is recorded at their midpoint, so
boundary error is at most `stride / 2`. At the default stride of 2 s that is
+/-1 s, which is the resolution the client accepted (A3).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# Seconds between samples. 2 s keeps boundary error at +/-1 s (A3's accepted
# resolution) while halving the number of crops a human must look at versus 1 s.
DEFAULT_STRIDE_S = 2

# Long edge the ROI crop is scaled to for review. The crop is taken at native
# resolution first; this only shrinks it for the browser.
VIEW_LONG_EDGE = 640

# Arc-suggestion threshold, as a fraction of the way from this clip's median to
# its 99th percentile. The arc metric is CLIP-RELATIVE - W1 and W2 differ ~3x at
# the median on the identical crop - so an absolute cut-off does not port
# between windows. Deliberately conservative: a missed suggestion costs the
# annotator a keystroke, a wrong one risks becoming an unexamined label.
ARC_SUGGEST_FRACTION = 0.25


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...")


def _probe_duration_s(clip: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed on {clip}")
    return float(proc.stdout.strip())


def _extract_samples(clip: Path, roi: dict, stride: int, out_dir: Path) -> int:
    """Crop the station ROI at `stride`-second intervals. Returns sample count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("t*.jpg"):
        stale.unlink()
    crop = f"crop={roi['w']}:{roi['h']}:{roi['x']}:{roi['y']}"
    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"fps=1/{stride},{crop},scale={VIEW_LONG_EDGE}:-1",
            "-q:v",
            "4",
            str(out_dir / "t%05d.jpg"),
        ]
    )
    return len(list(out_dir.glob("t*.jpg")))


def _load_arc_suggestions(
    arc_csv: Path | None, window: str, stride: int, count: int
) -> tuple[list[bool], float | None]:
    """Per-sample `spawanie` hints from the arc timeline, thresholded per clip."""
    if arc_csv is None or not arc_csv.exists():
        return [False] * count, None

    per_second: dict[int, float] = {}
    with arc_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("window") != window:
                continue
            per_second[int(row["t_s"])] = float(row["arc_metric"])
    if not per_second:
        return [False] * count, None

    values = sorted(per_second.values())
    median = statistics.median(values)
    p99 = values[min(len(values) - 1, int(len(values) * 0.99))]
    threshold = median + (p99 - median) * ARC_SUGGEST_FRACTION

    hints: list[bool] = []
    for i in range(count):
        # Any arcing second inside this sample's window suggests welding - an arc
        # is intermittent by nature and a sample is a window, not an instant.
        window_vals = [per_second.get(t, 0.0) for t in range(i * stride, (i + 1) * stride)]
        hints.append(any(v >= threshold for v in window_vals))
    return hints, threshold


def _render_html(out_html: Path, payload: dict, activities: list[dict], crops_rel: str) -> None:
    keys = "".join(
        f"<li><kbd>{i + 1}</kbd> {html.escape(a['name_pl'])}"
        f" <span class='cat'>{html.escape(a.get('category') or '—')}</span></li>"
        for i, a in enumerate(activities)
    )
    out_html.write_text(
        _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__KEYS__", keys)
        .replace("__CROPS__", crops_rel)
        .replace("__TITLE__", html.escape(f"{payload['fixture']} · {payload['window']}")),
        encoding="utf-8",
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#111;--fg:#eee;--dim:#888;--edge:#333;--hot:#f59e0b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:1fr 18rem;gap:1rem;padding:1rem;
  height:100vh;box-sizing:border-box}
.stage{display:flex;flex-direction:column;min-width:0}
img{max-width:100%;max-height:70vh;object-fit:contain;background:#000;border:1px solid var(--edge)}
.meta{display:flex;gap:1rem;align-items:center;margin:.5rem 0;flex-wrap:wrap}
.badge{padding:.1rem .5rem;border:1px solid var(--edge);border-radius:.25rem}
.suggest{color:var(--hot);border-color:var(--hot)}
.strip{display:flex;flex-wrap:wrap;gap:1px;margin-top:.5rem;max-height:14vh;overflow:auto}
.cell{width:8px;height:14px;background:#222;cursor:pointer}
.cell.done{background:#22c55e}.cell.cur{outline:2px solid #fff}
aside{overflow:auto;border-left:1px solid var(--edge);padding-left:1rem}
ul{list-style:none;padding:0;margin:0 0 1rem}
li{padding:.15rem 0}.cat{color:var(--dim);font-size:12px}
kbd{background:#222;border:1px solid var(--edge);border-radius:.2rem;padding:0 .3rem}
button{background:#222;color:var(--fg);border:1px solid var(--edge);border-radius:.25rem;
  padding:.4rem .6rem;cursor:pointer;width:100%;margin-bottom:.4rem}
.note{color:var(--dim);font-size:12px}
</style></head><body>
<div class="wrap">
  <div class="stage">
    <img id="shot" alt="">
    <div class="meta">
      <span class="badge" id="pos"></span>
      <span class="badge" id="time"></span>
      <span class="badge" id="label"></span>
      <span class="badge suggest" id="hint" hidden>sugestia: spawanie —
        <kbd>s</kbd> aby przyjąć</span>
    </div>
    <div class="strip" id="strip"></div>
  </div>
  <aside>
    <p class="note">Klawisze 1–7 etykietują i przechodzą dalej. ←/→ nawigacja,
      <kbd>0</kbd> czyści, <kbd>s</kbd> przyjmuje sugestię.</p>
    <ul>__KEYS__</ul>
    <button id="export">Eksportuj interwały (JSON)</button>
    <button id="clear">Wyczyść wszystko</button>
    <p class="note" id="stats"></p>
    <p class="note">Postęp zapisuje się w tej przeglądarce automatycznie.</p>
  </aside>
</div>
<script>
const D = __PAYLOAD__;
const CROPS = "__CROPS__";
const KEY = "interval-annot:" + D.fixture + ":" + D.window;
const ACTS = D.activities.map(a => a.id);
let labels = new Array(D.samples.length).fill(null);
try { const s = localStorage.getItem(KEY); if (s) labels = JSON.parse(s); } catch {}
if (labels.length !== D.samples.length) labels = new Array(D.samples.length).fill(null);
let i = labels.findIndex(v => v === null); if (i < 0) i = 0;

const $ = id => document.getElementById(id);
const strip = $("strip");
D.samples.forEach((s, n) => {
  const c = document.createElement("div");
  c.className = "cell"; c.title = s.t_s + " s";
  c.onclick = () => { i = n; draw(); };
  strip.appendChild(c);
});

function pad(n){ return String(n).padStart(5, "0"); }
function mmss(t){
  const m = Math.floor(t/60), s = t%60;
  return m + ":" + String(s).padStart(2,"0");
}

function draw() {
  const s = D.samples[i];
  $("shot").src = CROPS + "/t" + pad(i + 1) + ".jpg";
  $("pos").textContent = (i + 1) + " / " + D.samples.length;
  $("time").textContent = mmss(s.t_s) + " (" + s.t_s + " s)";
  $("label").textContent = labels[i] ?? "— brak etykiety —";
  $("hint").hidden = !(s.arc_suggest && labels[i] === null);
  [...strip.children].forEach((c, n) => {
    c.classList.toggle("done", labels[n] !== null);
    c.classList.toggle("cur", n === i);
  });
  const done = labels.filter(v => v !== null).length;
  $("stats").textContent = done + " / " + labels.length + " oznaczonych (" +
    Math.round(100 * done / labels.length) + "%)";
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch {}
}

function setLabel(id) { labels[i] = id; if (i < labels.length - 1) i++; draw(); }

addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { i = Math.min(labels.length - 1, i + 1); draw(); }
  else if (e.key === "ArrowLeft") { i = Math.max(0, i - 1); draw(); }
  else if (e.key === "0") { labels[i] = null; draw(); }
  else if (e.key === "s" && D.samples[i].arc_suggest) { setLabel("spawanie"); }
  else if (/^[1-9]$/.test(e.key)) { const a = ACTS[+e.key - 1]; if (a) setLabel(a); }
  else return;
  e.preventDefault();
});

/* Fold consecutive same-label samples into intervals. A boundary sits at the
   midpoint between the last sample of one label and the first of the next, so
   the error is at most stride/2 rather than a full stride. */
function fold() {
  const out = [];
  let start = null, cur = null;
  for (let n = 0; n <= labels.length; n++) {
    const v = n < labels.length ? labels[n] : Symbol("end");
    if (v !== cur) {
      if (cur !== null && typeof cur === "string") {
        const s0 = D.samples[start].t_s, sPrev = D.samples[n - 1].t_s;
        out.push({
          activity_id: cur,
          start_s: start === 0 ? 0 : +(s0 - D.stride_s / 2).toFixed(1),
          end_s: n === labels.length ? D.duration_s : +(sPrev + D.stride_s / 2).toFixed(1),
        });
      }
      cur = typeof v === "string" ? v : null; start = n;
    }
  }
  return out;
}

$("export").onclick = () => {
  const intervals = fold();
  const doc = {
    fixture: D.fixture, window: D.window, zone_id: D.zone_id,
    annotated_at: new Date().toISOString().slice(0, 10),
    source: "human",
    stride_s: D.stride_s, duration_s: D.duration_s,
    coverage: { labelled: labels.filter(v => v !== null).length, total: labels.length },
    intervals,
    samples: D.samples.map((s, n) => ({ t_s: s.t_s, activity_id: labels[n] })),
  };
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = D.window + ".intervals.json";
  a.click();
};

$("clear").onclick = () => {
  if (!confirm("Usunąć wszystkie etykiety w tej przeglądarce?")) return;
  labels = new Array(D.samples.length).fill(null); i = 0; draw();
};

draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="fixture manifest.source.json")
    ap.add_argument("--slot", required=True, help="clip slot to annotate, e.g. W1")
    ap.add_argument("--clip", type=Path, help="override the clip path")
    ap.add_argument("--arc-csv", type=Path, help="arc-timeline.csv for spawanie hints")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE_S)
    ap.add_argument("--out", type=Path, help="output dir (default: manifest's dir)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe are required")

    manifest = json.loads(args.manifest.read_text())
    out_dir: Path = args.out or args.manifest.parent
    roi = manifest.get("station_roi", {}).get("crop")
    if not roi:
        raise SystemExit(
            "manifest has no station_roi.crop - this tool crops the station, not the hall"
        )

    clips = {c["slot"]: c for c in manifest.get("clips", [])}
    if args.slot not in clips:
        raise SystemExit(f"unknown slot {args.slot}; manifest has {sorted(clips)}")

    clip_path = args.clip
    if clip_path is None:
        preserved = clips[args.slot].get("r2_key_preserved")
        if not preserved:
            raise SystemExit(f"slot {args.slot} has no r2_key_preserved; pass --clip")
        clip_path = out_dir / Path(preserved).name
    if not clip_path.exists():
        raise SystemExit(f"clip not found: {clip_path}")

    duration = _probe_duration_s(clip_path)
    crops_dir = out_dir / "crops" / args.slot
    count = _extract_samples(clip_path, roi, args.stride, crops_dir)
    if count == 0:
        raise SystemExit("ffmpeg produced no samples")

    arc_csv = args.arc_csv
    if arc_csv is None:
        candidate = out_dir / "arc-timeline.csv"
        arc_csv = candidate if candidate.exists() else None
    hints, threshold = _load_arc_suggestions(arc_csv, args.slot, args.stride, count)

    activities = [a for a in manifest.get("activities", []) if a.get("category") is not None]
    unresolved = [a for a in manifest.get("activities", []) if a.get("category") is None]
    activities = activities + unresolved  # `nierozpoznane` last, but still bindable

    payload = {
        "fixture": manifest.get("fixture", "unknown"),
        "window": args.slot,
        "zone_id": manifest.get("station_roi", {}).get("activity", "station"),
        "stride_s": args.stride,
        "duration_s": round(duration, 1),
        "activities": [
            {"id": a["id"], "name_pl": a.get("name_pl", a["id"]), "category": a.get("category")}
            for a in activities
        ],
        "samples": [{"t_s": i * args.stride, "arc_suggest": bool(hints[i])} for i in range(count)],
    }

    scaffold = out_dir / f"{args.slot}.intervals.scaffold.json"
    scaffold.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    out_html = out_dir / f"{args.slot}.timeline.html"
    _render_html(out_html, payload, payload["activities"], f"crops/{args.slot}")

    suggested = sum(hints)
    print(f"samples:    {count} @ {args.stride}s stride ({duration:.1f}s clip)")
    print(f"boundary:   +/-{args.stride / 2:.1f}s")
    if threshold is not None:
        print(f"arc hints:  {suggested} samples suggested as spawanie (threshold {threshold:.2f})")
    else:
        print("arc hints:  none (no arc-timeline.csv for this window)")
    print(f"scaffold:   {scaffold}")
    print(f"open:       {out_html}")


if __name__ == "__main__":
    main()
