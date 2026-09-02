# C.0 measurement report — `hala-prawe-v1`

Generated 2026-09-01 by `benchmarks/activity/tools/evaluate_arms.py`.

## Coverage

- **W1** 2026-08-28 09:00-09:20 Europe/Warsaw — morning, pre-break
- **W2** 2026-08-28 10:20-10:40 Europe/Warsaw — late morning, pre-break
- **W3** 2026-09-01 18:25-18:45 Europe/Warsaw — evening, second shift (17:00-23:00)

**2 of 3 windows are pre-break.** The aggregate still leans that way, and the one window that does not differs in both shift and operator, so a per-window difference cannot be attributed to either alone.

Three windows, one station, two operators, two of the three pre-break. Cross-validation bounds overfitting to one window; it does not establish transfer to another station. Two confounds are now entangled and this fixture cannot separate them: W3 differs from W1/W2 in BOTH shift and operator, so a drop on W3 could be either and calling it 'the afternoon is harder' would be unsupported. Aggregate figures still lean pre-break, two windows to one. With three folds the variance on a small class stays wide; treat anything under ~100 held-out samples as indicative, not settled.

## Split

Protocol: **3-fold cross-validation over the three windows**, declared 2026-09-02.
- fold A: train ['W1', 'W2'] → held out ['W3']
- fold B: train ['W1', 'W3'] → held out ['W2']
- fold C: train ['W2', 'W3'] → held out ['W1']

Per-activity accuracy is computed on the UNION of the three folds' held-out predictions - every labelled sample is predicted exactly once, by a model that never saw it. One confusion matrix per arm over that union, plus the per-fold matrices so a fold-specific collapse is visible.

## Delivery vocabulary: 3 categories

`pozostale` = `sciaganie_elementu` + `inna_czynnosc` + `postoj` + `brak_na_stanowisku` — declared in `manifest.source.json`.

Every figure in this section is over 3 categories and is **not comparable** with the per-arm sections below, which score all 7 separately. A merge cannot be undone by reading harder, so the two vocabularies never share a table.

`nierozpoznane` is **not** a member of the bucket and keeps its own row. It is neither work nor downtime, and folding the honest "cannot tell" into a work bucket would convert unknown time into measured time.

### Held-out union

#### `tcn-pixel-518-delivered`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 91.8% | 82.8% | 1.11× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 552 | 84.6% | 91.9% | 0.92× | 0.2 pp | ❌ |
| `pozostale` | 441 | 65.8% | 64.3% | 1.02× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 0.24× | 1.6 pp | — |

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither — and on this fixture they disagree. Cells are recall (time reported, n).

#### `tcn-pixel-518-delivered`

| Category | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 98.7% (1.13×, n=224) | 93.0% (0.97×, n=273) | 84.2% (1.24×, n=247) |
| `ukladanie_pretow` | 91.7% (0.99×, n=180) | 97.2% (1.07×, n=179) | 66.3% (0.72×, n=193) |
| `pozostale` | 72.3% (0.78×, n=195) | 72.2% (1.60×, n=90) | 53.8% (0.99×, n=156) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (0.00×, n=4) |

## Arm: `tcn-pixel-518-delivered`

*#122 - the delivered vocabulary trained directly, against #121's collapsed figures as the floor*

**Hardware verdict: UNMEASURED** — `peak_vram_mib` missing

**Cost: 18 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 0.8% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 91.8% | 82.8% | 1.11× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 552 | 84.6% | 91.9% | 0.92× | 0.2 pp | ❌ |
| `sciaganie_elementu` | 110 | 0.0% | n/a | 0.00× | 0.9 pp | ❌ |
| `inna_czynnosc` | 177 | 0.0% | n/a | 0.00× | 0.6 pp | ❌ |
| `postoj` | 62 | 0.0% | n/a | 0.00× | 1.6 pp | ❌ |
| `brak_na_stanowisku` | 92 | 0.0% | n/a | 0.00× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 0.24× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 98.7% (n=224) | 93.0% (n=273) | 84.2% (n=247) |
| `ukladanie_pretow` | 91.7% (n=180) | 97.2% (n=179) | 66.3% (n=193) |
| `sciaganie_elementu` | 0.0% (n=22) | 0.0% (n=30) | 0.0% (n=58) |
| `inna_czynnosc` | 0.0% (n=65) | 0.0% (n=42) | 0.0% (n=70) |
| `postoj` | 0.0% (n=42) | 0.0% (n=16) | 0.0% (n=4) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 0.0% (n=24) |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `nierozpoznane` | `pozostale` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|
| `brak_na_stanowisku` | 0 | 92 | 0 | 0 |
| `inna_czynnosc` | 11 | 68 | 81 | 17 |
| `nierozpoznane` | 0 | 59 | 0 | 3 |
| `postoj` | 4 | 46 | 4 | 8 |
| `sciaganie_elementu` | 0 | 84 | 17 | 9 |
| `spawanie` | 0 | 57 | 683 | 4 |
| `ukladanie_pretow` | 0 | 45 | 40 | 467 |

### Boundary timing error

249 real activity changes, 220 predicted. Median error **2.0 s**, p90 10.0 s, max 24.0 s; 61.8% land within 2 s. Spurious boundaries (no real change within 4 s): **54**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arc-flash baseline on `spawanie`

Reported at **two operating points**, because a single threshold tells a misleading story about this signal. *Conservative* is the clip-relative cut-off the annotation hints used. *Oracle F1* is the best threshold available in hindsight on that same clip — in-sample, unavailable in production, and deliberately generous: an arm that costs a GPU should have to beat the baseline's best day, not a strawman.

**Conservative**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 1.13 | 44.2% | 90.0% | 0.49× | 0.593 |
| W2 | 1.68 | 37.4% | 96.2% | 0.39× | 0.538 |
| W3 | 3.65 | 37.7% | 33.7% | 1.12× | 0.356 |

Union on `spawanie`: recall **39.5%**, precision 59.8%, time reported 0.66×.

**Oracle F1 (in-sample)**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 0.05 | 99.1% | 44.7% | 2.22× | 0.616 |
| W2 | 0.12 | 99.6% | 46.3% | 2.15× | 0.633 |
| W3 | 0.04 | 93.1% | 44.4% | 2.10× | 0.601 |

Union on `spawanie`: recall **97.3%**, precision 45.2%, time reported 2.15×.

Cost: **0 GPU-seconds** at either point.

**The baseline clears the 85% recall bar on `spawanie` — and that is a finding about the bar, not about the baseline.** It reaches 97.3% recall by calling 2.15× as much time `spawanie` as actually was, at 45.2% precision. A recall-only bar is gameable by any arm willing to over-call the common class, so no arm should be promoted on recall alone. The `Time reported` column is what separates a measurement from a guess that happens to overlap the truth.

Either way the baseline is the **cost floor, not a candidate**: it cannot distinguish `ukladanie_pretow` from `postoj` at all, which is five of the seven activities and all of the hard part.

## Go / no-go

_Not generated. The bar is numeric but the decision is human — recorded as a comment on issue #117 by @tkowalczyk._
