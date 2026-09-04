# C.0 measurement report — `hala-prawe-v1`

Generated 2026-09-01 by `benchmarks/activity/tools/evaluate_arms.py`.

## Coverage

- **W1** 2026-08-28 09:00-09:20 Europe/Warsaw — morning, pre-break
- **W2** 2026-08-28 10:20-10:40 Europe/Warsaw — late morning, pre-break
- **W3** 2026-09-01 18:25-18:45 Europe/Warsaw — evening, second shift (17:00-23:00)
- **W4** 2026-09-04 06:00-06:20 Europe/Warsaw — morning, BEFORE the configured 07:00 shift start - the operator is at the bench from at least 05:57 local, which the stored shift windows do not describe
- **W5** 2026-09-04 07:10-07:30 Europe/Warsaw — morning, inside the configured 07:00-15:00 shift

**2 of 5 windows are pre-break.** The aggregate still leans that way, and the one window that does not differs in both shift and operator, so a per-window difference cannot be attributed to either alone.

Three windows, one station, two operators, two of the three pre-break. Cross-validation bounds overfitting to one window; it does not establish transfer to another station. Two confounds are now entangled and this fixture cannot separate them: W3 differs from W1/W2 in BOTH shift and operator, so a drop on W3 could be either and calling it 'the afternoon is harder' would be unsupported. Aggregate figures still lean pre-break, two windows to one. With three folds the variance on a small class stays wide; treat anything under ~100 held-out samples as indicative, not settled.

## Split

Protocol: **5-fold cross-validation over the five windows, plus one ablation fold that isolates what the current-layout material is worth**, declared 2026-09-02.
- fold A: train ['W2', 'W3', 'W4', 'W5'] → held out ['W1']
- fold B: train ['W1', 'W3', 'W4', 'W5'] → held out ['W2']
- fold C: train ['W1', 'W2', 'W4', 'W5'] → held out ['W3']
- fold D: train ['W1', 'W2', 'W3', 'W5'] → held out ['W4']
- fold E: train ['W1', 'W2', 'W3', 'W4'] → held out ['W5']
- fold E0-ablation: train ['W1', 'W2', 'W3'] → held out ['W5']

Per-activity accuracy is computed on the UNION of the three folds' held-out predictions - every labelled sample is predicted exactly once, by a model that never saw it. One confusion matrix per arm over that union, plus the per-fold matrices so a fold-specific collapse is visible.

## Delivery vocabulary: 3 categories

`pozostale` = `sciaganie_elementu` + `inna_czynnosc` + `postoj` + `brak_na_stanowisku` — declared in `manifest.source.json`.

Every figure in this section is over 3 categories and is **not comparable** with the per-arm sections below, which score all 7 separately. A merge cannot be undone by reading harder, so the two vocabularies never share a table.

`nierozpoznane` is **not** a member of the bucket and keeps its own row. It is neither work nor downtime, and folding the honest "cannot tell" into a work bucket would convert unknown time into measured time.

### Held-out union

#### `tcn-pixel-518`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 91.5% | 89.6% | 1.02× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.2% | 90.2% | 0.91× | 0.1 pp | ❌ |
| `pozostale` | 568 | 51.8% | 51.6% | 1.00× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 2.08× | 1.6 pp | — |

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither — and on this fixture they disagree. Cells are recall (time reported, n).

#### `tcn-pixel-518`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.15×, n=224) | 96.3% (1.04×, n=273) | 76.1% (1.04×, n=247) | 93.1% (0.96×, n=276) | 92.1% (0.94×, n=267) |
| `ukladanie_pretow` | 94.4% (1.02×, n=180) | 98.3% (1.15×, n=179) | 25.4% (0.27×, n=193) | 98.5% (1.16×, n=264) | 88.3% (0.91×, n=265) |
| `pozostale` | 22.1% (0.25×, n=195) | 62.2% (1.24×, n=90) | 67.9% (1.77×, n=156) | 45.8% (0.47×, n=59) | 91.2% (1.56×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (4.00×, n=4) | n/a | n/a |

## Arm: `tcn-pixel-518`

*2 of #120 - the C.0 model with pixels in place of geometry*

**Hardware verdict: OK** — 668 MiB peak on one card

**Cost: 29 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.3% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 91.5% | 89.6% | 1.02× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.2% | 90.2% | 0.91× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 56.0% | 65.8% | 0.85× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 30.6% | 22.8% | 1.34× | 0.5 pp | ❌ |
| `postoj` | 78 | 15.4% | 10.7% | 1.44× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 4.3% | 30.8% | 0.14× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 2.08× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 96.3% (n=273) | 76.1% (n=247) | 93.1% (n=276) | 92.1% (n=267) |
| `ukladanie_pretow` | 94.4% (n=180) | 98.3% (n=179) | 25.4% (n=193) | 98.5% (n=264) | 88.3% (n=265) |
| `sciaganie_elementu` | 77.3% (n=22) | 76.7% (n=30) | 22.4% (n=58) | 58.8% (n=34) | 76.3% (n=38) |
| `inna_czynnosc` | 12.3% (n=65) | 31.0% (n=42) | 37.1% (n=70) | 25.0% (n=20) | 73.7% (n=19) |
| `postoj` | 7.1% (n=42) | 18.8% (n=16) | 0.0% (n=4) | 20.0% (n=5) | 45.5% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 16.7% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 4 | 2 | 69 | 16 | 0 | 0 | 1 |
| `inna_czynnosc` | 0 | 66 | 19 | 6 | 11 | 81 | 33 |
| `nierozpoznane` | 7 | 29 | 0 | 15 | 2 | 4 | 5 |
| `postoj` | 1 | 17 | 25 | 12 | 6 | 2 | 15 |
| `sciaganie_elementu` | 0 | 27 | 2 | 24 | 102 | 11 | 16 |
| `spawanie` | 0 | 79 | 0 | 1 | 3 | 1177 | 27 |
| `ukladanie_pretow` | 1 | 70 | 14 | 38 | 31 | 38 | 889 |

### Boundary timing error

377 real activity changes, 542 predicted. Median error **1.0 s**, p90 6.0 s, max 28.0 s; 80.3% land within 2 s. Spurious boundaries (no real change within 4 s): **175**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arc-flash baseline on `spawanie`

Reported at **two operating points**, because a single threshold tells a misleading story about this signal. *Conservative* is the clip-relative cut-off the annotation hints used. *Oracle F1* is the best threshold available in hindsight on that same clip — in-sample, unavailable in production, and deliberately generous: an arm that costs a GPU should have to beat the baseline's best day, not a strawman.

**Conservative**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 1.13 | 44.2% | 90.0% | 0.49× | 0.593 |
| W2 | 1.68 | 37.4% | 96.2% | 0.39× | 0.538 |
| W3 | 3.65 | 37.7% | 33.7% | 1.12× | 0.356 |
| W4 | 2.38 | 36.6% | 95.3% | 0.38× | 0.529 |
| W5 | 1.69 | 38.6% | 99.0% | 0.39× | 0.555 |

Union on `spawanie`: recall **38.7%**, precision 70.9%, time reported 0.55×.

**Oracle F1 (in-sample)**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 0.05 | 99.1% | 44.7% | 2.22× | 0.616 |
| W2 | 0.12 | 99.6% | 46.3% | 2.15× | 0.633 |
| W3 | 0.04 | 93.1% | 44.4% | 2.10× | 0.601 |
| W4 | 0.39 | 100.0% | 46.2% | 2.17× | 0.632 |
| W5 | 1.32 | 53.2% | 95.3% | 0.56× | 0.683 |

Union on `spawanie`: recall **88.7%**, precision 48.6%, time reported 1.83×.

Cost: **0 GPU-seconds** at either point.

**The baseline clears the 85% recall bar on `spawanie` — and that is a finding about the bar, not about the baseline.** It reaches 88.7% recall by calling 1.83× as much time `spawanie` as actually was, at 48.6% precision. A recall-only bar is gameable by any arm willing to over-call the common class, so no arm should be promoted on recall alone. The `Time reported` column is what separates a measurement from a guess that happens to overlap the truth.

Either way the baseline is the **cost floor, not a candidate**: it cannot distinguish `ukladanie_pretow` from `postoj` at all, which is five of the seven activities and all of the hard part.

## Go / no-go

_Not generated. The bar is numeric but the decision is human — recorded as a comment on issue #117 by @tkowalczyk._
