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

## Arm: `tcn-pose`

**Hardware verdict: OK** — 710 MiB peak on one card

**Cost: 555 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 5.6% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 72.0% | 70.8% | 1.02× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 552 | 85.3% | 69.7% | 1.22× | 0.2 pp | ✅ |
| `sciaganie_elementu` | 110 | 46.4% | 72.9% | 0.64× | 0.9 pp | ❌ |
| `inna_czynnosc` | 177 | 13.6% | 23.5% | 0.58× | 0.6 pp | ❌ |
| `postoj` | 62 | 1.6% | 1.2% | 1.29× | 1.6 pp | ❌ |
| `brak_na_stanowisku` | 92 | 9.8% | 64.3% | 0.15× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 4.8% | 3.0% | 1.61× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 97.8% (n=224) | 91.2% (n=273) | 27.5% (n=247) |
| `ukladanie_pretow` | 93.9% (n=180) | 98.9% (n=179) | 64.8% (n=193) |
| `sciaganie_elementu` | 77.3% (n=22) | 66.7% (n=30) | 24.1% (n=58) |
| `inna_czynnosc` | 12.3% (n=65) | 31.0% (n=42) | 4.3% (n=70) |
| `postoj` | 0.0% (n=42) | 6.2% (n=16) | 0.0% (n=4) |
| `brak_na_stanowisku` | 4.5% (n=66) | 0.0% (n=2) | 25.0% (n=24) |

**Fails the bar on:** `spawanie`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 9 | 2 | 9 | 6 | 0 | 63 | 3 |
| `inna_czynnosc` | 0 | 24 | 24 | 13 | 8 | 60 | 48 |
| `nierozpoznane` | 0 | 25 | 3 | 30 | 3 | 0 | 1 |
| `postoj` | 0 | 8 | 28 | 1 | 1 | 18 | 6 |
| `sciaganie_elementu` | 0 | 5 | 3 | 4 | 51 | 26 | 21 |
| `spawanie` | 5 | 24 | 26 | 21 | 6 | 536 | 126 |
| `ukladanie_pretow` | 0 | 14 | 7 | 5 | 1 | 54 | 471 |

### Boundary timing error

249 real activity changes, 382 predicted. Median error **0.0 s**, p90 10.0 s, max 24.0 s; 79.5% land within 2 s. Spurious boundaries (no real change within 4 s): **124**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-rich`

**Hardware verdict: UNMEASURED** — `peak_vram_mib` missing

**Cost: 10 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 6.6% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 78.8% | 75.5% | 1.04× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 552 | 79.2% | 74.4% | 1.06× | 0.2 pp | ❌ |
| `sciaganie_elementu` | 110 | 36.4% | 69.0% | 0.53× | 0.9 pp | ❌ |
| `inna_czynnosc` | 177 | 16.4% | 21.0% | 0.78× | 0.6 pp | ❌ |
| `postoj` | 62 | 4.8% | 2.8% | 1.73× | 1.6 pp | ❌ |
| `brak_na_stanowisku` | 92 | 6.5% | 40.0% | 0.16× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.90× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 97.3% (n=224) | 92.7% (n=273) | 46.6% (n=247) |
| `ukladanie_pretow` | 91.1% (n=180) | 96.6% (n=179) | 51.8% (n=193) |
| `sciaganie_elementu` | 68.2% (n=22) | 80.0% (n=30) | 1.7% (n=58) |
| `inna_czynnosc` | 16.9% (n=65) | 35.7% (n=42) | 4.3% (n=70) |
| `postoj` | 0.0% (n=42) | 18.8% (n=16) | 0.0% (n=4) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 25.0% (n=24) |

**Fails the bar on:** `spawanie`, `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 6 | 0 | 66 | 14 | 0 | 6 | 0 |
| `inna_czynnosc` | 0 | 29 | 17 | 20 | 13 | 62 | 36 |
| `nierozpoznane` | 5 | 28 | 0 | 25 | 3 | 0 | 1 |
| `postoj` | 0 | 13 | 27 | 3 | 1 | 13 | 5 |
| `sciaganie_elementu` | 0 | 9 | 3 | 5 | 40 | 27 | 26 |
| `spawanie` | 3 | 40 | 4 | 29 | 0 | 586 | 82 |
| `ukladanie_pretow` | 1 | 19 | 1 | 11 | 1 | 82 | 437 |

### Boundary timing error

249 real activity changes, 408 predicted. Median error **0.0 s**, p90 8.0 s, max 22.0 s; 84.4% land within 2 s. Spurious boundaries (no real change within 4 s): **133**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `vlm-qwen2.5-vl-3b-v1`

**Hardware verdict: OK** — 7542 MiB peak on one card

**Cost: 701 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 0.0% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 95.8% | 45.5% | 2.11× | 0.1 pp | ⚠️ gamed |
| `ukladanie_pretow` | 552 | 0.4% | 100.0% | 0.00× | 0.2 pp | ❌ |
| `sciaganie_elementu` | 110 | 5.5% | 8.5% | 0.65× | 0.9 pp | ❌ |
| `inna_czynnosc` | 177 | 11.9% | 13.4% | 0.89× | 0.6 pp | ❌ |
| `postoj` | 62 | 3.2% | 100.0% | 0.03× | 1.6 pp | ❌ |
| `brak_na_stanowisku` | 92 | 0.0% | n/a | 0.00× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | n/a | 0.00× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 98.7% (n=224) | 91.9% (n=273) | 97.6% (n=247) |
| `ukladanie_pretow` | 1.1% (n=180) | 0.0% (n=179) | 0.0% (n=193) |
| `sciaganie_elementu` | 9.1% (n=22) | 10.0% (n=30) | 1.7% (n=58) |
| `inna_czynnosc` | 26.2% (n=65) | 9.5% (n=42) | 0.0% (n=70) |
| `postoj` | 4.8% (n=42) | 0.0% (n=16) | 0.0% (n=4) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 0.0% (n=24) |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.
**Passes but unusable on:** `spawanie` — recall bought by over-calling the class.

### Confusion matrix

| truth ↓ / pred → | `inna_czynnosc` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|
| `brak_na_stanowisku` | 67 | 0 | 3 | 22 | 0 |
| `inna_czynnosc` | 21 | 0 | 7 | 149 | 0 |
| `nierozpoznane` | 25 | 0 | 7 | 30 | 0 |
| `postoj` | 33 | 2 | 0 | 27 | 0 |
| `sciaganie_elementu` | 2 | 0 | 6 | 102 | 0 |
| `spawanie` | 4 | 0 | 27 | 713 | 0 |
| `ukladanie_pretow` | 5 | 0 | 21 | 524 | 2 |

### Boundary timing error

249 real activity changes, 215 predicted. Median error **4.0 s**, p90 140.0 s, max 186.0 s; 34.2% land within 2 s. Spurious boundaries (no real change within 4 s): **93**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `vlm-qwen2.5-vl-3b-v2`

**Hardware verdict: OK** — 7554 MiB peak on one card

**Cost: 774 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 0.0% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 744 | 99.6% | 47.8% | 2.08× | 0.1 pp | ⚠️ gamed |
| `ukladanie_pretow` | 552 | 0.0% | n/a | 0.00× | 0.2 pp | ❌ |
| `sciaganie_elementu` | 110 | 0.0% | n/a | 0.00× | 0.9 pp | ❌ |
| `inna_czynnosc` | 177 | 0.0% | n/a | 0.00× | 0.6 pp | ❌ |
| `postoj` | 62 | 0.0% | n/a | 0.00× | 1.6 pp | ❌ |
| `brak_na_stanowisku` | 92 | 100.0% | 36.8% | 2.72× | 1.1 pp | ⚠️ gamed |
| `nierozpoznane` | 62 | 0.0% | n/a | 0.00× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 |
|---|---:|---:|---:|
| `spawanie` | 100.0% (n=224) | 98.9% (n=273) | 100.0% (n=247) |
| `ukladanie_pretow` | 0.0% (n=180) | 0.0% (n=179) | 0.0% (n=193) |
| `sciaganie_elementu` | 0.0% (n=22) | 0.0% (n=30) | 0.0% (n=58) |
| `inna_czynnosc` | 0.0% (n=65) | 0.0% (n=42) | 0.0% (n=70) |
| `postoj` | 0.0% (n=42) | 0.0% (n=16) | 0.0% (n=4) |
| `brak_na_stanowisku` | 100.0% (n=66) | 100.0% (n=2) | 100.0% (n=24) |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`. An 84% class fails even if the average clears.
**Passes but unusable on:** `spawanie`, `brak_na_stanowisku` — recall bought by over-calling the class.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `spawanie` |
|---|---|---|
| `brak_na_stanowisku` | 92 | 0 |
| `inna_czynnosc` | 31 | 146 |
| `nierozpoznane` | 61 | 1 |
| `postoj` | 42 | 20 |
| `sciaganie_elementu` | 13 | 97 |
| `spawanie` | 3 | 741 |
| `ukladanie_pretow` | 8 | 544 |

### Boundary timing error

249 real activity changes, 103 predicted. Median error **4.0 s**, p90 42.0 s, max 84.0 s; 40.2% land within 2 s. Spurious boundaries (no real change within 4 s): **15**.

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
