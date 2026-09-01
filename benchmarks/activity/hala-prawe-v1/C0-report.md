# C.0 measurement report — `hala-prawe-v1`

Generated 2026-09-01 by `benchmarks/activity/tools/evaluate_arms.py`.

## Pre-break bias

**Both annotated windows are pre-break** (W1 09:00–09:20, W2 10:20–10:40, both 2026-08-28; W3 was dropped by decision on 2026-09-01). Every figure in this report therefore describes the first half of a shift only, and inherits that bias permanently. Quote it with the caveat attached or do not quote it.

## Split

Protocol: **2-fold cross-validation over the two windows**, declared 2026-09-01.
- fold A: train ['W1'] → held out ['W2']
- fold B: train ['W2'] → held out ['W1']

Per-activity accuracy is computed on the UNION of the two folds' held-out predictions - every labelled sample is predicted exactly once, by a model that never saw it. One confusion matrix per arm over that union, plus the per-fold matrices so a fold-specific collapse is visible.

## Arm: `tcn-pose`

**Hardware verdict: OK** — 710 MiB peak on one card

**Cost: 445 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 9.9% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 497 | 94.6% | 93.1% | 1.02× | 0.2 pp | ✅ |
| `ukladanie_pretow` | 359 | 96.4% | 91.5% | 1.05× | 0.3 pp | ✅ |
| `sciaganie_elementu` | 52 | 73.1% | 70.4% | 1.04× | 1.9 pp | ❌ |
| `inna_czynnosc` | 107 | 26.2% | 32.6% | 0.80× | 0.9 pp | ❌ |
| `postoj` | 58 | 12.1% | 12.5% | 0.97× | 1.7 pp | ❌ |
| `brak_na_stanowisku` | 68 | 0.0% | 0.0% | 0.01× | 1.5 pp | ❌ |
| `nierozpoznane` | 58 | 0.0% | 0.0% | 2.05× | 1.7 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

**Fails the bar on:** `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 0 | 0 | 66 | 2 | 0 | 0 | 0 |
| `inna_czynnosc` | 0 | 28 | 16 | 12 | 11 | 28 | 12 |
| `nierozpoznane` | 1 | 18 | 0 | 34 | 1 | 0 | 4 |
| `postoj` | 0 | 6 | 34 | 7 | 2 | 1 | 8 |
| `sciaganie_elementu` | 0 | 6 | 1 | 0 | 38 | 2 | 5 |
| `spawanie` | 0 | 22 | 1 | 0 | 1 | 470 | 3 |
| `ukladanie_pretow` | 0 | 6 | 1 | 1 | 1 | 4 | 346 |

### Boundary timing error

175 real activity changes, 186 predicted. Median error **0.0 s**, p90 8.0 s, max 18.0 s; 83.7% land within 2 s. Spurious boundaries (no real change within 4 s): **17**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-rich`

**Hardware verdict: UNMEASURED** — `peak_vram_mib` missing

**Cost: 5 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 10.1% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 497 | 95.4% | 93.9% | 1.02× | 0.2 pp | ✅ |
| `ukladanie_pretow` | 359 | 95.3% | 89.5% | 1.06× | 0.3 pp | ✅ |
| `sciaganie_elementu` | 52 | 73.1% | 74.5% | 0.98× | 1.9 pp | ❌ |
| `inna_czynnosc` | 107 | 27.1% | 31.9% | 0.85× | 0.9 pp | ❌ |
| `postoj` | 58 | 6.9% | 8.3% | 0.83× | 1.7 pp | ❌ |
| `brak_na_stanowisku` | 68 | 0.0% | 0.0% | 0.01× | 1.5 pp | ❌ |
| `nierozpoznane` | 58 | 0.0% | 0.0% | 2.09× | 1.7 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

**Fails the bar on:** `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 0 | 1 | 66 | 1 | 0 | 0 | 0 |
| `inna_czynnosc` | 0 | 29 | 17 | 10 | 9 | 23 | 19 |
| `nierozpoznane` | 1 | 22 | 0 | 32 | 1 | 0 | 2 |
| `postoj` | 0 | 9 | 36 | 4 | 2 | 2 | 5 |
| `sciaganie_elementu` | 0 | 6 | 1 | 0 | 38 | 2 | 5 |
| `spawanie` | 0 | 14 | 0 | 0 | 0 | 474 | 9 |
| `ukladanie_pretow` | 0 | 10 | 1 | 1 | 1 | 4 | 342 |

### Boundary timing error

175 real activity changes, 200 predicted. Median error **0.0 s**, p90 8.0 s, max 18.0 s; 85.5% land within 2 s. Spurious boundaries (no real change within 4 s): **25**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `vlm-qwen2.5-vl-3b`

**Hardware verdict: OK** — 7542 MiB peak on one card

**Cost: 701 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 0.0% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 497 | 95.0% | 48.2% | 1.97× | 0.2 pp | ⚠️ gamed |
| `ukladanie_pretow` | 359 | 0.6% | 100.0% | 0.01× | 0.3 pp | ❌ |
| `sciaganie_elementu` | 52 | 9.6% | 8.1% | 1.19× | 1.9 pp | ❌ |
| `inna_czynnosc` | 107 | 19.6% | 13.6% | 1.44× | 0.9 pp | ❌ |
| `postoj` | 58 | 3.4% | 100.0% | 0.03× | 1.7 pp | ❌ |
| `brak_na_stanowisku` | 68 | 0.0% | n/a | 0.00× | 1.5 pp | ❌ |
| `nierozpoznane` | 58 | 0.0% | n/a | 0.00× | 1.7 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.
**Passes but unusable on:** `spawanie` — recall bought by over-calling the class.

### Confusion matrix

| truth ↓ / pred → | `inna_czynnosc` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|
| `brak_na_stanowisku` | 66 | 0 | 2 | 0 | 0 |
| `inna_czynnosc` | 21 | 0 | 6 | 80 | 0 |
| `nierozpoznane` | 25 | 0 | 7 | 26 | 0 |
| `postoj` | 33 | 2 | 0 | 23 | 0 |
| `sciaganie_elementu` | 2 | 0 | 5 | 45 | 0 |
| `spawanie` | 3 | 0 | 22 | 472 | 0 |
| `ukladanie_pretow` | 4 | 0 | 20 | 333 | 2 |

### Boundary timing error

175 real activity changes, 196 predicted. Median error **4.0 s**, p90 28.0 s, max 62.0 s; 48.6% land within 2 s. Spurious boundaries (no real change within 4 s): **82**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `vlm-qwen2.5-vl-3b-v2`

**Hardware verdict: OK** — 7554 MiB peak on one card

**Cost: 754 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 0.0% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 273 | 98.9% | 52.3% | 1.89× | 0.4 pp | ⚠️ gamed |
| `ukladanie_pretow` | 179 | 0.0% | n/a | 0.00× | 0.6 pp | ❌ |
| `sciaganie_elementu` | 30 | 0.0% | n/a | 0.00× | 3.3 pp | ❌ |
| `inna_czynnosc` | 42 | 0.0% | n/a | 0.00× | 2.4 pp | ❌ |
| `postoj` | 16 | 0.0% | n/a | 0.00× | 6.2 pp | ❌ |
| `brak_na_stanowisku` | 2 | 100.0% | 2.4% | 42.00× | 50.0 pp | ⚠️ gamed |
| `nierozpoznane` | 58 | 0.0% | n/a | 0.00× | 1.7 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`. An 84% class fails even if the average clears.
**Passes but unusable on:** `spawanie`, `brak_na_stanowisku` — recall bought by over-calling the class.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `spawanie` |
|---|---|---|
| `brak_na_stanowisku` | 2 | 0 |
| `inna_czynnosc` | 10 | 32 |
| `nierozpoznane` | 57 | 1 |
| `postoj` | 6 | 10 |
| `sciaganie_elementu` | 3 | 27 |
| `spawanie` | 3 | 270 |
| `ukladanie_pretow` | 3 | 176 |

### Boundary timing error

91 real activity changes, 45 predicted. Median error **4.0 s**, p90 32.0 s, max 64.0 s; 47.3% land within 2 s. Spurious boundaries (no real change within 4 s): **7**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arc-flash baseline on `spawanie`

Reported at **two operating points**, because a single threshold tells a misleading story about this signal. *Conservative* is the clip-relative cut-off the annotation hints used. *Oracle F1* is the best threshold available in hindsight on that same clip — in-sample, unavailable in production, and deliberately generous: an arm that costs a GPU should have to beat the baseline's best day, not a strawman.

**Conservative**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 1.13 | 44.2% | 90.0% | 0.49× | 0.593 |
| W2 | 1.68 | 37.4% | 96.2% | 0.39× | 0.538 |

Union on `spawanie`: recall **40.4%**, precision 93.1%, time reported 0.43×.

**Oracle F1 (in-sample)**

| Window | Threshold | Recall | Precision | Time reported | F1 |
|---|---:|---:|---:|---:|---:|
| W1 | 0.05 | 99.1% | 44.7% | 2.22× | 0.616 |
| W2 | 0.12 | 99.6% | 46.3% | 2.15× | 0.633 |

Union on `spawanie`: recall **99.4%**, precision 45.6%, time reported 2.18×.

Cost: **0 GPU-seconds** at either point.

**The baseline clears the 85% recall bar on `spawanie` — and that is a finding about the bar, not about the baseline.** It reaches 99.4% recall by calling 2.18× as much time `spawanie` as actually was, at 45.6% precision. A recall-only bar is gameable by any arm willing to over-call the common class, so no arm should be promoted on recall alone. The `Time reported` column is what separates a measurement from a guess that happens to overlap the truth.

Either way the baseline is the **cost floor, not a candidate**: it cannot distinguish `ukladanie_pretow` from `postoj` at all, which is five of the seven activities and all of the hard part.

## Go / no-go

_Not generated. The bar is numeric but the decision is human — recorded as a comment on issue #117 by @tkowalczyk._
