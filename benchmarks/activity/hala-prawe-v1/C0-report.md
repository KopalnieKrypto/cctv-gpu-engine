# C.0 measurement report — `hala-prawe-v1`

Generated 2026-09-01 by `benchmarks/activity/tools/evaluate_arms.py`.

## Pre-break bias

**Both annotated windows are pre-break** (W1 09:00–09:20, W2 10:20–10:40, both 2026-08-28; W3 was dropped by decision on 2026-09-01). Every figure in this report therefore describes the first half of a shift only, and inherits that bias permanently. Quote it with the caveat attached or do not quote it.

## Split

Protocol: **2-fold cross-validation over the two windows**, declared 2026-09-01.
- fold A: train ['W1'] → held out ['W2']
- fold B: train ['W2'] → held out ['W1']

Per-activity accuracy is computed on the UNION of the two folds' held-out predictions - every labelled sample is predicted exactly once, by a model that never saw it. One confusion matrix per arm over that union, plus the per-fold matrices so a fold-specific collapse is visible.

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
