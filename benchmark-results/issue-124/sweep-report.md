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

#### `tcn-pixel-518-d0.03`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 83.4% | 87.9% | 0.95× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 63.2% | 83.1% | 0.76× | 0.1 pp | ❌ |
| `pozostale` | 568 | 60.7% | 38.9% | 1.56× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.10× | 1.6 pp | — |

#### `tcn-pixel-518-d0.045`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 81.7% | 86.0% | 0.95× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 61.1% | 83.8% | 0.73× | 0.1 pp | ❌ |
| `pozostale` | 568 | 61.8% | 38.9% | 1.59× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.35× | 1.6 pp | — |

#### `tcn-pixel-518-d0.06`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 78.9% | 92.0% | 0.86× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 74.7% | 77.0% | 0.97× | 0.1 pp | ❌ |
| `pozostale` | 568 | 60.7% | 45.1% | 1.35× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 1.3% | 1.27× | 1.6 pp | — |

#### `tcn-pixel-518-d0.09`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 83.5% | 91.9% | 0.91× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 84.3% | 90.0% | 0.94× | 0.1 pp | ❌ |
| `pozostale` | 568 | 60.7% | 49.9% | 1.22× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 0.8% | 2.00× | 1.6 pp | — |

#### `tcn-pixel-518-d0.125`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.3% | 84.9% | 1.09× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.6% | 90.3% | 0.91× | 0.1 pp | ❌ |
| `pozostale` | 568 | 46.7% | 53.2% | 0.88× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.81× | 1.6 pp | — |

#### `tcn-pixel-518-d0.18`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.9% | 84.1% | 1.10× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 78.7% | 92.4% | 0.85× | 0.1 pp | ❌ |
| `pozostale` | 568 | 50.0% | 53.0% | 0.94× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.92× | 1.6 pp | — |

#### `tcn-pixel-518-d0.25`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.5% | 88.5% | 1.05× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 80.9% | 92.7% | 0.87× | 0.1 pp | ❌ |
| `pozostale` | 568 | 54.4% | 55.2% | 0.99× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 0.7% | 2.39× | 1.6 pp | — |

#### `tcn-pixel-518-d0.35`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 93.9% | 87.3% | 1.08× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.5% | 89.4% | 0.92× | 0.1 pp | ❌ |
| `pozostale` | 568 | 51.9% | 61.6% | 0.84× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 3.2% | 1.5% | 2.21× | 1.6 pp | — |

#### `tcn-pixel-518-d0.5`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 95.3% | 90.8% | 1.05× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 84.8% | 92.9% | 0.91× | 0.1 pp | ❌ |
| `pozostale` | 568 | 51.4% | 58.9% | 0.87× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 4.8% | 1.8% | 2.63× | 1.6 pp | — |

#### `tcn-pixel-518-d0.75`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 94.3% | 91.8% | 1.03× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 83.6% | 87.9% | 0.95× | 0.1 pp | ❌ |
| `pozostale` | 568 | 53.2% | 57.7% | 0.92× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 6.5% | 3.2% | 2.02× | 1.6 pp | — |

#### `tcn-pixel-518-d1`

| Category | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 90.5% | 93.1% | 0.97× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 86.4% | 89.6% | 0.96× | 0.1 pp | ✅ |
| `pozostale` | 568 | 57.7% | 55.3% | 1.04× | 0.2 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.79× | 1.6 pp | — |

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither — and on this fixture they disagree. Cells are recall (time reported, n).

#### `tcn-pixel-518-d0.03`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (1.22×, n=224) | 96.7% (1.14×, n=273) | 37.7% (0.55×, n=247) | 92.0% (0.93×, n=276) | 90.3% (0.91×, n=267) |
| `ukladanie_pretow` | 62.8% (0.78×, n=180) | 88.8% (1.05×, n=179) | 30.1% (0.50×, n=193) | 38.6% (0.41×, n=264) | 94.7% (1.09×, n=265) |
| `pozostale` | 45.6% (0.73×, n=195) | 55.6% (1.11×, n=90) | 68.6% (2.25×, n=156) | 93.2% (3.95×, n=59) | 64.7% (0.88×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (4.25×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.045`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.21×, n=224) | 96.7% (1.15×, n=273) | 28.3% (0.53×, n=247) | 90.9% (0.95×, n=276) | 91.4% (0.91×, n=267) |
| `ukladanie_pretow` | 75.6% (0.80×, n=180) | 79.3% (0.97×, n=179) | 0.5% (0.01×, n=193) | 44.3% (0.46×, n=264) | 100.0% (1.31×, n=265) |
| `pozostale` | 53.3% (0.68×, n=195) | 64.4% (1.26×, n=90) | 82.7% (2.84×, n=156) | 89.8% (3.49×, n=59) | 10.3% (0.12×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (6.00×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.06`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 98.2% (1.13×, n=224) | 97.8% (1.07×, n=273) | 37.7% (0.43×, n=247) | 91.7% (0.97×, n=276) | 68.5% (0.69×, n=267) |
| `ukladanie_pretow` | 79.4% (0.88×, n=180) | 97.2% (1.26×, n=179) | 4.7% (0.11×, n=193) | 82.2% (0.89×, n=264) | 100.0% (1.54×, n=265) |
| `pozostale` | 62.6% (0.81×, n=195) | 52.2% (0.91×, n=90) | 81.4% (2.71×, n=156) | 78.0% (1.61×, n=59) | 4.4% (0.12×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 25.0% (12.25×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.09`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.18×, n=224) | 97.1% (1.06×, n=273) | 25.1% (0.32×, n=247) | 97.5% (1.01×, n=276) | 95.9% (0.97×, n=267) |
| `ukladanie_pretow` | 95.0% (1.02×, n=180) | 84.9% (1.01×, n=179) | 50.3% (0.65×, n=193) | 91.7% (0.96×, n=264) | 94.0% (1.02×, n=265) |
| `pozostale` | 49.7% (0.50×, n=195) | 68.9% (1.46×, n=90) | 60.3% (2.13×, n=156) | 74.6% (1.00×, n=59) | 70.6% (1.04×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 25.0% (16.00×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.125`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.20×, n=224) | 94.9% (0.95×, n=273) | 78.9% (1.43×, n=247) | 93.5% (0.95×, n=276) | 94.8% (0.96×, n=267) |
| `ukladanie_pretow` | 94.4% (1.01×, n=180) | 93.3% (1.07×, n=179) | 31.1% (0.46×, n=193) | 97.0% (1.06×, n=264) | 90.6% (0.93×, n=265) |
| `pozostale` | 19.5% (0.21×, n=195) | 74.4% (1.66×, n=90) | 34.0% (0.99×, n=156) | 76.3% (0.97×, n=59) | 91.2% (1.43×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (1.25×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.18`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.14×, n=224) | 91.6% (0.92×, n=273) | 92.3% (1.68×, n=247) | 88.8% (0.90×, n=276) | 93.6% (0.95×, n=267) |
| `ukladanie_pretow` | 95.6% (0.99×, n=180) | 94.4% (1.06×, n=179) | 17.1% (0.19×, n=193) | 98.9% (1.11×, n=264) | 81.5% (0.84×, n=265) |
| `pozostale` | 23.1% (0.26×, n=195) | 84.4% (1.78×, n=90) | 37.2% (0.96×, n=156) | 76.3% (0.98×, n=59) | 88.2% (1.75×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (0.00×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.25`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 98.2% (1.11×, n=224) | 93.0% (0.96×, n=273) | 85.4% (1.30×, n=247) | 90.9% (0.93×, n=276) | 95.5% (0.97×, n=267) |
| `ukladanie_pretow` | 93.3% (0.98×, n=180) | 93.9% (1.05×, n=179) | 17.1% (0.17×, n=193) | 98.1% (1.11×, n=264) | 93.2% (0.96×, n=265) |
| `pozostale` | 30.8% (0.37×, n=195) | 74.4% (1.67×, n=90) | 48.7% (1.29×, n=156) | 72.9% (0.85×, n=59) | 92.6% (1.28×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 25.0% (11.00×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.35`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (1.13×, n=224) | 97.8% (1.00×, n=273) | 95.1% (1.45×, n=247) | 83.7% (0.86×, n=276) | 94.8% (0.97×, n=267) |
| `ukladanie_pretow` | 96.1% (1.05×, n=180) | 94.4% (1.06×, n=179) | 23.3% (0.25×, n=193) | 99.2% (1.23×, n=264) | 91.7% (0.94×, n=265) |
| `pozostale` | 28.7% (0.31×, n=195) | 78.9% (1.52×, n=90) | 50.0% (0.99×, n=156) | 49.2% (0.63×, n=59) | 89.7% (1.32×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 50.0% (9.50×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.5`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.14×, n=224) | 96.3% (0.98×, n=273) | 88.3% (1.20×, n=247) | 96.7% (0.98×, n=276) | 95.9% (0.98×, n=267) |
| `ukladanie_pretow` | 94.4% (1.01×, n=180) | 98.3% (1.16×, n=179) | 42.0% (0.43×, n=193) | 97.7% (1.06×, n=264) | 87.5% (0.89×, n=265) |
| `pozostale` | 23.6% (0.26×, n=195) | 66.7% (1.40×, n=90) | 51.9% (1.08×, n=156) | 72.9% (0.83×, n=59) | 91.2% (1.49×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 75.0% (13.00×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d0.75`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (1.14×, n=224) | 92.7% (0.95×, n=273) | 95.1% (1.19×, n=247) | 92.4% (0.95×, n=276) | 93.3% (0.94×, n=267) |
| `ukladanie_pretow` | 94.4% (1.04×, n=180) | 100.0% (1.30×, n=179) | 27.5% (0.31×, n=193) | 98.5% (1.14×, n=264) | 91.3% (0.94×, n=265) |
| `pozostale` | 27.7% (0.31×, n=195) | 58.9% (1.21×, n=90) | 62.8% (1.38×, n=156) | 57.6% (0.63×, n=59) | 92.6% (1.47×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 100.0% (7.25×, n=4) | n/a | n/a |

#### `tcn-pixel-518-d1`

| Category | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (1.14×, n=224) | 95.2% (0.98×, n=273) | 69.6% (0.85×, n=247) | 91.3% (0.92×, n=276) | 96.6% (0.99×, n=267) |
| `ukladanie_pretow` | 94.4% (1.02×, n=180) | 98.3% (1.19×, n=179) | 40.9% (0.42×, n=193) | 98.9% (1.18×, n=264) | 93.6% (0.95×, n=265) |
| `pozostale` | 23.6% (0.27×, n=195) | 66.7% (1.32×, n=90) | 84.0% (1.97×, n=156) | 49.2% (0.56×, n=59) | 91.2% (1.21×, n=68) |
| `nierozpoznane` | n/a | 0.0% (0.00×, n=58) | 0.0% (0.75×, n=4) | n/a | n/a |

## Arm: `tcn-pixel-518-d0.03`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 258 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 2.3% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 83.4% | 87.9% | 0.95× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 63.2% | 83.1% | 0.76× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 42.3% | 25.5% | 1.66× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 19.9% | 16.7% | 1.19× | 0.5 pp | ❌ |
| `postoj` | 78 | 14.1% | 6.2% | 2.26× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 9.8% | 6.0% | 1.64× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.10× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (n=224) | 96.7% (n=273) | 37.7% (n=247) | 92.0% (n=276) | 90.3% (n=267) |
| `ukladanie_pretow` | 62.8% (n=180) | 88.8% (n=179) | 30.1% (n=193) | 38.6% (n=264) | 94.7% (n=265) |
| `sciaganie_elementu` | 31.8% (n=22) | 53.3% (n=30) | 5.2% (n=58) | 88.2% (n=34) | 55.3% (n=38) |
| `inna_czynnosc` | 21.5% (n=65) | 35.7% (n=42) | 7.1% (n=70) | 5.0% (n=20) | 42.1% (n=19) |
| `postoj` | 11.9% (n=42) | 0.0% (n=16) | 0.0% (n=4) | 40.0% (n=5) | 36.4% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 37.5% (n=24) | n/a | n/a |

**Fails the bar on:** `spawanie`, `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 9 | 4 | 2 | 20 | 25 | 0 | 32 |
| `inna_czynnosc` | 20 | 43 | 22 | 25 | 21 | 56 | 29 |
| `nierozpoznane` | 2 | 25 | 0 | 5 | 2 | 16 | 12 |
| `postoj` | 3 | 19 | 23 | 11 | 7 | 9 | 6 |
| `sciaganie_elementu` | 20 | 25 | 2 | 16 | 77 | 12 | 30 |
| `spawanie` | 58 | 35 | 9 | 52 | 29 | 1074 | 30 |
| `ukladanie_pretow` | 39 | 106 | 10 | 47 | 141 | 55 | 683 |

### Boundary timing error

377 real activity changes, 779 predicted. Median error **0.0 s**, p90 6.0 s, max 32.0 s; 84.8% land within 2 s. Spurious boundaries (no real change within 4 s): **326**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.045`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 258 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 2.8% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 81.7% | 86.0% | 0.95× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 61.1% | 83.8% | 0.73× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 40.1% | 36.3% | 1.10× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 27.8% | 22.5% | 1.24× | 0.5 pp | ❌ |
| `postoj` | 78 | 11.5% | 9.1% | 1.27× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 26.1% | 7.2% | 3.64× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.35× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 96.7% (n=273) | 28.3% (n=247) | 90.9% (n=276) | 91.4% (n=267) |
| `ukladanie_pretow` | 75.6% (n=180) | 79.3% (n=179) | 0.5% (n=193) | 44.3% (n=264) | 100.0% (n=265) |
| `sciaganie_elementu` | 68.2% (n=22) | 63.3% (n=30) | 3.4% (n=58) | 91.2% (n=34) | 15.8% (n=38) |
| `inna_czynnosc` | 20.0% (n=65) | 59.5% (n=42) | 24.3% (n=70) | 25.0% (n=20) | 0.0% (n=19) |
| `postoj` | 16.7% (n=42) | 0.0% (n=16) | 0.0% (n=4) | 40.0% (n=5) | 0.0% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 100.0% (n=24) | n/a | n/a |

**Fails the bar on:** `spawanie`, `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 24 | 2 | 1 | 60 | 0 | 0 | 5 |
| `inna_czynnosc` | 35 | 60 | 17 | 1 | 18 | 56 | 29 |
| `nierozpoznane` | 4 | 22 | 0 | 6 | 2 | 12 | 16 |
| `postoj` | 3 | 7 | 25 | 9 | 7 | 11 | 16 |
| `sciaganie_elementu` | 33 | 19 | 2 | 0 | 73 | 19 | 36 |
| `spawanie` | 104 | 63 | 26 | 0 | 16 | 1052 | 26 |
| `ukladanie_pretow` | 132 | 94 | 13 | 23 | 85 | 73 | 661 |

### Boundary timing error

377 real activity changes, 652 predicted. Median error **0.0 s**, p90 14.0 s, max 42.0 s; 73.9% land within 2 s. Spurious boundaries (no real change within 4 s): **298**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.06`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 260 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 2.6% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 78.9% | 92.0% | 0.86× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 74.7% | 77.0% | 0.97× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 39.0% | 38.0% | 1.03× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 19.4% | 17.8% | 1.09× | 0.5 pp | ❌ |
| `postoj` | 78 | 23.1% | 12.8% | 1.81× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 9.8% | 4.5% | 2.18× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 1.3% | 1.27× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 98.2% (n=224) | 97.8% (n=273) | 37.7% (n=247) | 91.7% (n=276) | 68.5% (n=267) |
| `ukladanie_pretow` | 79.4% (n=180) | 97.2% (n=179) | 4.7% (n=193) | 82.2% (n=264) | 100.0% (n=265) |
| `sciaganie_elementu` | 31.8% (n=22) | 83.3% (n=30) | 17.2% (n=58) | 85.3% (n=34) | 0.0% (n=38) |
| `inna_czynnosc` | 24.6% (n=65) | 19.0% (n=42) | 25.7% (n=70) | 0.0% (n=20) | 0.0% (n=19) |
| `postoj` | 21.4% (n=42) | 18.8% (n=16) | 0.0% (n=4) | 60.0% (n=5) | 27.3% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 37.5% (n=24) | n/a | n/a |

**Fails the bar on:** `spawanie`, `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 9 | 8 | 11 | 61 | 0 | 0 | 3 |
| `inna_czynnosc` | 29 | 42 | 13 | 5 | 30 | 47 | 50 |
| `nierozpoznane` | 3 | 9 | 1 | 16 | 5 | 6 | 22 |
| `postoj` | 2 | 14 | 19 | 18 | 2 | 2 | 21 |
| `sciaganie_elementu` | 19 | 23 | 5 | 12 | 71 | 4 | 48 |
| `spawanie` | 26 | 76 | 26 | 18 | 27 | 1016 | 98 |
| `ukladanie_pretow` | 113 | 64 | 4 | 11 | 52 | 29 | 808 |

### Boundary timing error

377 real activity changes, 624 predicted. Median error **1.0 s**, p90 22.0 s, max 32.0 s; 72.0% land within 2 s. Spurious boundaries (no real change within 4 s): **254**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.09`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1342 MiB peak on one card

**Cost: 264 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.1% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 83.5% | 91.9% | 0.91× | 0.1 pp | ❌ |
| `ukladanie_pretow` | 1081 | 84.3% | 90.0% | 0.94× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 39.6% | 52.9% | 0.75× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 17.1% | 24.2% | 0.71× | 0.5 pp | ❌ |
| `postoj` | 78 | 24.4% | 8.6% | 2.85× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 5.4% | 2.8% | 1.97× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 0.8% | 2.00× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 97.1% (n=273) | 25.1% (n=247) | 97.5% (n=276) | 95.9% (n=267) |
| `ukladanie_pretow` | 95.0% (n=180) | 84.9% (n=179) | 50.3% (n=193) | 91.7% (n=264) | 94.0% (n=265) |
| `sciaganie_elementu` | 77.3% (n=22) | 50.0% (n=30) | 0.0% (n=58) | 41.2% (n=34) | 68.4% (n=38) |
| `inna_czynnosc` | 4.6% (n=65) | 54.8% (n=42) | 0.0% (n=70) | 10.0% (n=20) | 47.4% (n=19) |
| `postoj` | 11.9% (n=42) | 25.0% (n=16) | 50.0% (n=4) | 80.0% (n=5) | 36.4% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 20.8% (n=24) | n/a | n/a |

**Fails the bar on:** `spawanie`, `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 5 | 18 | 19 | 7 | 38 | 0 | 5 |
| `inna_czynnosc` | 32 | 37 | 26 | 34 | 11 | 43 | 33 |
| `nierozpoznane` | 22 | 15 | 1 | 10 | 2 | 8 | 4 |
| `postoj` | 3 | 12 | 25 | 19 | 7 | 1 | 11 |
| `sciaganie_elementu` | 14 | 16 | 14 | 20 | 72 | 12 | 34 |
| `spawanie` | 61 | 21 | 17 | 95 | 4 | 1075 | 14 |
| `ukladanie_pretow` | 44 | 34 | 22 | 37 | 2 | 31 | 911 |

### Boundary timing error

377 real activity changes, 595 predicted. Median error **0.0 s**, p90 6.0 s, max 32.0 s; 87.8% land within 2 s. Spurious boundaries (no real change within 4 s): **193**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.125`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 260 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 3.7% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.3% | 84.9% | 1.09× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.6% | 90.3% | 0.91× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 65.9% | 56.9% | 1.16× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 24.5% | 24.9% | 0.99× | 0.5 pp | ❌ |
| `postoj` | 78 | 24.4% | 26.8% | 0.91× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 0.0% | 0.0% | 0.03× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.81× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 94.9% (n=273) | 78.9% (n=247) | 93.5% (n=276) | 94.8% (n=267) |
| `ukladanie_pretow` | 94.4% (n=180) | 93.3% (n=179) | 31.1% (n=193) | 97.0% (n=264) | 90.6% (n=265) |
| `sciaganie_elementu` | 81.8% (n=22) | 70.0% (n=30) | 37.9% (n=58) | 79.4% (n=34) | 84.2% (n=38) |
| `inna_czynnosc` | 7.7% (n=65) | 52.4% (n=42) | 14.3% (n=70) | 45.0% (n=20) | 36.8% (n=19) |
| `postoj` | 11.9% (n=42) | 37.5% (n=16) | 0.0% (n=4) | 0.0% (n=5) | 72.7% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 0.0% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 0 | 7 | 67 | 2 | 0 | 10 | 6 |
| `inna_czynnosc` | 0 | 53 | 17 | 4 | 16 | 89 | 37 |
| `nierozpoznane` | 1 | 30 | 0 | 26 | 1 | 4 | 0 |
| `postoj` | 2 | 14 | 25 | 19 | 9 | 1 | 8 |
| `sciaganie_elementu` | 0 | 18 | 2 | 1 | 120 | 27 | 14 |
| `spawanie` | 0 | 45 | 1 | 6 | 16 | 1188 | 31 |
| `ukladanie_pretow` | 0 | 46 | 0 | 13 | 49 | 80 | 893 |

### Boundary timing error

377 real activity changes, 480 predicted. Median error **0.0 s**, p90 40.0 s, max 74.0 s; 79.4% land within 2 s. Spurious boundaries (no real change within 4 s): **130**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.18`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 255 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.0% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.9% | 84.1% | 1.10× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 78.7% | 92.4% | 0.85× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 46.2% | 81.6% | 0.57× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 39.4% | 22.4% | 1.76× | 0.5 pp | ❌ |
| `postoj` | 78 | 19.2% | 28.8% | 0.67× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 1.1% | 100.0% | 0.01× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.92× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 91.6% (n=273) | 92.3% (n=247) | 88.8% (n=276) | 93.6% (n=267) |
| `ukladanie_pretow` | 95.6% (n=180) | 94.4% (n=179) | 17.1% (n=193) | 98.9% (n=264) | 81.5% (n=265) |
| `sciaganie_elementu` | 81.8% (n=22) | 43.3% (n=30) | 0.0% (n=58) | 73.5% (n=34) | 73.7% (n=38) |
| `inna_czynnosc` | 16.9% (n=65) | 85.7% (n=42) | 20.0% (n=70) | 55.0% (n=20) | 68.4% (n=19) |
| `postoj` | 9.5% (n=42) | 12.5% (n=16) | 0.0% (n=4) | 60.0% (n=5) | 54.5% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 4.2% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 1 | 13 | 66 | 0 | 0 | 10 | 2 |
| `inna_czynnosc` | 0 | 85 | 18 | 4 | 5 | 87 | 17 |
| `nierozpoznane` | 0 | 51 | 0 | 4 | 2 | 2 | 3 |
| `postoj` | 0 | 20 | 29 | 15 | 5 | 1 | 8 |
| `sciaganie_elementu` | 0 | 50 | 2 | 2 | 84 | 31 | 13 |
| `spawanie` | 0 | 60 | 0 | 0 | 4 | 1196 | 27 |
| `ukladanie_pretow` | 0 | 101 | 4 | 27 | 3 | 95 | 851 |

### Boundary timing error

377 real activity changes, 507 predicted. Median error **0.0 s**, p90 24.0 s, max 46.0 s; 80.4% land within 2 s. Spurious boundaries (no real change within 4 s): **136**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.25`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 254 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.9% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 92.5% | 88.5% | 1.05× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 80.9% | 92.7% | 0.87× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 58.8% | 75.9% | 0.77× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 37.0% | 23.7% | 1.56× | 0.5 pp | ❌ |
| `postoj` | 78 | 29.5% | 33.8% | 0.87× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 4.3% | 30.8% | 0.14× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 1.6% | 0.7% | 2.39× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 98.2% (n=224) | 93.0% (n=273) | 85.4% (n=247) | 90.9% (n=276) | 95.5% (n=267) |
| `ukladanie_pretow` | 93.3% (n=180) | 93.9% (n=179) | 17.1% (n=193) | 98.1% (n=264) | 93.2% (n=265) |
| `sciaganie_elementu` | 77.3% (n=22) | 76.7% (n=30) | 22.4% (n=58) | 64.7% (n=34) | 84.2% (n=38) |
| `inna_czynnosc` | 16.9% (n=65) | 57.1% (n=42) | 32.9% (n=70) | 45.0% (n=20) | 68.4% (n=19) |
| `postoj` | 19.0% (n=42) | 37.5% (n=16) | 0.0% (n=4) | 60.0% (n=5) | 54.5% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 16.7% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 4 | 11 | 69 | 4 | 0 | 3 | 1 |
| `inna_czynnosc` | 1 | 80 | 29 | 6 | 12 | 67 | 21 |
| `nierozpoznane` | 1 | 31 | 1 | 26 | 0 | 0 | 3 |
| `postoj` | 0 | 16 | 22 | 23 | 8 | 2 | 7 |
| `sciaganie_elementu` | 0 | 34 | 1 | 3 | 107 | 26 | 11 |
| `spawanie` | 0 | 58 | 9 | 0 | 3 | 1191 | 26 |
| `ukladanie_pretow` | 7 | 108 | 17 | 6 | 11 | 57 | 875 |

### Boundary timing error

377 real activity changes, 520 predicted. Median error **0.0 s**, p90 8.0 s, max 56.0 s; 86.4% land within 2 s. Spurious boundaries (no real change within 4 s): **127**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.35`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 270 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.6% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 93.9% | 87.3% | 1.08× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 82.5% | 89.4% | 0.92× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 50.5% | 71.3% | 0.71× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 35.2% | 28.9% | 1.22× | 0.5 pp | ❌ |
| `postoj` | 78 | 21.8% | 30.4% | 0.72× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 13.0% | 38.7% | 0.34× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 3.2% | 1.5% | 2.21× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (n=224) | 97.8% (n=273) | 95.1% (n=247) | 83.7% (n=276) | 94.8% (n=267) |
| `ukladanie_pretow` | 96.1% (n=180) | 94.4% (n=179) | 23.3% (n=193) | 99.2% (n=264) | 91.7% (n=265) |
| `sciaganie_elementu` | 81.8% (n=22) | 50.0% (n=30) | 31.0% (n=58) | 55.9% (n=34) | 57.9% (n=38) |
| `inna_czynnosc` | 9.2% (n=65) | 64.3% (n=42) | 32.9% (n=70) | 30.0% (n=20) | 73.7% (n=19) |
| `postoj` | 19.0% (n=42) | 25.0% (n=16) | 0.0% (n=4) | 20.0% (n=5) | 36.4% (n=11) |
| `brak_na_stanowisku` | 1.5% (n=66) | 0.0% (n=2) | 45.8% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 12 | 2 | 70 | 6 | 0 | 1 | 1 |
| `inna_czynnosc` | 0 | 76 | 17 | 7 | 12 | 72 | 32 |
| `nierozpoznane` | 1 | 37 | 2 | 15 | 1 | 1 | 5 |
| `postoj` | 0 | 13 | 22 | 17 | 7 | 4 | 15 |
| `sciaganie_elementu` | 7 | 39 | 8 | 5 | 92 | 19 | 12 |
| `spawanie` | 2 | 32 | 0 | 0 | 4 | 1208 | 41 |
| `ukladanie_pretow` | 9 | 64 | 18 | 6 | 13 | 79 | 892 |

### Boundary timing error

377 real activity changes, 534 predicted. Median error **0.0 s**, p90 6.0 s, max 18.0 s; 83.3% land within 2 s. Spurious boundaries (no real change within 4 s): **156**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.5`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 278 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 5.4% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 95.3% | 90.8% | 1.05× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 84.8% | 92.9% | 0.91× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 70.9% | 72.5% | 0.98× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 30.6% | 28.8% | 1.06× | 0.5 pp | ❌ |
| `postoj` | 78 | 29.5% | 31.1% | 0.95× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 2.2% | 13.3% | 0.16× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 4.8% | 1.8% | 2.63× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 96.3% (n=273) | 88.3% (n=247) | 96.7% (n=276) | 95.9% (n=267) |
| `ukladanie_pretow` | 94.4% (n=180) | 98.3% (n=179) | 42.0% (n=193) | 97.7% (n=264) | 87.5% (n=265) |
| `sciaganie_elementu` | 90.9% (n=22) | 73.3% (n=30) | 44.8% (n=58) | 85.3% (n=34) | 84.2% (n=38) |
| `inna_czynnosc` | 7.7% (n=65) | 31.0% (n=42) | 40.0% (n=70) | 35.0% (n=20) | 68.4% (n=19) |
| `postoj` | 9.5% (n=42) | 37.5% (n=16) | 0.0% (n=4) | 60.0% (n=5) | 90.9% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 8.3% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 2 | 3 | 84 | 3 | 0 | 0 | 0 |
| `inna_czynnosc` | 0 | 66 | 19 | 10 | 14 | 70 | 37 |
| `nierozpoznane` | 11 | 34 | 3 | 9 | 2 | 0 | 3 |
| `postoj` | 2 | 11 | 28 | 23 | 4 | 1 | 9 |
| `sciaganie_elementu` | 0 | 24 | 9 | 1 | 129 | 8 | 11 |
| `spawanie` | 0 | 41 | 1 | 3 | 5 | 1227 | 10 |
| `ukladanie_pretow` | 0 | 50 | 19 | 25 | 24 | 46 | 917 |

### Boundary timing error

377 real activity changes, 495 predicted. Median error **0.0 s**, p90 6.0 s, max 28.0 s; 90.1% land within 2 s. Spurious boundaries (no real change within 4 s): **126**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d0.75`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1344 MiB peak on one card

**Cost: 277 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 4.2% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 94.3% | 91.8% | 1.03× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 83.6% | 87.9% | 0.95× | 0.1 pp | ❌ |
| `sciaganie_elementu` | 182 | 68.7% | 52.7% | 1.30× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 33.3% | 35.5% | 0.94× | 0.5 pp | ❌ |
| `postoj` | 78 | 25.6% | 24.7% | 1.04× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 1.1% | 50.0% | 0.02× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 6.5% | 3.2% | 2.02× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.1% (n=224) | 92.7% (n=273) | 95.1% (n=247) | 92.4% (n=276) | 93.3% (n=267) |
| `ukladanie_pretow` | 94.4% (n=180) | 100.0% (n=179) | 27.5% (n=193) | 98.5% (n=264) | 91.3% (n=265) |
| `sciaganie_elementu` | 77.3% (n=22) | 56.7% (n=30) | 60.3% (n=58) | 70.6% (n=34) | 84.2% (n=38) |
| `inna_czynnosc` | 15.4% (n=65) | 31.0% (n=42) | 42.9% (n=70) | 35.0% (n=20) | 63.2% (n=19) |
| `postoj` | 14.3% (n=42) | 31.2% (n=16) | 0.0% (n=4) | 0.0% (n=5) | 81.8% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 4.2% (n=24) | n/a | n/a |

**Fails the bar on:** `ukladanie_pretow`, `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 1 | 1 | 75 | 11 | 0 | 0 | 4 |
| `inna_czynnosc` | 0 | 72 | 16 | 9 | 20 | 58 | 41 |
| `nierozpoznane` | 0 | 29 | 4 | 15 | 4 | 0 | 10 |
| `postoj` | 0 | 7 | 18 | 20 | 11 | 2 | 20 |
| `sciaganie_elementu` | 0 | 20 | 5 | 5 | 125 | 11 | 16 |
| `spawanie` | 0 | 34 | 0 | 0 | 6 | 1214 | 33 |
| `ukladanie_pretow` | 1 | 40 | 7 | 21 | 71 | 37 | 904 |

### Boundary timing error

377 real activity changes, 485 predicted. Median error **2.0 s**, p90 8.0 s, max 28.0 s; 83.3% land within 2 s. Spurious boundaries (no real change within 4 s): **103**.

The annotation's own boundaries are only accurate to ±1 s (2 s stride, boundary at the sample midpoint), so error below 1 s is not resolvable by this fixture and should not be read as precision.

## Arm: `tcn-pixel-518-d1`

*#124 - resolution sweep for the zone-annotator floor*

**Hardware verdict: OK** — 1342 MiB peak on one card

**Cost: 236 GPU-seconds per video-hour**, measured on **cctv-vps**.

Abstention (`nierozpoznane` predicted): 3.7% of samples.

### Per-activity accuracy (held-out union)

| Activity | Support | Recall (the bar) | Precision | Time reported | 1 error = | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `spawanie` | 1287 | 90.5% | 93.1% | 0.97× | 0.1 pp | ✅ |
| `ukladanie_pretow` | 1081 | 86.4% | 89.6% | 0.96× | 0.1 pp | ✅ |
| `sciaganie_elementu` | 182 | 56.0% | 77.9% | 0.72× | 0.5 pp | ❌ |
| `inna_czynnosc` | 216 | 39.8% | 24.9% | 1.60× | 0.5 pp | ❌ |
| `postoj` | 78 | 26.9% | 24.7% | 1.09× | 1.3 pp | ❌ |
| `brak_na_stanowisku` | 92 | 13.0% | 37.5% | 0.35× | 1.1 pp | ❌ |
| `nierozpoznane` | 62 | 0.0% | 0.0% | 1.79× | 1.6 pp | — |

*Time reported* is predicted seconds over true seconds for the activity — the number a chronometraż client feels. Above 1.25× a passing recall is marked **gamed**: the class was bought by over-calling it, and a work-study that over-reports productive time is worse than one that under-reports it.

### Per held-out window

The union above is one number over folds that held out different material. Where those folds disagree, the mean describes neither.

| Activity | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| `spawanie` | 99.6% (n=224) | 95.2% (n=273) | 69.6% (n=247) | 91.3% (n=276) | 96.6% (n=267) |
| `ukladanie_pretow` | 94.4% (n=180) | 98.3% (n=179) | 40.9% (n=193) | 98.9% (n=264) | 93.6% (n=265) |
| `sciaganie_elementu` | 86.4% (n=22) | 76.7% (n=30) | 13.8% (n=58) | 67.6% (n=34) | 76.3% (n=38) |
| `inna_czynnosc` | 7.7% (n=65) | 33.3% (n=42) | 71.4% (n=70) | 20.0% (n=20) | 68.4% (n=19) |
| `postoj` | 19.0% (n=42) | 31.2% (n=16) | 0.0% (n=4) | 0.0% (n=5) | 72.7% (n=11) |
| `brak_na_stanowisku` | 0.0% (n=66) | 0.0% (n=2) | 50.0% (n=24) | n/a | n/a |

**Fails the bar on:** `sciaganie_elementu`, `inna_czynnosc`, `postoj`, `brak_na_stanowisku`. An 84% class fails even if the average clears.

### Confusion matrix

| truth ↓ / pred → | `brak_na_stanowisku` | `inna_czynnosc` | `nierozpoznane` | `postoj` | `sciaganie_elementu` | `spawanie` | `ukladanie_pretow` |
|---|---|---|---|---|---|---|---|
| `brak_na_stanowisku` | 12 | 4 | 67 | 8 | 0 | 0 | 1 |
| `inna_czynnosc` | 6 | 86 | 17 | 8 | 14 | 51 | 34 |
| `nierozpoznane` | 10 | 30 | 0 | 13 | 1 | 0 | 8 |
| `postoj` | 0 | 10 | 24 | 21 | 6 | 1 | 16 |
| `sciaganie_elementu` | 0 | 37 | 2 | 14 | 102 | 8 | 19 |
| `spawanie` | 0 | 77 | 0 | 11 | 4 | 1165 | 30 |
| `ukladanie_pretow` | 4 | 101 | 1 | 10 | 4 | 27 | 934 |

### Boundary timing error

377 real activity changes, 500 predicted. Median error **0.0 s**, p90 6.0 s, max 32.0 s; 81.6% land within 2 s. Spurious boundaries (no real change within 4 s): **139**.

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
