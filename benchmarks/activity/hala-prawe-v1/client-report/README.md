# Client report for `hala-prawe-v1` (C.0)

The Polish client-facing report published at
`gpu-exchange` → `packages/data-ops/src/scripts/seed-report-wykonalnosc.html`.
The **template lives here**, next to the measurements it quotes, and the built
HTML is copied into `gpu-exchange` for seeding.

## Why the template is here and not there

The 2026-09-01 version of this report told a client that
`brak_na_stanowisku` reached **98.5%**. No arm ever scored that. It was the pose
detector's own hit rate on W1's empty-bench frames, in-sample, lifted from a
docstring and typed into a column of held-out classifier accuracies.

Nothing caught it, because the HTML was hand-written in a repo that has no
access to the numbers. So the body is now a template with `@@PLACEHOLDER@@`
slots and `fill_report.py` substitutes them from committed measurement JSON. An
unfilled slot is a hard error, and a figure with no measurement behind it has
nowhere to come from.

## Build

```bash
uv run benchmarks/activity/tools/evaluate_arms.py \
  --manifest benchmarks/activity/hala-prawe-v1/manifest.source.json \
  --predictions benchmarks/activity/hala-prawe-v1/predictions/*.json \
  --out  benchmarks/activity/hala-prawe-v1/C0-report.md \
  --json-out benchmarks/activity/hala-prawe-v1/C0-report.json

cd benchmarks/activity/hala-prawe-v1/client-report
cat report.head.html report.body.html > /tmp/template.html
uv run fill_report.py ../C0-report.json ../empty-station-rule.json \
  /tmp/template.html /tmp/seed-report-wykonalnosc.html
```

Then copy `/tmp/seed-report-wykonalnosc.html` over the file of the same name in
`gpu-exchange/packages/data-ops/src/scripts/` and run that repo's
`db:seed-report-wykonalnosc:<env>`.

## Two things the filler enforces

- **An unfilled `@@SLOT@@` fails the build.** A number cannot be quietly dropped.
- **An em dash fails the build.** The client asked for none, and a rewrite pass
  reintroduces them by habit.

## Two groups, never their average

`Z` (*znana*) is the union of the two held-out morning windows, whose folds
trained on evening material too. `N` (*nowa*) is the evening window, held out by
the fold that trained on mornings alone.

They are reported separately because on `spawanie` they are 94.2% and 27.5%.
The union over all three folds is 72%, which describes neither and hides the
finding the report is built around. The union stays in `C0-report.json` as the
split's own reporting rule requires; it does not reach the client body.
