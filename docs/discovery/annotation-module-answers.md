# Discovery answers, annotation module (`#119`)

**Answered 2026-09-10 by the requester**, against the 22-question set derived from
[`annotation-module.md`](./annotation-module.md). This file is the input to
`/blueprint`. Where an answer settles a decision, the design consequence is stated
next to it. Where two answers conflicted, the resolution taken on 2026-09-10 is
recorded with it.

## Raw answers

| Q | Answer | Decision it settles |
|---|---|---|
| Q1 | 2. Several people on our team, each on their own machine | More than one annotator, so a project has to be portable between machines |
| Q2 | 4. Don't care, pick for us | Ours to choose (see "Deployment shape" below) |
| Q3 | 2. Internal, but showable to a client as part of the offer | Module in this repo. Presentable UI, no multi-tenancy, no external login |
| Q4 | 3. A different domain entirely, different vocabulary | Nothing may assume welding. Vocabulary, zones, suggestions all project-scoped |
| Q5 | 3. Three to six clips, across days and shifts | Project is a set of clips. Coverage checks have material to work with |
| Q6 | 1, 2, 3. Shift, operator, lighting | Date and camera are NOT hand-entered, so derive them automatically |
| Q7 | 1. Yes, specific footage can be ordered within days | Coverage gaps become actionable requests, not just warnings |
| Q8 | 4. Don't know, propose something | Ours to propose (see "Storage" below) |
| Q9 | 3. Saved, reusable vocabularies | Vocabulary is a first-class object with its own identity and version |
| Q10 | 2. Shortest countable activity is about five seconds | Sampling stride 2 s, not 5 s (see "Stride" below) |
| Q11 | 2. Two to four zones per clip | Multi-zone is v1, not later. Independent timeline per zone |
| Q12 | 1. Cameras can be repositioned or zoomed onto the station | The resolvability check produces a framing recommendation, not just a flag |
| Q13 | 1. "Cannot tell" excluded from the split and reported openly | Export always carries the unrecognised rate |
| Q14 | 1. Tool says "too few occurrences, do not report separately" | Per-label support gate with a stated threshold |
| Q15 | a-e all **O** (warn), nothing blocks | See conflict 1 |
| Q16 | 2. Only a named person may lift a block, with a trace | See conflict 1 |
| Q17 | 1. Second annotator needed in v1 | Agreement figure is v1 scope (see conflict 3) |
| Q18 | 1. Monitoring policy or consent covers it | Verify the wording covers model training, not just recording |
| Q19 | 3. Client hardware only, nothing stays with us | See conflict 2 |
| Q20 | 4. More than twenty stations | See "Scale" below |
| Q21 | 1. Blocks a paid deployment, **date not given** | Still open |
| Q22 | 1. Model-assisted suggestions from v1 | See conflict 4 |

## What the answers decide

### Deployment shape (Q1, Q2, Q3, Q19)

**A local application, not a website and not a server.** Q2 left the choice to us
and every other answer narrows it to the same place: several annotators on their
own machines (Q1), footage that may not rest on our infrastructure (Q19), and no
external users (Q3). A hosted upload path would buy accounts, a storage tier and a
personal-data surface that no answer asks for.

A project is therefore a directory: clips, zone crops, labels, manifest. Portable
between machines, openable by whoever holds it.

### Storage (Q8, Q19)

Proposal, since Q8 asked for one: the project directory lives on client-side
storage, and where the client already runs the appliance, on its rolling buffer.
That path is already proven here, W4 and W5 of the current fixture were cut from
exactly such a buffer. The module reads clips from it and writes the project back
beside them.

This satisfies Q19 as written, and it also creates conflict 2 below.

### Stride (Q10)

The requester named a five-second floor for the shortest countable activity. A
five-second stride would sample it at best once and at worst not at all, so the
stride has to sit at half that floor: **2 s**, which is also the stride of every
existing export, so the round-trip acceptance criterion in `#119` stays trivial.
Boundary error lands at one second.

### Scale (Q5, Q10, Q11, Q17, Q20)

The answers multiply:

| Factor | Value |
|---|---|
| Stride | 2 s, so 1800 samples per zone per hour of footage |
| Zones per clip (Q11) | 2 to 4 |
| Clips per station (Q5) | 3 to 6 |
| Stations (Q20) | more than 20 |
| Passes (Q17) | 2 |

Twenty stations at three zones per camera view is about seven views, three to six
clips each, so 21 to 42 clips. At twenty minutes per clip that is 600 samples per
zone, 1800 per clip, 3600 after the second pass, and **75 000 to 150 000
confirmations** across the project set.

Everything hand-labelled on this project to date is **1799 samples**. The answers
describe forty to eighty times that. Three design consequences follow, and they are
not optional:

1. Q22's model-assisted suggestions stop being a convenience and become the only
   way the number is reachable.
2. Q17's second pass needs to be a sampled second pass, not a full one (conflict 3).
3. Nothing in the tool may hold a project in memory or recompute per-label support
   from scratch on every keystroke.

### Condition fields (Q6, Q7)

Hand-entered: shift, operator, lighting. Derived automatically: date from the
container, camera from the clip source. Q7 turns the collinearity warning into a
request the requester can actually fulfil, so the warning should name the missing
combination ("no evening shift with operator A"), not just report that two
variables moved together.

Worth watching: shift and lighting will move together on almost every site, so that
particular pair will fire constantly and needs to be phrased as information rather
than as an error.

### Framing (Q12)

Cameras can be moved, so the resolvability check earns a second output: a framing
recommendation. Given the 2026-09-03 incident, where a production run reported zero
seconds of welding because the work drifted outside the part of the rectangle the
head actually reads, the check must be expressed in terms of the area the model
consumes, not the rectangle the annotator drew.

## Four conflicts, all settled 2026-09-10

### 1. Nothing blocks (Q15) but somebody may lift a block (Q16)

Q15 marked all five integrity rules as warnings, which leaves nothing for Q16's
named person to lift. One of the two answers is not what the requester meant.

**Settled:** a, b, c and e warn. **d blocks.** Changing the train/test split once a
model result is known is the only one of the five that invalidates numbers already
shown to someone, and it is the one this project got wrong: the current fixture's
manifest still carries `declared_before_any_model_run: false`.

Q16 applies to exactly that rule. The block is liftable by one named person, and the
lift is written into the export as an amendment carrying who lifted it and why, so a
split that moved after the fact is visible to whoever reads the numbers later.

### 2. Nothing stays with us (Q19) versus retraining

Q19 as written means the labelled crops are deleted when a project closes. The
current head exists because that did not happen: retraining on one current-layout
window moved `spawanie` recall from 56.2% to 92.1%, and reported welding time from
56% of the truth to 94%. A head whose training material has been deleted cannot be
retrained when the bench is rearranged, and rearrangement is precisely what triggers
the need.

**Settled: tiered retention, keyed to purpose rather than to a calendar.**

- **Source clips never leave the client.** The project references each one by hash,
  byte size, real time window and path, and never copies it into the project
  directory. The module reads clips in place, from the appliance buffer where one
  exists. Q19 is satisfied for the thing Q19 is actually about.
- **Only zone crops and their labels travel.** That is the entire training set, and
  nothing else is needed to fit or refit a head. The crop is the authored rectangle,
  and the manifest records what fraction of the frame it represents, so the client can
  see how little left the site rather than take it on trust.
- **The operator field is a project-scoped pseudonym, never a name.** Q6 already
  accepted "operator A", so the tool enforces that shape and the condition metadata is
  pseudonymous by construction.
- **Retention is bound to the head, not to a date.** The export carries
  `retention: { basis: "model-lifetime", bound_to: <head version> }`. When that head is
  withdrawn or superseded, everything bound to it is deleted and a receipt is written:
  what went, how many samples, when.
- **That chain already matches how this repo treats reports.** A report names the head
  that measured it, the panel withholds totals when the head no longer matches the
  catalogue, and a completed task's input is reaped immediately. A withdrawn head has
  no reports left to defend, so its training crops have nothing left to serve. The
  deletion rule falls out of the existing design instead of being bolted onto it.
- **The manifest is the audit surface.** It lists everything that left the premises,
  with hashes and counts, so the carve-out is checkable rather than promised.

The fallback if even this is refused is fitting the head on the client's own hardware.
The appliance is a mini-PC with no GPU, so that is a different project, not a setting.

### 3. Two full passes (Q17) at the scale of Q20

A second pass over everything doubles a number that is already forty to eighty times
anything attempted here.

**Settled: a stratified, blind agreement set, not a second pass over everything.**

- **What gets double-labelled.** Per project: a per-label floor of up to 50 samples of
  every label in the vocabulary, spread over at least three clips, plus a 2% random
  slice of each clip. On a six-clip project that is roughly 500 to 600 samples against
  the 10 800 a full second pass costs, about 5%.
- **Why a floor per label rather than a flat percentage.** A flat 5% of a label that
  occurs 62 times yields three samples and no usable number, and the rare labels are
  exactly the ones where the agreement figure matters.
- **The second pass is blind.** The second annotator never sees the first label, and
  the samples arrive shuffled and mixed into ordinary work so they cannot tell which
  ones are being compared. A visible first label anchors the second and inflates the
  result into meaninglessness.
- **What comes out.** Agreement per label and overall, chance-corrected as well as
  raw, because one dominant label makes raw agreement look excellent for free. Plus
  the confusion pairs: two labels a human cannot separate is a defect in the
  vocabulary, not in the annotator, and the remedy is to merge or redefine them.
- **What the number does.** A label whose agreement sits under the bar carries a
  caveat through to the export, the same way `reliable: false` already propagates in
  the station pipeline. A quality signal that changes nothing downstream is decoration.
- **One annotator twice** is the fallback where only one person is available. Same
  mechanism, recorded as self-consistency rather than agreement, because it measures
  whether one person is stable, not whether the vocabulary is clear.
- Full double-labelling stays available for a project that wants it.

### 4. Assisted labelling (Q22) in a domain with no model (Q4)

Q4 says the next real job is a different domain with a different vocabulary. There
is no model for it, so there is nothing to pre-seed from on the first clip.

**Settled: suggestions bootstrap inside the project, and they are gated per label.**

- **Cold.** No suggestions at all. The annotator labels a seed block by hand. Nothing
  is pre-seeded from another project, because Q4 says both the domain and the
  vocabulary are new.
- **Warm.** A small head fits on that project's own crops, frozen backbone plus about
  a megabyte of head, which is the architecture this repo already runs. It predicts
  the samples nobody has reached yet.
- **Gated, and this is the part that carries the design.** A suggestion appears only
  for labels where the freshly fitted head clears the bar on samples it did not train
  on, inside the same project. Labels under the bar show nothing at all. This project
  has already produced a head at 91.5% on one activity and useless on another, and a
  wrong suggestion is worse than no suggestion: correcting it costs more than deciding
  fresh, and it pulls the annotator toward the model's own mistake.
- **Refit on a cadence** as labels accumulate, with the gate re-evaluated each time,
  so labels enter and leave the suggestion set as the evidence changes.
- **The confirmation rule does not move.** Every label starts null, a suggestion is a
  highlight rather than a value, and a keystroke is required. The export records per
  sample whether it was confirmed from a suggestion, which head suggested it, and what
  that head's gate score was, so any figure can be recomputed with machine-seeded
  labels excluded.
- **The agreement set from conflict 3 is always drawn without suggestions.** Otherwise
  the agreement figure measures the model's consistency instead of the annotators'.
- **Measure the assist, and watch for rubber-stamping.** The project reports the
  override rate per label. An override rate near zero on a label whose head scores 86%
  is not a good sign: it means somebody is confirming the one in seven the head gets
  wrong.

## Still open

- **Q18 is a claim, not a document.** The monitoring policy covers recording; whether
  its wording covers model training and processing by our team is worth reading before
  the module ingests a second site's footage. It does not gate the design, only the
  second site.
- **Q21 carries no date, and is not being chased.** v1 scope is decided by the
  requirements in this file, not by a delivery date.

## Next step

`/blueprint` on `#119`. Every conflict is settled, so the PRD has no open forks to
route around.
