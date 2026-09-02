# Discovery brief — annotation module (`#119`)

**Hand this file to `/ask` in a fresh session.** It is written for an interviewer
who has none of the context, so it front-loads what is already settled and points
at where the real decisions are.

Suggested invocation:

```
/ask Read docs/discovery/annotation-module.md and interview me on it.
```

---

## The idea, in the requester's words

> I want to make a separate module / app / site out of this. I upload a video
> file (MP4), I define what labels I want to apply to individual frames (and set
> the sampling frequency in seconds), I mark a zone — the crop of the whole frame
> I care about while labelling — I label, and I get a result ready to hand to a
> model.

Tracked as `#119` in this repo.

## Where this comes from

There is already a working single-purpose version:
`benchmarks/activity/tools/build_interval_annotation.py`. It has been used for
real on three 20-minute windows of a client's CCTV and produced **1799
hand-labelled samples** that gated a live feasibility study (`#117`).

So this is not a greenfield idea. It is "generalise a thing that works, without
losing the parts that took a week of being wrong to learn."

**Read before interviewing:**

- `benchmarks/activity/tools/build_interval_annotation.py` — the existing tool
- `benchmarks/activity/hala-prawe-v1/METHODOLOGY.md` — how the ground truth was
  made and every trap found
- `benchmarks/activity/README.md` — the two annotation shapes and why they differ
- The `#119` issue body — the long-form version of this brief

---

## What is already settled — do NOT spend questions here

These are measured, not preferred. Treat them as constraints and move on.

1. **Sampling is by presentation timestamp, never frame number.** One clip's
   container declared `120/1` while the true rate was ~20 fps; frame-number
   indexing silently mis-mapped every timestamp in it.
2. **The zone crop comes from the native frame, never a display copy.** Native
   station crop gets 32.4% detection recall; the same content at 640 px
   full-frame gets 0.0%.
3. **A suggestion is never a label.** Every label starts `null`. Pre-seeded hints
   must be confirmed by a keystroke or they do not enter the export.
4. **Interval boundaries land at the midpoint** between two differently-labelled
   samples, so timing error is at most `stride/2`.
5. **The zone is the unit of measurement, not a person.** No tracking, no
   re-identification, no faces. This is what makes hand annotation tractable at
   all.
6. **The tool produces ground truth and stops.** No model registry, no training
   loop, no experiment tracker.

## The result that should shape the whole interview

A model trained on two 20-minute windows from one morning scored **94.2%** on
its target activity. On a third window — same camera, same station, evening
shift, different operator — the same model scored **27.5%**.

The footage was fine: a model that trains on nothing scored 97.6% on it.

**So a tool that ingests one video and hands back "your dataset" produces
confident, wrong models.** Whatever comes out of this interview has to take that
seriously. It is the single most important input to the design.

A second lesson from the same fixture: that third window differs in *both* shift
and operator, so we cannot attribute the drop to either. Conditions that move
together cannot be separated afterwards.

## The other hard-won lessons, compressed

Use these to pressure-test. Each one is a real failure from a real project.

| Lesson | Evidence |
|---|---|
| An "I cannot tell" label must exist and be mandatory | A dataset with zero of them on hard footage is one where somebody guessed. Ours ran 3.4%. |
| Rare labels and under-sampled labels look identical and need opposite remedies | One activity got 62 samples in an hour. Not a labelling shortfall — it barely happens at that station. |
| The train/test split must be frozen before any model runs | We widened ours after three models had been measured. The manifest now carries `declared_before_any_model_run: false` and an integrity warning. |
| Partial coverage flatters every downstream number | An unanswered sample that quietly leaves the denominator makes a model look better than it is. |
| Recall alone is gameable | A brightness threshold hit 99.4% recall on welding while reporting 2.18× the real welding time. |
| A number with no measurement behind it will reach a client | We published 98.5% for one class. No model ever scored it — it was a detector statistic from a docstring, typed into a table by hand. |
| Concatenated recorder chunks produce silent decode gaps | A clip that yields 540 samples instead of 600 shifts every label after the gap. |
| Source material disappears | The client's originals were deleted by a retention reaper; the preserved copies did not resolve. Three windows exist on one unbacked-up machine. |

---

## Open decisions — this is where the interview belongs

Roughly in dependency order. The interviewer should not assume any of these.

### 1. Who actually uses this, and where does it run

The existing tool is a **static HTML file with `localStorage`** and zero
infrastructure. It works, it survives a browser refresh, and it has no server,
no auth, no upload path and no personal-data surface.

"Upload an MP4 to an app" implies a server, storage, authentication and a
personal-data position. **Push back before accepting it.** Is the requirement
actually "someone other than me labels footage on a machine that is not mine", or
is it "the current thing should be less annoying to set up"? Those have very
different costs.

### 2. Where it lives

Own repo, module in `cctv-gpu-engine`, or a page in the `gpu-exchange` platform.
Depends on whether it is ever meant to stand alone as a product. Prior art and
every lesson above live in this repo.

### 3. Frames or clips — this is a one-way door

The export shape decides which model families are reachable:

- **Per-sample labels + cropped stills** → pose-sequence models (the winner
  here), temporal action segmentation (MS-TCN, ASFormer), MLP heads, frame
  classifiers.
- **Short clip windows around each sample** → also video models (VideoMAE, X3D,
  TimeSformer, VideoSwin).

Cheaper to design for clips now than to retrofit. But it multiplies storage.
Worth asking whether video models are ever plausibly in scope.

### 4. How opinionated should the tool be

Does it **block** on bad practice or merely warn? Concretely: refuse to export
with unlabelled gaps, refuse a vocabulary with no "cannot tell", refuse to call a
project ready when every clip shares the same conditions, refuse to re-split
after a model has run. Every one of those is a place where a human under time
pressure will want the override — and where we already know what the override
costs.

### 5. Multi-clip project semantics

How are per-clip conditions captured — structured fields (shift, operator,
lighting, camera, date) or free text? Structured is what allows "these two
variables moved together, you cannot separate them" as a warning. Free text is
what people actually fill in.

### 6. Storage and retention

Local filesystem, or object storage alongside the clips? The originals on this
project vanished. Whatever is chosen should be backed up, and the tool should
record hash, size, real time window and source per clip.

### 7. Second annotator and agreement

There is currently no inter-annotator agreement number for the existing fixture,
so nobody can say how much of a model's error is the model and how much is the
ceiling of the labels. Cheapest quality signal available. v1 or later?

### 8. Consent and personal data

This ingests CCTV of identifiable people. This repo defers RODO deliberately,
which is fine for an internal benchmark and not fine for a tool other people
upload footage into. Needs a stated position before external exposure, even if
the position is "internal only, no external upload".

### 9. Scale

How many clips per project, how many hours total, how long a single labelling
session, how many projects. The current tool handles 600 samples per window
comfortably; nobody has tested it at 10 000.

### 10. Does it need to run any model itself

Purely labels, or does it also run a pose pass / offer model-assisted
pre-labelling? Note lesson 3 above binds either way: assisted labels must be
flagged as such in the export, or you end up evaluating a model against its own
output.

---

## What a good outcome looks like

The session should end with the `/ask` standard close: decisions restated, open
questions named, and a signed-off assumption list ready for `/blueprint`.

Specifically, these should be settled by the end:

- [ ] Deployment shape (static tool / local app / hosted) and who uses it
- [ ] Repo location
- [ ] Export contract: frames or clips, and what the schema carries beyond labels
- [ ] Blocking vs warning on each integrity rule
- [ ] Multi-clip project model and how conditions are recorded
- [ ] Storage, retention and backup
- [ ] Whether inter-annotator agreement is in v1
- [ ] Personal-data position
- [ ] Rough scale targets

## One thing to check the design against at the end

The export must be consumable by an honest scorer, not just a trainer. That
means it carries, per label: support, the convention for interval boundaries, and
enough to compute the ratio between predicted and true duration. A brightness
threshold in this project passed a recall-only bar at 99.4% while over-reporting
the measured time by 2.18×. "Ready for a model" is a weaker claim than "ready to
be scored honestly", and the second one is what the tool should promise.
