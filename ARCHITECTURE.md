# Architecture

This document explains the design decisions, and — more usefully — the
reasoning behind them, so that changing one later is an informed act rather
than an archaeological dig.

The system is built for one operating assumption: **it will be interrupted, it
will encounter media it cannot handle, and it will run unattended against more
data than fits in memory or in one night.** Nearly every decision below follows
from that.

---

## 1. Shape of the system

```
                 ┌──────────────┐
  Incoming/ ───▶ │  Discovery   │──┐
                 └──────────────┘  │  enqueue
                                   ▼
                        ┌─────────────────────┐
                        │   jobs table        │  ◀── the only coordination point
                        │  (durable queue)    │
                        └─────────────────────┘
                                   │ claim
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
          CPU worker pool                      GPU worker pool
                 │                                   │
                 └────────────┬──────────────────────┘
                              ▼
   ingest → segment → analyze → fingerprint → dedupe → categorize
          → materialize → export
```

Each stage is a small function that does one thing and enqueues the next. The
alternative — one long per-source function — was rejected because it makes the
unit of loss the whole recording. Here, a crash costs the single stage that was
in flight.

Key modules:

| Module | Responsibility |
| --- | --- |
| `config.py` | Layered, typed settings. The only place paths and thresholds come from. |
| `constants.py` | Every tunable default, with the reasoning in comments. |
| `db/` | Models, durable queue, repositories. |
| `media/` | ffmpeg/ffprobe wrappers, disc expansion. |
| `segment/` | Signal collection, boundary fusion, segmentation policy. |
| `analyze/` | The analyzer contract, registry, built-in suite, and the caching runner. |
| `dedupe/` | Fingerprinting and duplicate resolution. |
| `categorize/` | Vote aggregation and the classification decision. |
| `library/` | On-disk layout and atomic materialization. |
| `export/` | Fieldstation export and broadcast-shaped scheduling. |
| `search/` | Query parsing, keyword index, vectors, fusion. |
| `pipeline/` | Discovery, stage handlers, workers, orchestrator, watcher, reports. |
| `sources/` | Optional archive downloaders. |
| `dashboard/` | Local FastAPI UI. |

---

## 2. The decisions that matter

### 2.1 The queue is the database

`jobs` is an ordinary table. Workers claim rows inside a transaction and hold a
**lease** with an expiry rather than a lock.

*Why a lease and not a lock:* a lock held by a process that has been killed is
indistinguishable from a lock held by a process that is working. A lease that
expires is self-healing — the next run reclaims it with no operator action. The
"stuck in RUNNING forever" state simply does not exist.

*Why the database and not Redis/RabbitMQ:* this is a single-operator system
processing a queue that is measured in thousands, not millions. Adding a broker
would add a second thing that must be running, backed up and recovered, in
exchange for throughput nobody needs. The database is already the durability
requirement.

*Why idempotent enqueue:* every job carries a `dedupe_key` derived from the
work it will do, with a unique constraint. Enqueueing the same work twice is a
no-op at the storage layer, so no caller anywhere has to remember to check
first. This is what makes rescanning `Incoming/` free.

### 2.2 Threads, not processes

Almost all wall-clock time is spent inside ffmpeg subprocesses or model
libraries that release the GIL. Threads also share one loaded Whisper model
across all workers; with processes, a 16-worker box would hold sixteen copies.
GPU work is a separate, smaller pool so a single card is never oversubscribed
while CPU workers keep going.

### 2.3 Segmentation fuses signals; it does not trust any one of them

Black-frame detection alone fails on real archived television constantly: VHS
fades through grey, a station may segue straight from promo to programme, a
re-encode may have compressed the black away.

So one ffmpeg decode pass collects seven signal types — black, silence, scene
change, freeze, aspect change, loudness jump, and corner-logo change — which
are clustered by time and scored by **distinct signal kind**. Three
scene-change hits in a row are one opinion; black + silence + logo-drop is
three independent ones and scores far higher.

*The grid bonus:* US cable in this era ran on a hard 15/30/60-second
advertising grid. A candidate boundary landing near a grid multiple, measured
from the previous accepted boundary, gets a small bonus. This is what recovers
the internal boundaries of a commercial pod where the black between spots was
too brief for `blackdetect` to catch them all.

*One decode pass* is not a micro-optimization. At 100 TB, decoding the same
six-hour capture once per detector is the difference between a night and a
week. The corner-logo branch uses `split` plus a separate `scdet` writing to
its own metadata file, so both live in the same pass.

*Adaptive weighting:* a detector that fires more than a few times a minute is
not finding programme boundaries — it is describing the content's editing
style. Its weight is scaled down in proportion to how far past a plausible rate
it went. This was forced by a real two-hour anime capture, where `scdet` fired
21×/minute and the corner-logo watcher 14×/minute, drowning the 97 black frames
and 143 silences that marked the actual transitions: 312 shot-length fragments
became 66 clips with the programme/commercial structure intact. Black and
silence are never adapted — they are physical events in the broadcast, not
reactions to what is on screen.

Everything after signal collection is pure functions over lists of floats
(`segment/fusion.py`), so segmentation quality is unit-tested without ffmpeg.

### 2.4 Analyzers are plugins; results are rows

An analyzer is a class with a `name`, a `version`, a list of backends it needs,
and a `run` method. Results are stored per `(clip, analyzer)` alongside the
version that produced them.

Three properties fall out:

- **Selective re-work.** Bump one analyzer's version and only that analyzer
  re-runs, on every clip, automatically. There is no code path that reprocesses
  a whole library.
- **Graceful degradation.** A missing backend raises `BackendUnavailableError`,
  which the runner records as `skipped` — not `failed`. The clip finishes. When
  the backend appears, `bb reanalyze --missing` finds exactly the clips that
  need it. This is why a first run on a bare machine still produces a usable
  library.
- **Shared inputs.** Frames and analysis audio are extracted once per clip by
  the runner, and only if an analyzer that is actually going to run asked for
  them. Adding a tenth vision analyzer costs inference and no extra decoding;
  a fully cached clip costs no decoding at all.

### 2.5 Analyzers vote; they do not classify

An analyzer emits `(category, weight, reason)` votes. `categorize/rules.py`
tallies them.

*Why:* if analyzers wrote a label directly, they would fight over it and the
last one to run would win. Voting is additive — a new analyzer improves
classification without knowing about any other — and it makes the result
explainable: every clip stores the votes that produced its label, and
`bb clip <id>` prints them.

Confidence is the winner's *share* of the vote **multiplied by** a saturation
term over the total evidence. These must multiply, not add: a single feeble
vote has a 100% share simply because nothing contradicted it, and adding the
terms turns that into a confident answer. (This was a real bug, caught by
`test_weak_evidence_falls_back_to_unknown`.)

Votes are persisted inside each analyzer's payload, so `bb recategorize`
re-classifies the entire library from the database in seconds without running a
single model.

*Missing evidence is not evidence.* An analyzer that did not run contributes
nothing. This sounds obvious and was violated: with Whisper absent, the speech
payload was empty, an empty payload read as "no speech detected", and every
long clip collected a music-video vote. Rules now check that a measurement
exists before reasoning about its value.

*Structural evidence:* the categorizer adds votes that only it can know — a
30-second clip inside a run of other 30-second clips is a commercial pod, and
that is often more reliable than anything visible inside any one of them.
Broadcast structure is itself evidence.

### 2.6 Nothing is deleted, ever

- Unclassifiable clips → `Library/Unknown/`, fully indexed and searchable.
- Duplicates → the best copy is canonical, the rest are marked as alternates
  and filed under `Duplicates/<canonical-id>/` rather than in the library. One
  clip has exactly one home; the library shows one copy of a spot, not four.
- Unusable files → `Rejected/` with a `.reason.txt` beside them.
- Processed originals → `Processing/completed/`, because re-cutting a source
  with better settings is a normal thing to want and impossible if the master
  was thrown away.

### 2.7 Deduplication is blocked, then layered

Comparing every clip against every other clip is O(n²) — five billion
comparisons at 100k clips. Candidates are therefore **blocked by duration**
(indexed), so a 30-second spot is only ever compared against other ~30-second
clips.

Within a candidate set, three signals are applied cheapest-first: SHA256
(exact, free), a perceptual hash sequence with a cheap first-frame pre-filter,
then Chromaprint. Perceptual frames are sampled at fixed *fractions* of each
clip, so two copies that differ by half a second of leading black still align.

The perceptual hash needs no OpenCV, NumPy or Pillow: ffmpeg is asked to decode
straight down to 9×8 grayscale raw frames — 72 bytes each — which is exactly a
difference hash's input.

### 2.8 Search is hybrid, and the query parser is rules

Three mechanisms, merged:

1. **Structured filters** from the query (category, network, year, entity,
   duration) applied as a hard gate — "every Taco Bell commercial" means
   commercials.
2. **Keyword** search over FTS5, with a LIKE fallback for SQLite builds without
   it.
3. **Vector** search over stored embeddings.

Keyword and vector results are merged with **Reciprocal Rank Fusion** rather
than a weighted score blend, because BM25 and cosine scores live on
incomparable scales; normalizing them against each other produces a number that
looks principled and behaves badly. RRF uses only rank, which does transfer.

The natural-language parser is a small rule set, not an LLM call: these queries
are highly patterned, the rules are inspectable when a query surprises you, and
searching costs nothing. Anything the rules do not understand falls through to
free text rather than being dropped.

Vector search is brute force. At 100k clips a 384-dimension scan is a few
hundred milliseconds with NumPy — fast enough for the dashboard, and free of
the operational burden an ANN index brings. `search/vectors.py` is the single
place that would change if the library outgrew it.

### 2.9 The knowledge base is data

`analyze/knowledge.py` holds the 2002–2006 cable dial: networks with their
markers, slogans, shows and eras; brands with aliases; on-camera personalities;
and format markers ("call now", "next on", "paid programming").

This is what makes "every Taco Bell commercial" work on day one with no
labelled training data and no LLM. Adding a network is a one-line edit, and
because analyzer versions drive re-analysis, bumping `KNOWLEDGE_VERSION`
re-labels the existing library automatically.

### 2.10 Atomicity everywhere media is written

Every clip that lands in the library appears completely or not at all: written
to a temporary name, then `os.replace`d into place. A half-written clip that
looks finished is the one failure mode that would silently poison the library.

Sidecar JSON is written for every clip, so the library is browsable — and in
the worst case reconstructible — from disk alone, without the database.

---

## 3. Data model

`sources` (a file) → `clips` (a span of it) → `analyses`, `clip_texts`,
`clip_tags`, `clip_entities`, `embeddings`. Plus `jobs`, `duplicate_groups`,
`downloads`, `events`, `runs`, `kv`.

Design notes:

- **Stage on the row.** Every source and clip records the last stage it
  completed, so resumption is a query rather than a bookkeeping file that can
  go stale.
- **`(source_id, seq)` is unique** on clips, so re-running segmentation updates
  rows in place instead of creating a second set — preserving the analysis
  already done against those ids.
- **Long text lives in `clip_texts`**, not on `clips`. A 90-minute transcript
  is hundreds of kilobytes and would otherwise be dragged into every listing
  query.
- **Embeddings are packed float32 BLOBs** with a precomputed norm: 1.5 KB
  instead of ~9 KB as JSON, which at 100k clips decides whether the index fits
  in page cache.
- **`UtcDateTime`** coerces every timestamp to UTC-aware on both read and
  write. SQLite silently drops the offset, which turns "has this lease expired"
  into a `TypeError` or, worse, a silently wrong answer.
- **`BigInteger` for sizes and hashes.** A capture exceeds `INTEGER` range on
  PostgreSQL, and an unsigned 64-bit perceptual hash overflows a signed
  `BIGINT` — hence the explicit signed/unsigned conversion in
  `util/hashing.py`. (Both were real crashes, now covered by tests.)

### PostgreSQL migration

The ORM is dialect-neutral: JSON becomes JSONB via a variant, timestamps are
timezone-aware, no `AUTOINCREMENT` tricks. SQLite-specific behaviour is
confined to two places — the pragma setup in `db/base.py` and the FTS backend
in `search/fts.py`, which sits behind a protocol. Changing
`database.url` and re-running `bb init` is the whole migration; a PostgreSQL
keyword backend using `tsvector` would be a third class implementing the same
protocol.

---

## 4. Concurrency and SQLite

Many worker threads against one SQLite file needs care, all of it concentrated
in `db/base.py`:

- **WAL** so the dashboard can read while workers write.
- **`busy_timeout`** so a contended write waits instead of raising.
- **`BEGIN IMMEDIATE`** for write transactions (`write_session()`), taking the
  write lock up front. Without this, two threads that read-then-write the same
  rows can both hold read locks and deadlock on upgrade.
- **Foreign keys ON**, since SQLite disables them by default and the cascade
  design depends on them.

Every one of these is a no-op on PostgreSQL, so the same code serves both.

`test_queue.py::test_concurrent_claims_never_double_book` runs six threads
against twenty jobs and asserts no job is claimed twice — the property
everything else depends on.

---

## 5. Failure handling

Failures are typed, and the type decides the response:

| Exception | Response |
| --- | --- |
| `RetryableError` | Exponential backoff with jitter, up to `max_attempts`. |
| `PermanentError` | Straight to `dead`. No attempts wasted on a file that will never decode. |
| `BackendUnavailableError` | Recorded as `skipped` for that analyzer; clip continues. |
| `AnalyzerError` | Recorded as `failed` for that analyzer only; clip continues. |
| Anything else | Treated as retryable, logged with a traceback. |

Jitter matters: when a NAS goes offline every worker fails at the same instant,
and without jitter they retry in lockstep forever.

The `has_runnable_work()` horizon is a subtle one worth knowing about. A job in
backoff is `pending` with a future `available_at`; counting it as runnable
would keep a draining run alive doing nothing until the backoff elapsed.
Anything beyond a 60-second horizon is left for the next run, which re-enqueues
it during recovery.

---

## 6. Scaling to 100 TB

| Concern | Approach |
| --- | --- |
| Identity checks | Cheap size+head+tail signature first; full SHA256 only on a hit. |
| Decode cost | One ffmpeg pass for all segmentation signals; frames extracted once and shared by every vision analyzer. |
| Duplicate comparison | Duration blocking turns O(n²) into a small candidate set. |
| Hot-table width | Transcripts and OCR in a side table; embeddings packed binary. |
| Queue growth | `purge_completed()` trims finished jobs; `EventRepository.purge()` trims the log. |
| Memory | `yield_per` batching on full-library scans; `low_memory` halves frame sampling. |
| Duplicate storage | Alternates moved to `Duplicates/`, never copied. |
| Directory size | Library is subdivided by category, then year/brand/series. |

---

## 7. Extension points

**A new analyzer** — subclass `Analyzer`, decorate with `@register`, import it
in `analyze/__init__.py`. The pipeline discovers it, notices no clip has a
result for it, and backfills the library on the next run.

**A new archive source** — subclass `ArchiveSource`, decorate with
`@register_source`. The download manager, queue and pipeline need no changes.

**A new keyword backend** — implement the `KeywordBackend` protocol.

**Better classification** — add votes, or edit `knowledge.py`, then run
`bb recategorize`. No models re-run.

**An LLM summarizer** — `analyze/summarize.py` is extractive by design (for
thirty-second commercials, extractive is more truthful than a paraphrase and
costs nothing). Replacing it means implementing the same interface and bumping
the version; the library re-summarizes itself.

---

## 8. What is deliberately *not* here

- **A broker, a scheduler daemon, or a service mesh.** One command, one
  process, one database file.
- **An ANN index.** Brute force is fast enough at the stated scale, and it
  cannot go stale.
- **Automatic deletion of anything.** Storage is cheaper than a mistake you
  cannot undo at 3 a.m.
- **Face recognition that invents names.** Detection is model work; *naming*
  needs a reference to compare against, so it matches an operator-supplied
  gallery and corroborates names that OCR or speech already found. A face
  alone is reported as a face.
- **An LLM in the query path.** Rules are inspectable and free.
