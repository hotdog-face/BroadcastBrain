# BroadcastBrain

Autonomous media ingestion for [Fieldstation 42](https://github.com/shane-mason/FieldStation42).
Turns a pile of archived television — Internet Archive downloads, VHS captures,
DVR recordings, DVD rips — into a segmented, analyzed, deduplicated, searchable
library, ready to be scheduled as 2002–2006 American cable.

Part of [Retro Cable TV](https://www.mitchellv.me/work/retro-tv), a CRT cable box for 2002 to 2006 TV · by [Mitchell Vizensky](https://www.mitchellv.me), game UX designer

```
Incoming/ → discovery → segmentation → AI analysis → metadata → dedupe
          → categorization → Library/ → Fieldstation Export/
```

Your job is three steps:

1. Drop videos into `Incoming/`.
2. Run `bb run`.
3. Wake up to a processed library.

---

## Install

Requires **Python 3.9+** and **ffmpeg** (everything else is optional).

```bash
git clone <this repo> && cd BroadcastBrain
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then add whatever AI you want. Each group is independent, and **the pipeline
runs without any of them** — analyzers whose backend is missing record
`skipped`, and `bb reanalyze --missing` backfills them later without redoing
anything else.

```bash
pip install -e '.[all]'          # everything except vendor GPU libraries
pip install -e '.[speech]'       # Whisper transcription
pip install -e '.[ocr]'          # on-screen text (also needs the tesseract binary)
pip install -e '.[vision]'       # logo / scene analysis
pip install -e '.[embeddings]'   # semantic search
pip install -e '.[dashboard]'    # the local web UI
```

External binaries: `ffmpeg` (required), `tesseract` (OCR), `chromaprint`
(audio fingerprinting), `7z` (reading `.iso` without mounting).

```bash
bb init      # create the directory tree, database and a starter config
bb doctor    # what is installed, what each missing piece would buy you
```

## Use

```bash
bb run                  # process everything in Incoming, export, write a report
bb run --watch          # ...then keep watching for new files. This is the overnight command.
bb dashboard            # live UI at http://127.0.0.1:8842
```

Search in plain English:

```bash
bb search "every Taco Bell commercial"
bb search "all MTV clips from 2004"
bb search "every appearance of Billy Mays"
bb search "music videos by Green Day"
bb search "every clip mentioning Star Wars"
bb similar 3f2a91bc
```

Inspect and correct:

```bash
bb status                     # queue, library and duplicate counts
bb clips --category commercial --year 2004
bb clip 3f2a91bc              # everything known about one clip, and *why* it was classified
bb jobs --failed              # what went wrong
bb report                     # last night's summary
bb reanalyze --missing        # after installing a backend
bb recategorize               # after changing classification rules — seconds, no models
bb resegment <source-id> --min-score 0.8   # a recording that split badly
```

Fetch material from Internet Archive:

```bash
bb fetch --list-presets
bb fetch --preset mtv-2004 --preset discovery
bb fetch "recorded off tv 2003" --plan-only
```

## What it does to a recording

A six-hour cable capture goes in. Out the other side:

**Segmentation** does not rely on black frames alone — real archived TV often
has none. It also watches for detectors that have stopped being informative:
on animation, scene and logo detection fire ~20 times a minute and would shred
a recording into shot-length fragments, so their weight is scaled down
automatically. One ffmpeg decode pass collects seven independent signals (black,
silence, scene change, freeze, aspect change, loudness jump, and the network
bug appearing or disappearing in a screen corner), which are then clustered and
scored into cut points. A bonus is applied to boundaries landing on the
15/30/60-second grid US cable actually ran on, which is what recovers the
individual spots inside a commercial pod.

**Analysis** runs a suite of independent analyzers — OCR, Whisper
transcription, logo detection, face detection, object detection, scene
character, audio fingerprinting, then estimation, summarization and embedding.
Each one's result is stored against `(clip, analyzer, version)`.

**Categorization** is a weighted vote. Analyzers do not classify; they vote
with a reason, and every vote is stored. `bb clip <id>` shows you exactly why
something was labelled a promo. Nothing that fails to clear the confidence bar
is guessed at — it is filed as `Unknown`, kept, and remains searchable.

**Deduplication** compares SHA256, a perceptual hash sequence and a Chromaprint
fingerprint, blocked by duration so the comparison stays cheap at scale. The
best copy becomes canonical; the rest are marked as alternates and filed under
`Duplicates/<canonical-id>/` instead of the library, so browsing `Commercials/`
shows one copy of a spot rather than four. **Nothing is ever deleted** — the
alternates keep their rows and stay searchable.

**Export** writes a Fieldstation-ready directory with ready-made schedule
blocks — a program, then a pod of spots with a network promo in the middle and
a bumper coming back:

> MythBusters → commercial → Discovery promo → commercial → bumper → Dirty Jobs

## Directory layout

```
BroadcastBrainData/
  Incoming/              drop files here
  Processing/            work in progress; completed originals archived here
  Library/
    Commercials/  TV Shows/  Movies/  Promos/  Station IDs/
    Network Bumpers/  Music Videos/  Unknown/
    Infomercials/  PPV Previews/  Weather/  News Breaks/  TV Guide/
  Duplicates/            alternates, grouped by the copy that won
  Rejected/              unusable files, with a .reason.txt beside each
  Database/  Logs/  Thumbnails/  Transcripts/  Metadata/  Downloads/  Cache/
  Fieldstation Export/   metadata.json, channels.json, schedule.json, catalog/, thumbnails/
```

## Configuration

Three layers, each overriding the one before: built-in defaults →
`broadcastbrain.toml` → environment variables.

```toml
[paths]
root = "/Volumes/Archive/BroadcastBrain"

[workers]
cpu_workers = 0        # 0 = auto
gpu_workers = 1

[segmentation]
min_boundary_score = 1.05   # lower cuts more aggressively (soft VHS transitions)
```

```bash
BB_WORKERS__CPU_WORKERS=4 bb run      # `__` separates nested keys
bb config                             # show what actually resolved, and from where
```

No path is hardcoded anywhere; every threshold lives in
[`constants.py`](src/broadcastbrain/constants.py) with a comment saying why.

## Interruption and resumption

Kill it at any point — `Ctrl-C`, power cut, OOM. On the next run:

- Jobs whose worker vanished are reclaimed by lease expiry.
- Anything that finished a stage but never queued the next is re-queued.
- Already-processed sources are recognized by SHA256 and skipped instantly.
- Already-analyzed clips are skipped per analyzer, by version.

The cost of an interruption is bounded by the single stage that was in flight.
This is covered by tests (`tests/test_pipeline.py::TestResumability`), not just
by intent.

## PostgreSQL

The ORM is dialect-neutral and SQLite-specific behaviour is confined to the
engine setup and the search backend. To migrate:

```toml
[database]
url = "postgresql+psycopg://user:pass@localhost/broadcastbrain"
```

```bash
pip install -e '.[postgres]' && bb init
```

## Development

```bash
pip install -e '.[dev]'
make test          # 208 tests, ~2 minutes, no ffmpeg required
make lint          # ruff
make typecheck     # mypy --strict
```

The test suite includes a full end-to-end pipeline run against a stub ffmpeg,
so discovery → segmentation → analysis → dedupe → categorization →
materialization → export is exercised on every commit without a single real
video file.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the reasoning behind
it.

## License

MIT.
