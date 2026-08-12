"""Boundary fusion: turning noisy signals into confident cut points.

The collector produces hundreds of overlapping observations. This module turns
them into a small set of boundaries, then into segments. The algorithm is
deliberately simple enough to reason about and to unit test without ffmpeg:

1. **Cluster.** Signals within ``cluster_window_s`` of each other describe the
   same transition. They collapse into one candidate at their weighted centre.
2. **Score.** Each candidate's score is the sum of ``weight x strength`` over
   its distinct signal kinds. Distinct kinds matter -- three scene-change hits
   in a row are one opinion, whereas black + silence + logo-drop is three
   independent ones and should score much higher.
3. **Grid bonus.** US cable in this era ran on a hard 15/30/60-second ad grid.
   A candidate landing near a grid multiple, measured from the previous
   accepted boundary, gets a bonus. This is what lets the fuser find the
   internal boundaries of a commercial pod where the black frames are too
   short for ``blackdetect`` to have caught them all.
4. **Select.** Candidates above threshold become boundaries, subject to the
   minimum segment length.
5. **Repair.** Segments that are too short are merged; segments that are far
   too long are re-split against the best sub-threshold candidates available,
   so a missed boundary costs you one oversized clip rather than an entire
   unusable six-hour "clip".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from ..constants import (
    AD_GRID_SECONDS,
    AD_GRID_TOLERANCE_S,
    ADAPTIVE_SIGNAL_KINDS,
    BOUNDARY_STRONG_SCORE,
    SIGNAL_NOISE_RATE_PER_MINUTE,
    SIGNAL_NOISE_WEIGHT_FLOOR,
)
from ..types import Boundary, Segment, Signal, SignalKind

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids a config <-> segment cycle
    from ..config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionConfig:
    """Everything the fuser needs, decoupled from the global Settings object.

    Keeping this a plain value object is what lets the whole segmentation
    algorithm be tested with a list of floats and no ffmpeg anywhere.
    """

    cluster_window_s: float
    min_boundary_score: float
    min_segment_s: float
    max_segment_s: float
    weights: dict[str, float]
    grid_bonus: float = 0.25
    grid_seconds: tuple[float, ...] = AD_GRID_SECONDS
    grid_tolerance_s: float = AD_GRID_TOLERANCE_S

    @classmethod
    def from_settings(cls, settings: Settings) -> FusionConfig:
        seg = settings.segmentation
        return cls(
            cluster_window_s=seg.cluster_window_s,
            min_boundary_score=seg.min_boundary_score,
            min_segment_s=seg.min_segment_s,
            max_segment_s=seg.max_segment_s,
            weights=dict(seg.signal_weights),
            grid_bonus=seg.ad_grid_bonus,
        )


def adapt_weights(
    weights: dict[str, float],
    counts: dict[str, int],
    duration_s: float,
    *,
    target_rate_per_minute: float = SIGNAL_NOISE_RATE_PER_MINUTE,
    floor: float = SIGNAL_NOISE_WEIGHT_FLOOR,
    kinds: Sequence[str] = ADAPTIVE_SIGNAL_KINDS,
) -> tuple[dict[str, float], list[str]]:
    """Down-weight detectors that fired so often they carry no information.

    A scene-change detector that fires twenty times a minute is not finding
    programme boundaries -- it is describing the editing style of the content.
    Left at full weight it drowns out the black frames and silences that mark
    the actual transitions, and the recording gets shredded into shot-length
    fragments.

    The correction is proportional: a detector at twice the expected rate keeps
    half its weight, at ten times it keeps a tenth, down to a floor so it can
    still break a tie. Detectors firing at a plausible rate are untouched, so
    on ordinary live-action broadcast this function does nothing at all.

    Returns:
        ``(adjusted_weights, notes)`` -- the notes are kept for the audit trail
        so an operator can see what was adapted and why.
    """
    if duration_s <= 0:
        return dict(weights), []

    minutes = duration_s / 60.0
    adjusted = dict(weights)
    notes: list[str] = []

    for kind in kinds:
        rate = counts.get(kind, 0) / minutes
        if rate <= target_rate_per_minute:
            continue
        scale = max(floor, target_rate_per_minute / rate)
        adjusted[kind] = adjusted.get(kind, 0.0) * scale
        notes.append(f"{kind}: {rate:.1f}/min, weight x{scale:.2f}")

    return adjusted, notes


@dataclass
class Candidate:
    """A clustered group of signals proposing one boundary."""

    time_s: float
    kinds: dict[SignalKind, float]  # kind -> best strength seen
    weight_sum: float = 0.0

    def score(self, weights: dict[str, float]) -> float:
        return sum(
            weights.get(kind.value, 0.0) * strength for kind, strength in self.kinds.items()
        )


def cluster_signals(signals: Sequence[Signal], window_s: float) -> list[Candidate]:
    """Collapse signals into candidates using a single ordered sweep.

    O(n log n) for the sort, O(n) for the sweep -- important because a long
    capture with loudness enabled produces tens of thousands of signals.
    """
    if not signals:
        return []
    ordered = sorted(signals, key=lambda s: s.time_s)
    candidates: list[Candidate] = []

    current_times: list[tuple[float, float]] = []  # (time, weight)
    current_kinds: dict[SignalKind, float] = {}
    anchor = ordered[0].time_s

    def close() -> None:
        if not current_kinds:
            return
        total_weight = sum(w for _t, w in current_times) or 1.0
        centre = sum(t * w for t, w in current_times) / total_weight
        candidates.append(Candidate(time_s=centre, kinds=dict(current_kinds), weight_sum=total_weight))

    for signal in ordered:
        if signal.time_s - anchor > window_s:
            close()
            current_times = []
            current_kinds = {}
            anchor = signal.time_s
        current_times.append((signal.time_s, max(0.05, signal.strength)))
        previous = current_kinds.get(signal.kind, 0.0)
        current_kinds[signal.kind] = max(previous, signal.strength)
    close()
    return candidates


def grid_alignment_bonus(
    time_s: float, previous_boundary_s: float, config: FusionConfig
) -> float:
    """Bonus for landing on the commercial grid, measured from the last cut.

    A 30-second spot ending exactly 30 seconds after the previous boundary is
    far more likely to be a real boundary than one at 27.4 seconds.
    """
    elapsed = time_s - previous_boundary_s
    if elapsed <= 0:
        return 0.0
    for unit in config.grid_seconds:
        remainder = elapsed % unit
        distance = min(remainder, unit - remainder)
        if distance <= config.grid_tolerance_s:
            # Closer to the grid line => more of the bonus.
            proximity = 1.0 - (distance / config.grid_tolerance_s)
            return config.grid_bonus * proximity
    return 0.0


def select_boundaries(
    candidates: Sequence[Candidate], duration_s: float, config: FusionConfig
) -> list[Boundary]:
    """Choose which candidates become real cuts.

    The scan is left-to-right and stateful because both the grid bonus and the
    minimum-segment rule depend on where the previous accepted boundary was.
    """
    boundaries: list[Boundary] = []
    previous = 0.0
    for candidate in sorted(candidates, key=lambda c: c.time_s):
        if candidate.time_s <= 0.0 or candidate.time_s >= duration_s:
            continue
        score = candidate.score(config.weights)
        score += grid_alignment_bonus(candidate.time_s, previous, config)

        if score < config.min_boundary_score:
            continue
        if candidate.time_s - previous < config.min_segment_s:
            # Too close to the last cut. Keep whichever is stronger.
            if boundaries and score > boundaries[-1].score:
                boundaries[-1] = Boundary(
                    time_s=candidate.time_s,
                    score=score,
                    kinds=tuple(candidate.kinds),
                )
                previous = candidate.time_s
            continue
        boundaries.append(
            Boundary(time_s=candidate.time_s, score=score, kinds=tuple(candidate.kinds))
        )
        previous = candidate.time_s
    return boundaries


def split_oversized(
    segments: list[Segment], candidates: Sequence[Candidate], config: FusionConfig
) -> list[Segment]:
    """Re-split segments that exceed ``max_segment_s``.

    A missed boundary in a six-hour capture would otherwise yield one enormous
    clip. Rather than lowering the global threshold (which would shred the rest
    of the recording into fragments), the threshold is relaxed *only inside*
    the oversized span, using the best candidate available there.
    """
    if not segments:
        return segments

    result: list[Segment] = []
    relaxed = config.min_boundary_score * 0.55

    for segment in segments:
        if segment.duration_s <= config.max_segment_s:
            result.append(segment)
            continue

        inner = [
            c
            for c in candidates
            if segment.start_s + config.min_segment_s < c.time_s < segment.end_s - config.min_segment_s
            and c.score(config.weights) >= relaxed
        ]
        if not inner:
            log.debug(
                "oversized segment %.1fs has no usable inner candidate; keeping it whole",
                segment.duration_s,
            )
            result.append(segment)
            continue

        cuts = _greedy_cuts(inner, segment.start_s, segment.end_s, config)
        edges = [segment.start_s, *cuts, segment.end_s]
        for i in range(len(edges) - 1):
            result.append(
                Segment(
                    index=0,  # renumbered by the caller
                    start_s=edges[i],
                    end_s=edges[i + 1],
                    signal_kinds=segment.signal_kinds,
                )
            )
    return _renumber(result)


def _greedy_cuts(
    candidates: Sequence[Candidate], start_s: float, end_s: float, config: FusionConfig
) -> list[float]:
    """Pick cuts inside a span, strongest first, honouring the minimum length."""
    chosen: list[float] = []
    for candidate in sorted(candidates, key=lambda c: c.score(config.weights), reverse=True):
        if (end_s - start_s) / max(1, len(chosen) + 1) < config.min_segment_s:
            break
        if all(abs(candidate.time_s - t) >= config.min_segment_s for t in chosen):
            chosen.append(candidate.time_s)
    return sorted(chosen)


def merge_short(segments: list[Segment], config: FusionConfig) -> list[Segment]:
    """Fold too-short segments into the neighbour they most likely belong to.

    Direction is chosen by boundary strength: a fragment is absorbed across its
    *weaker* edge, because the stronger edge is the one that is probably a real
    transition.
    """
    if len(segments) < 2:
        return _renumber(segments)

    #: Merging can create a segment that is still short (two 1-second
    #: fragments make a 2-second one), so sweep until stable.
    working = list(segments)
    for _pass in range(3):
        merged = _merge_short_once(working, config)
        if len(merged) == len(working):
            working = merged
            break
        working = merged
    return _renumber(working)


def _merge_short_once(segments: Sequence[Segment], config: FusionConfig) -> list[Segment]:
    pending = list(segments)
    result: list[Segment] = []
    index = 0
    while index < len(pending):
        segment = pending[index]
        if segment.duration_s >= config.min_segment_s:
            result.append(segment)
            index += 1
            continue

        has_next = index + 1 < len(pending)
        # Absorb the fragment across its weaker edge; the stronger edge is
        # more likely to be a genuine transition worth preserving.
        merge_back = bool(result) and (not has_next or segment.start_score <= segment.end_score)
        if merge_back:
            result.append(_join(result.pop(), segment))
        elif has_next:
            pending[index + 1] = _join(segment, pending[index + 1])
        else:
            result.append(segment)
        index += 1
    return result


def _join(left: Segment, right: Segment) -> Segment:
    return Segment(
        index=left.index,
        start_s=left.start_s,
        end_s=right.end_s,
        start_score=left.start_score,
        end_score=right.end_score,
        signal_kinds=tuple(sorted(set(left.signal_kinds) | set(right.signal_kinds), key=lambda k: k.value)),
    )


def boundaries_to_segments(
    boundaries: Sequence[Boundary], duration_s: float
) -> list[Segment]:
    """Convert cut points into spans covering the whole source.

    Every second of the source ends up in exactly one segment. Nothing is
    dropped -- material between boundaries that looks like nothing in
    particular still becomes a clip, and is classified ``unknown`` later.
    """
    edges: list[tuple[float, float, tuple[SignalKind, ...]]] = [(0.0, 0.0, ())]
    edges += [(b.time_s, b.score, b.kinds) for b in boundaries]
    edges.append((duration_s, 0.0, ()))

    segments: list[Segment] = []
    for index in range(len(edges) - 1):
        start_s, start_score, start_kinds = edges[index]
        end_s, end_score, end_kinds = edges[index + 1]
        if end_s - start_s <= 0:
            continue
        segments.append(
            Segment(
                index=index,
                start_s=start_s,
                end_s=end_s,
                start_score=start_score,
                end_score=end_score,
                signal_kinds=tuple(set(start_kinds) | set(end_kinds)),
            )
        )
    return _renumber(segments)


def fuse(
    signals: Iterable[Signal], duration_s: float, config: FusionConfig
) -> tuple[list[Segment], list[Boundary]]:
    """Run the whole fusion pipeline.

    Returns:
        ``(segments, boundaries)`` -- segments cover ``[0, duration_s)``
        exactly, and boundaries are kept for the audit trail stored on the
        source row.
    """
    signal_list = list(signals)
    if duration_s <= 0:
        return [], []

    candidates = cluster_signals(signal_list, config.cluster_window_s)
    boundaries = select_boundaries(candidates, duration_s, config)
    segments = boundaries_to_segments(boundaries, duration_s)
    segments = merge_short(segments, config)
    segments = split_oversized(segments, candidates, config)

    log.debug(
        "fusion: %d signals -> %d candidates -> %d boundaries -> %d segments",
        len(signal_list), len(candidates), len(boundaries), len(segments),
    )
    return segments, boundaries


def _renumber(segments: list[Segment]) -> list[Segment]:
    return [
        Segment(
            index=index,
            start_s=s.start_s,
            end_s=s.end_s,
            start_score=s.start_score,
            end_score=s.end_score,
            signal_kinds=s.signal_kinds,
        )
        for index, s in enumerate(segments)
    ]


def strong_boundary_count(boundaries: Sequence[Boundary]) -> int:
    """How many cuts we are confident about -- a segmentation quality signal."""
    return sum(1 for b in boundaries if b.score >= BOUNDARY_STRONG_SCORE)


def describe(segments: Sequence[Segment], boundaries: Sequence[Boundary]) -> dict[str, object]:
    """Audit record stored on the source, so a bad run can be diagnosed later."""
    durations = [s.duration_s for s in segments]
    return {
        "segment_count": len(segments),
        "boundary_count": len(boundaries),
        "strong_boundaries": strong_boundary_count(boundaries),
        "median_duration_s": round(_median(durations), 2) if durations else 0.0,
        "shortest_s": round(min(durations), 2) if durations else 0.0,
        "longest_s": round(max(durations), 2) if durations else 0.0,
        "boundaries": [b.as_dict() for b in boundaries[:2000]],
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
