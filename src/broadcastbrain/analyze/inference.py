"""Estimation: network, air year, genre and series, from accumulated evidence.

This analyzer runs late and reads nothing but other analyzers' payloads. That
separation is deliberate. OCR's job is to report what is written on the screen;
deciding that "the copyright line says 2004, the bug says Discovery, therefore
this is a 2004 Discovery clip" is a different kind of reasoning, and keeping it
in one place means the logic can be improved -- and the whole library
re-labelled by bumping this analyzer's version -- without re-running a single
model.

Every estimate carries a confidence and the evidence that produced it.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Optional

from ..constants import (
    COMMERCIAL_SLOT_SECONDS,
    COMMERCIAL_SLOT_TOLERANCE_S,
    DURATION_PRIORS,
    NETWORK_BUMPER_MAX_S,
    PLAUSIBLE_YEAR_RANGE,
    PROGRAMME_SEGMENT_MAX_S,
    PROGRAMME_SEGMENT_MIN_S,
    STATION_ID_MAX_S,
)
from ..types import Category, EntityKind
from . import knowledge
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)


#: A path component that is exactly a four-digit year, e.g. ".../2004/ad.mp4".
_PATH_YEAR = re.compile(r"^(19[3-9]\d|20[0-4]\d)$")


def _year_from_path(path: str) -> Optional[int]:
    """A year stated by the directory a recording was filed in.

    Only an exact match counts: a folder called "2004" is a date, whereas
    "Top 2004 Ads Vol 3" is a title that happens to contain digits. The era
    bounds are not applied -- an operator filing under 1998 means 1998 --
    but implausible values are still rejected by the pattern itself.
    """
    if not path:
        return None
    for part in reversed(PurePath(path).parts):
        match = _PATH_YEAR.match(part.strip())
        if match:
            return int(match.group(0))
    return None


@dataclass
class Estimate:
    """A value, how sure we are, and why."""

    value: Optional[Any]
    confidence: float
    evidence: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:8],
        }


@register
class EstimationAnalyzer(Analyzer):
    """Fuse every other analyzer's output into network / year / genre / type."""

    name = "estimation"
    version = f"13+kb{knowledge.KNOWLEDGE_VERSION}"
    requires = ()
    order = 70
    description = "Network, air-year, genre and content-type estimation"

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        result = AnalyzerResult(backend="rules")

        network = self._estimate_network(ctx)
        year = self._estimate_year(ctx, network)
        series = self._estimate_series(ctx)
        genre = self._estimate_genre(ctx, network, series)

        result.payload = {
            "network": network.as_dict(),
            "air_year": year.as_dict(),
            "series": series.as_dict(),
            "genre": genre.as_dict(),
            "duration_priors": self._duration_prior_scores(ctx.clip.duration_s),
        }
        result.confidence = max(network.confidence, year.confidence, 0.2)

        # Every estimated field is written on every run, including when the
        # answer is now "unknown". Writing only non-empty values means a
        # correction can never undo a previous mistake: the clip keeps the old
        # network forever, with the evidence that justified it long gone. For a
        # system whose whole premise is that re-analysis fixes things, a field
        # that can be set but not cleared is a slow-growing lie.
        result.fields["network"] = str(network.value) if network.value else None
        result.fields["network_confidence"] = round(network.confidence, 3)
        result.fields["air_year"] = int(year.value) if year.value else None
        result.fields["air_year_confidence"] = round(year.confidence, 3)
        result.fields["series"] = str(series.value) if series.value else None
        result.fields["genre"] = str(genre.value) if genre.value else None

        if network.value:
            result.add_entity(EntityKind.NETWORK, str(network.value), network.confidence)
            result.add_tag(str(network.value).lower(), network.confidence)
        if year.value:
            result.add_tag(str(year.value), year.confidence)
        if series.value:
            result.add_entity(EntityKind.SHOW, str(series.value), series.confidence)
        if genre.value:
            result.add_tag(str(genre.value).lower(), genre.confidence * 0.8)

        self._vote_content_type(ctx, result)
        return result

    # -- network ------------------------------------------------------------

    def _estimate_network(self, ctx: AnalysisContext) -> Estimate:
        """Weighted vote across OCR, speech, and any show that was identified.

        On-screen text outranks speech: a bug burned into the corner is the
        network, whereas a network *named* in dialogue is often a competitor
        being mentioned in a promo.
        """
        scores: dict[str, float] = defaultdict(float)
        evidence: list[str] = []

        for match in knowledge.match_networks(ctx.text_of("ocr", "text")):
            scores[match.label] += match.confidence * 1.0
            evidence.append(f"ocr:{match.label}({match.evidence})")
        for match in knowledge.match_networks(ctx.text_of("speech", "text")):
            scores[match.label] += match.confidence * 0.6
            evidence.append(f"speech:{match.label}({match.evidence})")

        text = ctx.all_text()
        for match in knowledge.match_shows(text):
            network = str(match.extra.get("network") or "")
            if network:
                scores[network] += match.confidence * 0.7
                evidence.append(f"show:{match.label}->{network}")

        # A named programming block ("Toonami") identifies its parent network
        # as reliably as the on-screen bug does.
        for match in knowledge.match_blocks(text):
            network = str(match.extra.get("network") or "")
            if network:
                scores[network] += match.confidence * 0.9
                evidence.append(f"block:{match.label}->{network}")

        # A visible bug does not say *which* network, but it does say there is
        # one -- which supports whatever the text already suggested.
        logo = ctx.payload_of("logo")
        if logo.get("has_logo") and scores:
            best = max(scores, key=lambda k: scores[k])
            scores[best] += 0.25
            evidence.append(f"logo:{logo.get('logo_corner')}")

        if not scores:
            return Estimate(None, 0.0, evidence)

        winner = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        # Confidence blends absolute support with how dominant the winner is.
        share = scores[winner] / total if total else 0.0
        confidence = min(0.98, 0.5 * min(1.0, scores[winner]) + 0.5 * share)
        return Estimate(winner, round(confidence, 3), evidence)

    # -- year ---------------------------------------------------------------

    def _estimate_year(self, ctx: AnalysisContext, network: Estimate) -> Estimate:
        estimate = self._estimate_year_inner(ctx, network)
        low, high = PLAUSIBLE_YEAR_RANGE
        if estimate.value is not None and not (low <= int(estimate.value) <= high):
            log.debug("discarding implausible year %s", estimate.value)
            return Estimate(None, 0.0, [*estimate.evidence, f"rejected {estimate.value}: implausible"])
        return estimate

    def _estimate_year_inner(self, ctx: AnalysisContext, network: Estimate) -> Estimate:
        """Date the clip.

        Priority order, strongest first:

        1. A copyright year read off an end card. Broadcast material is
           legally obliged to carry one and it is almost never wrong.
        2. Any other four-digit year seen on screen (a promo's "premieres
           2004", a news lower-third date).
        3. The era in which the identified network carried that identity.
        4. The configured target era, as a last resort at low confidence.
        """
        evidence: list[str] = []
        ocr = ctx.payload_of("ocr")

        copyright_year = ocr.get("copyright_year")
        if isinstance(copyright_year, int):
            evidence.append(f"copyright:{copyright_year}")
            return Estimate(copyright_year, 0.9, evidence)

        # An operator who filed a clip under "Commercials/2004/" has stated its
        # year outright. That is better evidence than anything guessed from the
        # picture, and second only to a copyright line read off the screen.
        foldered = _year_from_path(ctx.clip.source_path)
        if foldered is not None:
            evidence.append(f"filed under a {foldered} folder")
            return Estimate(foldered, 0.8, evidence)

        years_seen = [int(y) for y in (ocr.get("years_seen") or []) if isinstance(y, (int, float))]
        if years_seen:
            # The latest year on screen is the best estimate: a 2004 broadcast
            # can show a 1998 copyright on library footage, but not vice versa.
            best = max(years_seen)
            evidence.append(f"on-screen years:{sorted(set(years_seen))}")
            return Estimate(best, 0.65, evidence)

        # A network's lifetime bounds a *range*, not a year. Collapsing
        # "MTV existed 2002-2006" to the midpoint 2004 is the same invention as
        # the era fallback below it, just wearing evidence as a disguise: on a
        # real library it stamped 110 clips with a 2004 they had no claim to,
        # in the same bucket as clips genuinely filed under 2004. The network
        # is already recorded in its own column, so nothing is lost by
        # declining to guess a year from it.
        if network.value and knowledge.network_era(str(network.value)):
            evidence.append(
                f"{network.value} bounds a range, not a year -- not used for dating"
            )

        # Nothing dated this clip. Returning the middle of the configured era
        # would put a specific, confident-looking year on it -- and on a real
        # library that produced 288 clips all claiming 2004, sitting in the
        # same histogram bucket as the ones genuinely filed under 2004. An
        # unknown date is recorded as unknown.
        evidence.append("no dating evidence found")
        return Estimate(None, 0.0, evidence)

    # -- series / genre -----------------------------------------------------

    def _estimate_series(self, ctx: AnalysisContext) -> Estimate:
        matches = knowledge.match_shows(ctx.all_text())
        if not matches:
            return Estimate(None, 0.0, [])
        best = max(matches, key=lambda m: m.confidence)
        return Estimate(best.label, best.confidence, [f"title match:{best.evidence}"])

    def _estimate_genre(
        self, ctx: AnalysisContext, network: Estimate, series: Estimate
    ) -> Estimate:
        evidence: list[str] = []
        scene = ctx.payload_of("scene")

        if scene.get("animation_score", 0.0) >= 0.65:
            evidence.append(f"animation score {scene.get('animation_score')}")
            return Estimate("animation", 0.6, evidence)

        if series.value:
            network_name = knowledge.network_for_show(str(series.value))
            if network_name:
                genres = knowledge.genres_for_network(network_name)
                if genres:
                    evidence.append(f"show {series.value} -> {network_name}")
                    return Estimate(genres[0], 0.5, evidence)

        if network.value:
            genres = knowledge.genres_for_network(str(network.value))
            if genres:
                evidence.append(f"network {network.value} genres {list(genres)}")
                return Estimate(genres[0], network.confidence * 0.55, evidence)

        return Estimate(None, 0.0, evidence)

    # -- content type -------------------------------------------------------

    def _duration_prior_scores(self, duration_s: float) -> dict[str, float]:
        """How well the clip's duration fits each content type's typical range."""
        scores: dict[str, float] = {}
        for label, (low, high) in DURATION_PRIORS.items():
            if low <= duration_s <= high:
                # Peak in the middle of the range, taper toward the edges.
                span = high - low
                centre = (low + high) / 2.0
                closeness = 1.0 - abs(duration_s - centre) / (span / 2.0) if span else 1.0
                scores[label] = round(0.35 + 0.35 * max(0.0, closeness), 3)
        return scores

    def _vote_content_type(self, ctx: AnalysisContext, result: AnalyzerResult) -> None:
        """Duration and pacing votes, on top of the text-driven ones."""
        duration = ctx.clip.duration_s

        for label, score in self._duration_prior_scores(duration).items():
            try:
                category = Category(label)
            except ValueError:  # pragma: no cover - priors table drift
                continue
            result.vote(category, score, f"duration {duration:.0f}s fits {label}")

        # Landing on a standard ad slot length is strong, independent evidence.
        for slot in COMMERCIAL_SLOT_SECONDS:
            if abs(duration - slot) <= COMMERCIAL_SLOT_TOLERANCE_S:
                result.vote(
                    Category.COMMERCIAL, 0.45, f"duration matches the {slot:.0f}s ad slot"
                )
                result.add_tag(f"{int(slot)}s-spot", 0.8)
                break

        # Only reason about speech if the speech analyzer actually ran. An
        # absent payload means "not measured", and treating that as "no speech"
        # turns every clip on a box without Whisper into a music-video
        # candidate. Missing evidence must never become evidence.
        speech = ctx.payload_of("speech")
        if "speech_ratio" in speech:
            speech_ratio = float(speech.get("speech_ratio") or 0.0)
            if speech_ratio >= 0.7 and duration <= 70:
                result.vote(Category.COMMERCIAL, 0.3, "wall-to-wall voiceover in a short clip")
            if speech_ratio <= 0.1 and duration >= 100:
                # Weak on purpose. Absence of speech is not presence of music:
                # an action sequence, a montage and a music video all look the
                # same to a transcriber. Deciding "music video" needs positive
                # evidence -- an identified track from the audio fingerprint --
                # which is what the music analyzer contributes when it can.
                result.vote(
                    Category.MUSIC_VIDEO, 0.12,
                    "little speech (weak: no music actually identified)",
                )

        logo = ctx.payload_of("logo")
        if logo.get("has_logo") is False and duration <= 65:
            result.vote(Category.COMMERCIAL, 0.2, "no network bug on a short clip")

        if duration <= 6:
            result.vote(Category.STATION_ID, 0.4, "very short clip")

        # The decisive interstitial rule. A station ID is a few seconds of the
        # network's own branding: no dialogue to transcribe, no product, just
        # the channel's name on screen. Duration alone cannot separate it from
        # the tail of a spot -- but duration *plus* network branding *minus*
        # any advertised brand can, and that is the whole of what these clips
        # contain. Without this, 48 of them sat in Unknown while the WOC tapes
        # they came from were full of exactly this material.
        branded = bool(knowledge.match_networks(ctx.all_text())) or bool(
            ctx.payload_of("estimation").get("network")
        )
        advertises = bool(knowledge.match_brands(ctx.all_text()))
        if branded and not advertises:
            if duration <= STATION_ID_MAX_S:
                result.vote(
                    Category.STATION_ID, 0.65,
                    f"{duration:.0f}s of network branding with nothing advertised",
                )
            elif duration <= NETWORK_BUMPER_MAX_S:
                result.vote(
                    Category.NETWORK_BUMPER, 0.6,
                    f"{duration:.0f}s of network branding with nothing advertised",
                )
        if duration >= 4200:
            result.vote(Category.MOVIE, 0.35, "feature-length duration")

        # A long continuous run between ad breaks is programming, even though
        # it is far shorter than a whole episode. Captured broadcast is cut
        # into 5-20 minute acts by the breaks themselves, so without this the
        # actual show content -- the most valuable material in the recording --
        # falls through every duration prior and lands in Unknown.
        # A recognized series is the strongest content signal available: it
        # comes from the knowledge base matching a title in OCR or speech. A
        # clip that names a show is that show, not an interstitial and not a
        # music video that happens to run the same length.
        series_name = self._estimate_series(ctx).value
        if series_name:
            result.vote(Category.TV_EPISODE, 0.55, f"identified series: {series_name}")

        if PROGRAMME_SEGMENT_MIN_S <= duration < PROGRAMME_SEGMENT_MAX_S:
            minutes = duration / 60.0
            result.vote(
                Category.TV_EPISODE,
                min(0.6, 0.3 + 0.03 * minutes),
                f"{minutes:.0f}-minute continuous segment between breaks",
            )
