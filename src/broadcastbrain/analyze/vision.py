"""Vision analyzers: network bug, faces, objects and scene character.

All four share the frames the runner extracted once, so the marginal cost of
each is inference only.

A note on face *recognition*: detecting faces is model work, but naming them
requires something to compare against. Rather than pretend, this module does
two honest things -- it matches against an operator-supplied gallery of
labelled portraits (``analysis.faces_gallery``) when one exists, and it
corroborates names that OCR or speech already found in the same clip. A face
detected while the words "Billy Mays here" are spoken is strong evidence; a
face alone is a face.
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from ..constants import LOGO_REGIONS
from ..errors import AnalyzerError
from ..types import Category, EntityKind
from .base import AnalysisContext, Analyzer, AnalyzerResult, sample_frames
from .registry import register

log = logging.getLogger(__name__)

_model_lock = threading.Lock()
_models: dict[str, Any] = {}

#: Above this duration, the absence of a network bug carries no information.
BUG_ABSENCE_MAX_SECONDS = 120.0

#: A station bug occupies a corner and barely changes; the picture behind it
#: does. A corner whose temporal variance is below this fraction of the full
#: frame's variance is treated as carrying a persistent overlay.
LOGO_STABILITY_RATIO = 0.35
#: ...but a corner that never changes because the whole clip is static (a
#: slate) is not a logo. Require the frame as a whole to be moving.
MIN_FRAME_ACTIVITY = 2.0
LOGO_MAX_FRAMES = 12
FACE_MAX_FRAMES = 6
OBJECT_MAX_FRAMES = 8
SCENE_MAX_FRAMES = 12

#: YOLO labels that say something about broadcast content worth tagging.
_INTERESTING_OBJECTS = {
    "person", "car", "truck", "motorcycle", "dog", "cat", "tv", "laptop", "cell phone",
    "bottle", "cup", "pizza", "cake", "sandwich", "hot dog", "donut", "chair", "couch",
    "bed", "book", "clock", "sports ball", "skateboard", "surfboard", "guitar",
}


def _load_cv2() -> Any:
    from .. import capabilities

    if not capabilities.has("opencv"):
        return None
    import cv2

    return cv2


# ---------------------------------------------------------------------------
# Logo / network bug
# ---------------------------------------------------------------------------


@register
class LogoAnalyzer(Analyzer):
    """Detect a persistent corner overlay -- the network bug.

    This is one of the most useful signals in the whole system for 2000s cable:
    the bug is present during programming and drops during national ad breaks,
    so its absence is real evidence that a clip is a commercial.
    """

    name = "logo"
    version = "2"
    requires = ("opencv", "numpy")
    order = 30
    needs_frames = True
    description = "Persistent corner overlay (station bug) detection"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return len(ctx.frames) >= 3

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        cv2 = _load_cv2()
        import numpy as np

        if cv2 is None:  # pragma: no cover - guarded by check_backends
            raise AnalyzerError(self.name, "opencv unavailable")

        frames = sample_frames(ctx.frames, LOGO_MAX_FRAMES)
        images = []
        for frame in frames:
            image = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image.astype("float32"))
        if len(images) < 3:
            raise AnalyzerError(self.name, "not enough readable frames")

        height, width = images[0].shape[:2]
        stack = np.stack([cv2.resize(img, (width, height)) for img in images])

        frame_activity = float(np.mean(np.std(stack, axis=0)))
        corners: dict[str, dict[str, float]] = {}
        for corner, (x, y, w, h) in LOGO_REGIONS.items():
            x0, y0 = int(x * width), int(y * height)
            x1, y1 = int((x + w) * width), int((y + h) * height)
            region = stack[:, y0:y1, x0:x1]
            if region.size == 0:
                continue
            temporal = float(np.mean(np.std(region, axis=0)))
            spatial = float(np.mean(np.std(region.reshape(region.shape[0], -1), axis=1)))
            corners[corner] = {
                "temporal_variance": round(temporal, 3),
                "spatial_detail": round(spatial, 3),
                "stability_ratio": round(temporal / frame_activity, 3) if frame_activity else 1.0,
            }

        best_corner, best = _best_logo_corner(corners, frame_activity)
        result = AnalyzerResult(backend="opencv")
        result.payload = {
            "frame_activity": round(frame_activity, 3),
            "corners": corners,
            "logo_corner": best_corner,
            "has_logo": best_corner is not None,
            "confidence": round(best, 3),
        }
        result.confidence = best

        if best_corner is not None:
            result.add_tag("has-network-bug", best)
            # Programming carries a bug; national spots usually do not.
            result.vote(Category.TV_EPISODE, 0.2 * best, f"network bug in {best_corner}")
        elif frame_activity >= MIN_FRAME_ACTIVITY:
            result.add_tag("no-network-bug", 0.6)
            # A missing bug is evidence of a national spot only on a clip short
            # enough to *be* one. Over a programme segment the bug routinely
            # disappears behind graphics, and the absence says nothing.
            if ctx.clip.duration_s <= BUG_ABSENCE_MAX_SECONDS:
                result.vote(Category.COMMERCIAL, 0.25, "no persistent network bug")
        return result


def _best_logo_corner(
    corners: dict[str, dict[str, float]], frame_activity: float
) -> tuple[Optional[str], float]:
    """Pick the corner most likely to hold a bug, with a confidence."""
    if frame_activity < MIN_FRAME_ACTIVITY:
        # The whole picture is static; "stable corner" means nothing here.
        return None, 0.0
    best_corner: Optional[str] = None
    best_score = 0.0
    for corner, stats in corners.items():
        ratio = stats["stability_ratio"]
        detail = stats["spatial_detail"]
        # Stable *and* containing something: a blank letterbox bar is stable
        # too, and is not a logo.
        if ratio <= LOGO_STABILITY_RATIO and detail > 6.0:
            score = min(1.0, (LOGO_STABILITY_RATIO - ratio) / LOGO_STABILITY_RATIO + detail / 80.0)
            if score > best_score:
                best_corner, best_score = corner, score
    return best_corner, round(best_score, 3)


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------


@register
class FaceAnalyzer(Analyzer):
    """Detect faces and, where possible, name them."""

    name = "faces"
    version = "1"
    requires = ("insightface", "numpy")
    needs_gpu = True
    order = 35
    needs_frames = True
    description = "Face detection + gallery matching + name corroboration"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return bool(ctx.frames)

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        import numpy as np

        app = _face_app(ctx.settings.analysis.face_model)
        cv2 = _load_cv2()
        if cv2 is None:
            raise AnalyzerError(self.name, "opencv is required for frame loading")

        gallery = _load_gallery(ctx.settings.analysis.faces_gallery, app, cv2)
        frames = sample_frames(ctx.frames, FACE_MAX_FRAMES)

        detections: list[dict[str, Any]] = []
        named: dict[str, float] = {}
        for index, frame in enumerate(frames):
            image = cv2.imread(str(frame))
            if image is None:
                continue
            try:
                faces = app.get(image)
            except Exception as exc:
                log.debug("face detection failed on %s: %s", frame.name, exc)
                continue
            for face in faces:
                entry: dict[str, Any] = {
                    "frame": frame.name,
                    "at_s": round(_frame_time(ctx, index, len(frames)), 2),
                    "score": round(float(getattr(face, "det_score", 0.0)), 3),
                    "bbox": [int(v) for v in getattr(face, "bbox", [])],
                }
                embedding = getattr(face, "normed_embedding", None)
                if embedding is not None and gallery:
                    name, similarity = _match_gallery(np, embedding, gallery)
                    if name and similarity >= ctx.settings.analysis.face_match_threshold:
                        entry["name"] = name
                        entry["similarity"] = round(similarity, 3)
                        named[name] = max(named.get(name, 0.0), similarity)
                detections.append(entry)

        result = AnalyzerResult(backend="insightface")
        result.payload = {
            "face_count": len(detections),
            "frames_with_faces": len({d["frame"] for d in detections}),
            "faces": detections[:100],
            "gallery_size": len(gallery),
            "named": named,
        }
        result.confidence = min(1.0, len(detections) / 6.0)

        for name, similarity in named.items():
            result.add_entity(EntityKind.PERSON, name, similarity)
            result.add_tag(name.lower(), similarity)

        # Corroborate names that OCR or speech already surfaced: a detected
        # face plus a spoken name is much stronger than either alone.
        if detections and not named:
            for mention in _people_from_context(ctx):
                result.add_entity(EntityKind.PERSON, mention, 0.6)
                result.payload.setdefault("corroborated", []).append(mention)

        if detections:
            result.add_tag("has-people", 0.9)
            faces_per_frame = len(detections) / max(1, len(frames))
            if faces_per_frame >= 2.0:
                result.add_tag("crowd", 0.6)
        return result


@functools.lru_cache(maxsize=2)
def _face_app(model_name: str) -> Any:
    """Load and cache the InsightFace pipeline."""
    from insightface.app import FaceAnalysis

    from .. import capabilities

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if capabilities.detect_gpu().vendor == "nvidia"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0 if capabilities.detect_gpu().vendor == "nvidia" else -1, det_size=(640, 640))
    return app


def _load_gallery(gallery_dir: Optional[Path], app: Any, cv2: Any) -> dict[str, list[Any]]:
    """Build ``{person: [embeddings]}`` from ``<gallery>/<Name>/*.jpg``.

    Cached per directory: embedding a gallery on every clip would dwarf the
    cost of the actual analysis.
    """
    if gallery_dir is None:
        return {}
    return _gallery_cache(str(gallery_dir), app, cv2)


def _gallery_cache(gallery_path: str, app: Any, cv2: Any) -> dict[str, list[Any]]:
    with _model_lock:
        cached = _models.get(f"gallery:{gallery_path}")
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    directory = Path(gallery_path)
    gallery: dict[str, list[Any]] = {}
    if directory.is_dir():
        for person_dir in sorted(directory.iterdir()):
            if not person_dir.is_dir():
                continue
            embeddings: list[Any] = []
            for image_path in sorted(person_dir.iterdir())[:20]:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                try:
                    faces = app.get(image)
                except Exception:
                    continue
                if faces:
                    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
                    embedding = getattr(best, "normed_embedding", None)
                    if embedding is not None:
                        embeddings.append(embedding)
            if embeddings:
                gallery[person_dir.name] = embeddings
        log.info("face gallery loaded: %d people from %s", len(gallery), directory)

    with _model_lock:
        _models[f"gallery:{gallery_path}"] = gallery
    return gallery


def _match_gallery(np: Any, embedding: Any, gallery: dict[str, list[Any]]) -> tuple[Optional[str], float]:
    """Nearest-neighbour match by cosine similarity over normalized vectors."""
    best_name: Optional[str] = None
    best_score = -1.0
    for name, references in gallery.items():
        for reference in references:
            score = float(np.dot(embedding, reference))
            if score > best_score:
                best_name, best_score = name, score
    return best_name, max(0.0, best_score)


def _people_from_context(ctx: AnalysisContext) -> list[str]:
    """Names that OCR or speech already found in this clip."""
    from . import knowledge

    text = ctx.all_text()
    return [match.label for match in knowledge.match_people(text)]


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@register
class ObjectAnalyzer(Analyzer):
    """Detect objects with YOLO -- what is physically in the picture."""

    name = "objects"
    version = "1"
    requires = ("ultralytics",)
    needs_gpu = True
    order = 40
    needs_frames = True
    description = "Object detection (YOLOv8)"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return bool(ctx.frames)

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        model = _yolo_model(ctx.settings.analysis.object_model)
        frames = sample_frames(ctx.frames, OBJECT_MAX_FRAMES)

        counts: dict[str, int] = {}
        confidences: dict[str, float] = {}
        try:
            predictions = model.predict([str(f) for f in frames], verbose=False)
        except Exception as exc:
            raise AnalyzerError(self.name, f"object detection failed: {exc}") from exc

        for prediction in predictions:
            names = getattr(prediction, "names", {}) or {}
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            for class_id, confidence in zip(
                boxes.cls.tolist() if hasattr(boxes, "cls") else [],
                boxes.conf.tolist() if hasattr(boxes, "conf") else [],
            ):
                label = str(names.get(int(class_id), int(class_id)))
                counts[label] = counts.get(label, 0) + 1
                confidences[label] = max(confidences.get(label, 0.0), float(confidence))

        result = AnalyzerResult(backend="ultralytics")
        result.payload = {
            "counts": counts,
            "confidences": {k: round(v, 3) for k, v in confidences.items()},
            "frames_analyzed": len(frames),
        }
        result.confidence = min(1.0, len(counts) / 5.0)

        for label, confidence in confidences.items():
            if label in _INTERESTING_OBJECTS and confidence >= 0.4:
                result.add_entity(EntityKind.OBJECT, label, confidence)
                result.add_tag(label, confidence)
        return result


@functools.lru_cache(maxsize=2)
def _yolo_model(weights: str) -> Any:
    from ultralytics import YOLO

    log.info("loading object detection model %s", weights)
    return YOLO(weights)


# ---------------------------------------------------------------------------
# Scene character
# ---------------------------------------------------------------------------


@register
class SceneAnalyzer(Analyzer):
    """Characterize the *look* of a clip: pace, palette, animation, letterbox.

    These cheap statistics are surprisingly discriminative for this material.
    A 30-second spot has a fast cut rate and saturated colours; a news break is
    slow and desaturated; animation has flat colour histograms and hard edges.
    """

    name = "scene"
    version = "1"
    requires = ("opencv", "numpy")
    order = 45
    needs_frames = True
    description = "Cut rate, palette, motion, animation and letterbox detection"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return len(ctx.frames) >= 2

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        cv2 = _load_cv2()
        import numpy as np

        if cv2 is None:  # pragma: no cover
            raise AnalyzerError(self.name, "opencv unavailable")

        frames = sample_frames(ctx.frames, SCENE_MAX_FRAMES)
        images = [cv2.imread(str(f)) for f in frames]
        images = [img for img in images if img is not None]
        if len(images) < 2:
            raise AnalyzerError(self.name, "not enough readable frames")

        brightness: list[float] = []
        saturation: list[float] = []
        edge_density: list[float] = []
        colour_spread: list[float] = []
        differences: list[float] = []
        previous_gray = None

        for image in images:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(hsv[:, :, 2])) / 255.0)
            saturation.append(float(np.mean(hsv[:, :, 1])) / 255.0)
            edges = cv2.Canny(gray, 80, 200)
            edge_density.append(float(np.count_nonzero(edges)) / edges.size)
            histogram = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            colour_spread.append(_entropy(np, histogram))
            if previous_gray is not None:
                resized = cv2.resize(gray, (previous_gray.shape[1], previous_gray.shape[0]))
                differences.append(float(np.mean(cv2.absdiff(previous_gray, resized))) / 255.0)
            previous_gray = gray

        cut_rate = _cut_rate(differences, ctx.clip.duration_s)
        letterboxed = _letterboxed(np, cv2, images[len(images) // 2])
        animation_score = _animation_score(
            mean(edge_density), mean(colour_spread), mean(saturation)
        )

        result = AnalyzerResult(backend="opencv")
        result.payload = {
            "brightness": round(mean(brightness), 4),
            "saturation": round(mean(saturation), 4),
            "edge_density": round(mean(edge_density), 4),
            "colour_entropy": round(mean(colour_spread), 4),
            "motion": round(mean(differences) if differences else 0.0, 4),
            "cut_rate_per_minute": round(cut_rate, 2),
            "letterboxed": letterboxed,
            "animation_score": round(animation_score, 3),
        }
        result.confidence = 0.6

        if letterboxed:
            result.add_tag("letterboxed", 0.8)
            result.vote(Category.MOVIE, 0.2, "letterboxed picture")
        if animation_score >= 0.6:
            result.add_tag("animation", animation_score)
        if cut_rate >= 30:
            result.add_tag("fast-cut", 0.7)
            # Rapid cutting is the house style of both spots and music videos.
            result.vote(Category.COMMERCIAL, 0.2, f"fast cutting ({cut_rate:.0f}/min)")
            result.vote(Category.MUSIC_VIDEO, 0.15, f"fast cutting ({cut_rate:.0f}/min)")
        elif cut_rate <= 6:
            result.add_tag("slow-cut", 0.6)
        if mean(saturation) < 0.18:
            result.add_tag("desaturated", 0.6)
        return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _entropy(np: Any, histogram: Any) -> float:
    """Shannon entropy of a normalized histogram, in bits."""
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _cut_rate(differences: list[float], duration_s: float) -> float:
    """Estimated shot changes per minute.

    Frames are sampled sparsely, so this cannot count cuts directly. What it
    measures is the *fraction* of sampled transitions that look like cuts, then
    scales that fraction by how far apart the samples were. It is an estimate
    and is treated as weak evidence everywhere it is used.
    """
    if not differences or duration_s <= 0:
        return 0.0
    threshold = max(0.12, mean(differences) * 1.8)
    cut_ratio = sum(1 for d in differences if d >= threshold) / len(differences)
    spacing_s = duration_s / (len(differences) + 1)
    if spacing_s <= 0:
        return 0.0
    return cut_ratio * (60.0 / spacing_s)


def _letterboxed(np: Any, cv2: Any, image: Any) -> bool:
    """True when the top and bottom of the frame are matte black bars."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    band = max(2, height // 12)
    top = float(np.mean(gray[:band]))
    bottom = float(np.mean(gray[-band:]))
    middle = float(np.mean(gray[band:-band])) if height > 2 * band else 255.0
    return top < 16 and bottom < 16 and middle > 30


def _animation_score(edge_density: float, colour_entropy: float, saturation: float) -> float:
    """Heuristic 0-1 score for cel/computer animation.

    Animation combines hard, clean edges with a small, saturated palette --
    the opposite of film grain, which has soft edges and a broad palette.
    """
    edge_component = min(1.0, edge_density / 0.12)
    palette_component = max(0.0, 1.0 - colour_entropy / 4.5)
    saturation_component = min(1.0, saturation / 0.45)
    return round(
        0.4 * edge_component + 0.35 * palette_component + 0.25 * saturation_component, 3
    )


def _frame_time(ctx: AnalysisContext, index: int, total: int) -> float:
    if total <= 1:
        return ctx.clip.midpoint_s
    return ctx.clip.duration_s * (index / (total - 1))
