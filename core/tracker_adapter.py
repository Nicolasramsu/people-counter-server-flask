"""Adaptador de tracker unificado para ByteTrack y BoxMOT.

Expone una única interfaz:
    adapter.update(detections, frame) -> sv.Detections

Internamente delega a ``sv.ByteTrack`` o a un tracker BoxMOT según
``TRACKER_BACKEND`` en config.py. Esto permite cambiar de backend sin
modificar ``PersonCounter`` ni las estrategias de conteo.

Cuando TRACKER_BACKEND="boxmot":
  - El tracker extrae features de apariencia del frame para Re-ID.
  - El modelo Re-ID se carga con boxmot.reid.ReID y se pasa al tracker.
  - Si los pesos no existen en ~/.cache/boxmot/, BoxMOT los descarga.
  - La salida se convierte a sv.Detections para que el resto del pipeline
    no necesite saber qué tracker está corriendo.

Compatibilidad: BoxMOT >= 19.0 (API con boxmot.trackers y boxmot.reid.ReID).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import supervision as sv

from config import (
    BOXMOT_ALGORITHM,
    INFERENCE_DEVICE,
    REID_MODEL,
    TRACKER_BACKEND,
    TRACKER_MATCHING_THRESHOLD,
)

logger = logging.getLogger("ContadorPersonas")

_REID_CACHE = Path.home() / ".cache" / "boxmot"


def _empty_detections() -> sv.Detections:
    """sv.Detections vacío con tracker_id explícito (TraceAnnotator lo requiere)."""
    d = sv.Detections.empty()
    d.tracker_id = np.array([], dtype=int)
    return d

# Algoritmos con Re-ID (requieren cargar un modelo de apariencia)
_REID_ALGORITHMS = {"botsort", "strongsort", "deepocsort"}


class TrackerAdapter:
    """Envuelve sv.ByteTrack o un tracker BoxMOT con interfaz homogénea."""

    def __init__(self, confidence: float) -> None:
        self._confidence = confidence
        self._tracker = self._build()
        logger.info(
            "Tracker inicializado: %s%s",
            TRACKER_BACKEND,
            f" / {BOXMOT_ALGORITHM}" if TRACKER_BACKEND == "boxmot" else "",
        )

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Actualiza el tracker y retorna detecciones con tracker_id asignado."""
        if TRACKER_BACKEND == "boxmot":
            return self._update_boxmot(detections, frame)
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Recrea el tracker desde cero (llamado en reset_counters)."""
        self._tracker = self._build()

    # ------------------------------------------------------------------
    # Construcción interna
    # ------------------------------------------------------------------

    def _build(self):
        if TRACKER_BACKEND == "boxmot":
            return self._build_boxmot()
        return sv.ByteTrack(
            track_activation_threshold=self._confidence,
            minimum_matching_threshold=TRACKER_MATCHING_THRESHOLD,
            frame_rate=30,
        )

    def _build_boxmot(self):
        try:
            from boxmot.trackers import (
                BotSort, DeepOcSort, OcSort, StrongSort,
            )
            from boxmot.trackers import ByteTrack as BoxmotByteTrack
        except ImportError:
            raise ImportError(
                "TRACKER_BACKEND='boxmot' requiere el paquete boxmot.\n"
                "Instálalo con:  pip install boxmot"
            )

        algo = BOXMOT_ALGORITHM.lower()
        device = (
            f"cuda:{INFERENCE_DEVICE}"
            if isinstance(INFERENCE_DEVICE, int)
            else str(INFERENCE_DEVICE)
        )
        half = isinstance(INFERENCE_DEVICE, int)  # FP16 solo con GPU

        tracker_map = {
            "botsort":    BotSort,
            "strongsort": StrongSort,
            "deepocsort": DeepOcSort,
            "ocsort":     OcSort,
            "bytetrack":  BoxmotByteTrack,
        }
        cls = tracker_map.get(algo)
        if cls is None:
            raise ValueError(
                f"Algoritmo BoxMOT desconocido: '{algo}'. "
                f"Válidos: {list(tracker_map)}"
            )

        if algo in _REID_ALGORITHMS:
            from boxmot.reid import ReID
            _REID_CACHE.mkdir(parents=True, exist_ok=True)
            reid_wrapper = ReID(
                path=_REID_CACHE / REID_MODEL,
                device=device,
                half=half,
            )
            # BotSort/StrongSort/DeepOcSort llaman get_features() directamente
            # sobre el modelo, que vive en reid_wrapper.model (PyTorchBackend).
            return cls(reid_model=reid_wrapper.model)

        # OcSort y ByteTrack: sin Re-ID, sin GPU
        return cls()

    # ------------------------------------------------------------------
    # Conversión BoxMOT ↔ supervision
    # ------------------------------------------------------------------

    def _update_boxmot(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Convierte a formato BoxMOT, actualiza, convierte de vuelta a sv.Detections."""
        conf = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        cls = (
            detections.class_id
            if detections.class_id is not None
            else np.zeros(len(detections), dtype=np.int32)
        )

        if len(detections) == 0:
            self._tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return _empty_detections()

        # BoxMOT espera: [x1, y1, x2, y2, conf, cls]
        dets = np.column_stack([detections.xyxy, conf, cls]).astype(np.float32)
        tracks = self._tracker.update(dets, frame)

        # BoxMOT retorna: [x1, y1, x2, y2, track_id, conf, cls, ind]
        if tracks is None or len(tracks) == 0:
            return _empty_detections()

        return sv.Detections(
            xyxy=tracks[:, :4].astype(np.float32),
            confidence=tracks[:, 5].astype(np.float32),
            class_id=tracks[:, 6].astype(int),
            tracker_id=tracks[:, 4].astype(int),
        )
