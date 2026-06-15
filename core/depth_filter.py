"""Filtro de profundidad para detecciones basado en mapa estéreo.

Recibe las detecciones del tracker y descarta las que tienen su centro
de bounding box fuera del rango de profundidad configurado.

Solo tiene efecto cuando se pasa un ``depth_frame`` (backend OAK-D W).
Con el backend OpenCV el argumento siempre es ``None`` y la clase
no interviene en el pipeline.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from config import MAX_DETECTION_DEPTH_M, MIN_DETECTION_DEPTH_M


class DepthFilter:
    """Filtra detecciones por rango de profundidad y genera etiquetas de distancia."""

    def __init__(
        self,
        min_m: float = MIN_DETECTION_DEPTH_M,
        max_m: float = MAX_DETECTION_DEPTH_M,
    ) -> None:
        self.min_m = min_m
        self.max_m = max_m

    def apply(self, detections: sv.Detections, depth_frame: np.ndarray) -> sv.Detections:
        """Retorna el subconjunto de detecciones dentro del rango válido."""
        if len(detections) == 0 or detections.xyxy is None:
            return detections
        fh, fw = depth_frame.shape[:2]
        mask = [self._in_range(box, depth_frame, fh, fw) for box in detections.xyxy]
        return detections[np.array(mask, dtype=bool)]

    def label(self, box: np.ndarray, depth_frame: np.ndarray) -> str:
        """Retorna la profundidad en el centro del bbox como string legible."""
        d_mm = self._sample(box, depth_frame)
        return f"{d_mm / 1000:.1f}m" if d_mm > 0 else "?"

    # ------------------------------------------------------------------

    def _in_range(
        self, box: np.ndarray, depth_frame: np.ndarray, fh: int, fw: int
    ) -> bool:
        d_mm = self._sample(box, depth_frame, fh, fw)
        if d_mm == 0:
            return False
        return self.min_m <= d_mm / 1000.0 <= self.max_m

    def _sample(
        self,
        box: np.ndarray,
        depth_frame: np.ndarray,
        fh: int | None = None,
        fw: int | None = None,
    ) -> int:
        """Profundidad (mm) en el centro del bounding box. 0 = medición inválida."""
        if fh is None or fw is None:
            fh, fw = depth_frame.shape[:2]
        cx = max(0, min(int((box[0] + box[2]) / 2), fw - 1))
        cy = max(0, min(int((box[1] + box[3]) / 2), fh - 1))
        return int(depth_frame[cy, cx])
