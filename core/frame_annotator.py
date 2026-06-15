"""Anotaciones visuales sobre frames de video.

Encapsula los tres anotadores de supervision (trace, box, label) y el
dibujado de puntos centrales. No contiene lógica de conteo: solo dibuja
lo que ``PersonCounter`` y la estrategia activa le indican.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from core.counting_strategy import CountingStrategy


class FrameAnnotator:
    """Superpone bounding boxes, trazas, etiquetas y overlay de estrategia."""

    def __init__(self) -> None:
        self._box   = sv.BoxAnnotator(thickness=2)
        self._label = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self._trace = sv.TraceAnnotator(thickness=2, trace_length=60)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str],
        strategy: CountingStrategy | None = None,
    ) -> np.ndarray:
        """Retorna el frame con todas las anotaciones aplicadas."""
        frame = self._trace.annotate(scene=frame, detections=detections)
        frame = self._box.annotate(scene=frame, detections=detections)
        if labels:
            frame = self._label.annotate(scene=frame, detections=detections, labels=labels)
        if strategy is not None:
            frame = strategy.draw_overlay(frame)
        self._draw_centers(frame, detections)
        return frame

    # ------------------------------------------------------------------

    def _draw_centers(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0 or detections.xyxy is None:
            return
        for box in detections.xyxy:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 7, (0, 180, 180), 1)
