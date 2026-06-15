"""Estrategia de conteo por campo de visión (FOV).

Acumula IDs de tracker únicos a lo largo de toda la sesión.

Corrección del bug original: el conjunto ``_seen_ids`` NUNCA se borra
durante la operación normal. El reset periódico del tracker recreaba
el tracker y limpiaba este conjunto, causando que las personas que
seguían en cuadro recibieran IDs nuevos y fueran contadas dos veces.

La limpieza de estado stale (``purge_stale_tracks``) es intencionalmente
un no-op: los IDs ya vistos deben permanecer en memoria para que
la misma persona no se cuente nuevamente si el tracker le asigna
un ID nuevo por pérdida de track.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from core.counting_strategy import CountingStrategy


class FovStrategy(CountingStrategy):
    """Cuenta personas únicas detectadas en cualquier momento de la sesión."""

    def __init__(self) -> None:
        self._seen_ids: set[int] = set()

    # ------------------------------------------------------------------
    # CountingStrategy interface
    # ------------------------------------------------------------------

    def update(self, detections: sv.Detections) -> None:
        if detections.tracker_id is not None:
            self._seen_ids.update(int(tid) for tid in detections.tracker_id)

    def get_counts(self) -> dict:
        total = len(self._seen_ids)
        return {"in_count": total, "out_count": 0, "fov_count": total}

    def reset(self) -> None:
        self._seen_ids.clear()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        # FOV no tiene zona visual propia; los contadores se dibujan
        # en el overlay global de PersonCounter.
        return frame

    def purge_stale_tracks(self, stale_ids: set[int]) -> None:
        # Intencionalmente vacío: borrar IDs vistos causaría doble conteo
        # si el tracker reasigna un nuevo ID a la misma persona física.
        pass
