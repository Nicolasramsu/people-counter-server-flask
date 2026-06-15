"""Limpieza periódica de estado interno para tracks inactivos.

Reemplaza el antiguo reset periódico del tracker (que causaba doble
conteo al recrear el tracker y asignar nuevos IDs). En lugar de
destruir el tracker, solo purga los diccionarios de estado de la
estrategia activa para tracks que llevan demasiado tiempo sin aparecer.
"""
from __future__ import annotations

import logging
import time

import supervision as sv

from config import STALE_TRACK_CLEANUP_INTERVAL_S, STALE_TRACK_THRESHOLD_S
from core.counting_strategy import CountingStrategy

logger = logging.getLogger("ContadorPersonas")


class StaleTrackCleaner:
    """Registra la última aparición de cada track y purga los obsoletos."""

    def __init__(self) -> None:
        self._last_seen: dict[int, float] = {}
        self._last_cleanup: float = time.monotonic()

    def update(self, detections: sv.Detections) -> None:
        """Actualiza el timestamp de los tracks visibles en el frame actual."""
        if detections.tracker_id is None:
            return
        now = time.monotonic()
        for tid in detections.tracker_id:
            self._last_seen[int(tid)] = now

    def maybe_purge(self, strategy: CountingStrategy | None) -> None:
        """Si pasó el intervalo, purga tracks stale de la estrategia activa."""
        now = time.monotonic()
        if (now - self._last_cleanup) < STALE_TRACK_CLEANUP_INTERVAL_S:
            return

        stale = {
            tid for tid, last in self._last_seen.items()
            if (now - last) > STALE_TRACK_THRESHOLD_S
        }
        if stale:
            if strategy is not None:
                strategy.purge_stale_tracks(stale)
            for tid in stale:
                self._last_seen.pop(tid, None)
            logger.debug("Tracks stale purgados: %d.", len(stale))

        self._last_cleanup = now

    def clear(self) -> None:
        self._last_seen.clear()
        self._last_cleanup = time.monotonic()
