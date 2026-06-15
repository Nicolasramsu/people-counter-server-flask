"""Estrategia de conteo por zona poligonal.

Reemplaza la lógica de ROI rectangular con ``sv.PolygonZone``, que
soporta polígonos arbitrarios de N vértices.  El polígono por defecto
es el frame completo, y puede configurarse como rectángulo mediante
``set_from_normalized_rect`` (compatible con la API existente) o como
un polígono libre con ``set_from_pixel_points``.

Dos mecanismos de protección contra doble conteo:

1. **Dwell time**: una persona solo se contabiliza si permanece dentro
   del polígono al menos ``dwell_time_s`` segundos continuos.  Filtra
   personas que simplemente cruzan el área sin detenerse.

2. **Cooldown de re-entrada**: si la misma persona (mismo tracker_id)
   sale y vuelve a entrar dentro de ``reentry_cooldown_s`` segundos,
   no se vuelve a contar.  Evita contar múltiples veces a alguien que
   sale momentáneamente y regresa al stand.

La detección de "dentro" se evalúa en el punto BOTTOM_CENTER del
bounding box (pie de la persona), que es el punto más estable y
físicamente significativo para determinar si alguien está en el área.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import supervision as sv

from config import POLYGON_DWELL_TIME_S, POLYGON_REENTRY_COOLDOWN_S
from core.counting_strategy import CountingStrategy


class PolygonStrategy(CountingStrategy):
    """Cuenta personas dentro de una zona poligonal arbitraria."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        polygon: np.ndarray | None = None,
        *,
        dwell_time_s: float = POLYGON_DWELL_TIME_S,
        reentry_cooldown_s: float = POLYGON_REENTRY_COOLDOWN_S,
    ) -> None:
        """
        Args:
            frame_width:       Ancho del frame en píxeles.
            frame_height:      Alto del frame en píxeles.
            polygon:           Array (N, 2) de coordenadas absolutas en píxeles.
                               Si es None, se usa el frame completo.
            dwell_time_s:      Segundos mínimos dentro del polígono para contar.
            reentry_cooldown_s: Segundos de espera antes de recontar a la misma persona.
        """
        self._fw = frame_width
        self._fh = frame_height
        self._dwell_s = dwell_time_s
        self._reentry_s = reentry_cooldown_s

        self.in_count = 0
        self.out_count = 0

        # {tid: timestamp de cuando entró al polígono en la visita actual}
        self._entry_time: dict[int, float] = {}
        # {tid: True si in_count fue incrementado en la visita actual}
        self._visit_counted: dict[int, bool] = {}
        # {tid: timestamp de la última vez que se incrementó in_count}
        self._last_counted: dict[int, float] = {}

        pts = polygon if polygon is not None else self._full_frame_polygon()
        self._polygon = pts
        self._zone = self._build_zone(pts)

    # ------------------------------------------------------------------
    # Configuración del polígono
    # ------------------------------------------------------------------

    def set_from_normalized_rect(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        """Actualiza el polígono desde coordenadas normalizadas (0–1).

        Compatible con la API existente de ``set_roi``.
        """
        pts = np.array([
            [int(self._fw * x1), int(self._fh * y1)],
            [int(self._fw * x2), int(self._fh * y1)],
            [int(self._fw * x2), int(self._fh * y2)],
            [int(self._fw * x1), int(self._fh * y2)],
        ], dtype=np.int32)
        self._apply_polygon(pts)

    def set_from_pixel_points(self, points: list[list[int]]) -> None:
        """Actualiza el polígono desde coordenadas absolutas en píxeles.

        Permite definir polígonos arbitrarios (no solo rectángulos).
        Ejemplo: ``[[100, 50], [540, 50], [600, 400], [80, 400]]``
        """
        self._apply_polygon(np.array(points, dtype=np.int32))

    # ------------------------------------------------------------------
    # CountingStrategy interface
    # ------------------------------------------------------------------

    def update(self, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return

        now = time.monotonic()
        inside_mask: np.ndarray = self._zone.trigger(detections=detections)

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            is_inside = bool(inside_mask[i])

            if is_inside:
                self._handle_inside(tid, now)
            else:
                self._handle_outside(tid)

    def get_counts(self) -> dict:
        return {"in_count": self.in_count, "out_count": self.out_count}

    def reset(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self._entry_time.clear()
        self._visit_counted.clear()
        self._last_counted.clear()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._polygon], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self._polygon], isClosed=True, color=(0, 255, 0), thickness=3)

        # Etiqueta en el primer vértice del polígono
        x, y = int(self._polygon[0][0]), int(self._polygon[0][1])
        cv2.putText(
            frame,
            f"IN: {self.in_count}  OUT: {self.out_count}",
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        return frame

    def purge_stale_tracks(self, stale_ids: set[int]) -> None:
        now = time.monotonic()
        for tid in stale_ids:
            self._entry_time.pop(tid, None)
            self._visit_counted.pop(tid, None)
            # last_counted solo se purga si el cooldown ya expiró;
            # de lo contrario se necesita para prevenir re-entradas.
            last = self._last_counted.get(tid)
            if last is not None and (now - last) >= self._reentry_s:
                self._last_counted.pop(tid)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_inside(self, tid: int, now: float) -> None:
        if tid not in self._entry_time:
            self._entry_time[tid] = now
            self._visit_counted[tid] = False

        if self._visit_counted[tid]:
            return   # Ya fue contado en esta visita

        dwell = now - self._entry_time[tid]
        if dwell < self._dwell_s:
            return   # Aún no cumple el tiempo mínimo de permanencia

        # Verificar cooldown de re-entrada
        last = self._last_counted.get(tid)
        if last is None or (now - last) >= self._reentry_s:
            self.in_count += 1
            self._last_counted[tid] = now

        # Marcar visita procesada independientemente del cooldown,
        # para no seguir evaluando en cada frame mientras está dentro.
        self._visit_counted[tid] = True

    def _handle_outside(self, tid: int) -> None:
        if tid not in self._entry_time:
            return
        was_counted = self._visit_counted.get(tid, False)
        del self._entry_time[tid]
        del self._visit_counted[tid]
        if was_counted:
            self.out_count += 1   # Solo cuenta salida si hubo entrada válida

    def _apply_polygon(self, polygon: np.ndarray) -> None:
        self._polygon = polygon
        self._zone = self._build_zone(polygon)
        # Limpiar estado de tracks que podrían estar dentro del polígono anterior
        self._entry_time.clear()
        self._visit_counted.clear()

    def _build_zone(self, polygon: np.ndarray) -> sv.PolygonZone:
        return sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )

    def _full_frame_polygon(self) -> np.ndarray:
        return np.array([
            [0,        0       ],
            [self._fw, 0       ],
            [self._fw, self._fh],
            [0,        self._fh],
        ], dtype=np.int32)
