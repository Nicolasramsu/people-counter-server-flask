"""Estrategia de conteo por cruce de línea.

Mejoras respecto a la implementación original:

- Cooldown por track: una vez que un track cruza la línea, se ignoran
  cruces adicionales del mismo track durante ``cooldown_s`` segundos.
  Esto elimina el doble conteo por oscilación del bounding box.

- Limpieza de estado stale: ``purge_stale_tracks`` elimina entradas de
  tracks que desaparecieron hace tiempo, evitando crecimiento indefinido
  de los diccionarios internos sin recrear el tracker.

- Reconfiguración sin pérdida de conteo: ``update_config`` mueve la
  línea y limpia el estado de lado (que ya no es válido para la nueva
  posición), pero no toca los contadores acumulados.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import supervision as sv

from config import AREA_THRESHOLD, LINE_CROSSING_COOLDOWN_S
from core.counting_strategy import CountingStrategy


class LineStrategy(CountingStrategy):
    """Cuenta personas que cruzan una línea horizontal y/o vertical."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        *,
        line_position_h: float = 0.5,
        line_position_v: float = 0.5,
        use_horizontal: bool = True,
        use_vertical: bool = False,
        area_threshold: float = AREA_THRESHOLD,
        cooldown_s: float = LINE_CROSSING_COOLDOWN_S,
    ) -> None:
        self._fw = frame_width
        self._fh = frame_height
        self.line_position_h = line_position_h
        self.line_position_v = line_position_v
        self.use_horizontal = use_horizontal
        self.use_vertical = use_vertical
        self._area_threshold = area_threshold
        self._cooldown_s = cooldown_s

        self.in_count = 0
        self.out_count = 0

        # Estado de lado por track. Se invalida al mover la línea.
        self._side_h: dict[int, str] = {}    # tid → "up" | "down"
        self._side_v: dict[int, str] = {}    # tid → "left" | "right"
        # Timestamp del último cruce registrado por track.
        self._last_cross: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Reconfiguración en caliente (sin pérdida de conteo)
    # ------------------------------------------------------------------

    def update_config(
        self,
        position_h: float | None = None,
        position_v: float | None = None,
        use_horizontal: bool | None = None,
        use_vertical: bool | None = None,
    ) -> None:
        """Actualiza parámetros de la línea sin resetear los contadores."""
        if position_h is not None:
            self.line_position_h = position_h
            self._side_h.clear()   # El lado anterior ya no es válido
        if position_v is not None:
            self.line_position_v = position_v
            self._side_v.clear()
        if use_horizontal is not None:
            self.use_horizontal = use_horizontal
        if use_vertical is not None:
            self.use_vertical = use_vertical

    # ------------------------------------------------------------------
    # CountingStrategy interface
    # ------------------------------------------------------------------

    def update(self, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return
        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0:
                continue
            if self.use_horizontal:
                self._check_horizontal(int(tid), x1, y1, x2, y2, bbox_area)
            if self.use_vertical:
                self._check_vertical(int(tid), x1, y1, x2, y2, bbox_area)

    def get_counts(self) -> dict:
        return {"in_count": self.in_count, "out_count": self.out_count}

    def reset(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self._side_h.clear()
        self._side_v.clear()
        self._last_cross.clear()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.use_horizontal:
            y = int(self._fh * self.line_position_h)
            cv2.line(frame, (0, y), (self._fw, y), (0, 255, 255), 3)
            cv2.putText(
                frame,
                f"IN: {self.in_count}  OUT: {self.out_count}",
                (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        if self.use_vertical:
            x = int(self._fw * self.line_position_v)
            cv2.line(frame, (x, 0), (x, self._fh), (255, 0, 255), 3)
            if not self.use_horizontal:
                cv2.putText(
                    frame,
                    f"IN: {self.in_count}  OUT: {self.out_count}",
                    (x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
                )
        return frame

    def purge_stale_tracks(self, stale_ids: set[int]) -> None:
        for tid in stale_ids:
            self._side_h.pop(tid, None)
            self._side_v.pop(tid, None)
            self._last_cross.pop(tid, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _in_cooldown(self, tid: int) -> bool:
        last = self._last_cross.get(tid)
        return last is not None and (time.monotonic() - last) < self._cooldown_s

    def _check_horizontal(
        self, tid: int, x1: float, y1: float, x2: float, y2: float, bbox_area: float
    ) -> None:
        y_line = int(self._fh * self.line_position_h)
        current_side = self._resolve_h_side(tid, x1, y1, x2, y2, y_line, bbox_area)
        prev_side = self._side_h.get(tid)

        if current_side is not None:
            if prev_side is not None and prev_side != current_side and not self._in_cooldown(tid):
                if current_side == "down":
                    self.in_count += 1
                else:
                    self.out_count += 1
                self._last_cross[tid] = time.monotonic()
            self._side_h[tid] = current_side

    def _check_vertical(
        self, tid: int, x1: float, y1: float, x2: float, y2: float, bbox_area: float
    ) -> None:
        x_line = int(self._fw * self.line_position_v)
        current_side = self._resolve_v_side(tid, x1, y1, x2, y2, x_line, bbox_area)
        prev_side = self._side_v.get(tid)

        if current_side is not None:
            if prev_side is not None and prev_side != current_side and not self._in_cooldown(tid):
                if current_side == "right":
                    self.in_count += 1
                else:
                    self.out_count += 1
                self._last_cross[tid] = time.monotonic()
            self._side_v[tid] = current_side

    def _resolve_h_side(
        self,
        tid: int, x1: float, y1: float, x2: float, y2: float,
        y_line: int, bbox_area: float,
    ) -> str | None:
        if y2 < y_line:
            return "up"
        if y1 > y_line:
            return "down"
        area_up   = (x2 - x1) * (y_line - y1) if y1 < y_line else 0.0
        area_down = (x2 - x1) * (y2 - y_line) if y2 > y_line else 0.0
        if area_up / bbox_area >= self._area_threshold:
            return "up"
        if area_down / bbox_area >= self._area_threshold:
            return "down"
        return self._side_h.get(tid)   # Ambiguo: mantener lado anterior

    def _resolve_v_side(
        self,
        tid: int, x1: float, y1: float, x2: float, y2: float,
        x_line: int, bbox_area: float,
    ) -> str | None:
        if x2 < x_line:
            return "left"
        if x1 > x_line:
            return "right"
        area_left  = (y2 - y1) * (x_line - x1) if x1 < x_line else 0.0
        area_right = (y2 - y1) * (x2 - x_line) if x2 > x_line else 0.0
        if area_left / bbox_area >= self._area_threshold:
            return "left"
        if area_right / bbox_area >= self._area_threshold:
            return "right"
        return self._side_v.get(tid)   # Ambiguo: mantener lado anterior
