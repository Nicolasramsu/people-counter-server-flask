"""Agregación de estadísticas por intervalo de tiempo y exportación CSV.

Mantiene el historial de conteos por período (por defecto 15 minutos)
y los totales por hora del día. Gestiona también la escritura del CSV
al finalizar o al solicitar exportación manual.
"""
from __future__ import annotations

import csv
import logging
import time
from datetime import datetime

from config import INTERVAL_SECONDS

logger = logging.getLogger("ContadorPersonas")


class IntervalStats:
    """Acumula conteos por intervalo y hora; exporta a CSV."""

    def __init__(self) -> None:
        self.interval_data: list  = []
        self.hourly_entries: dict = {}
        self._last_time: float | None = None
        self._in_start  = 0
        self._out_start = 0
        self._fov_start = 0

    def start(self) -> None:
        """Registra el momento de inicio. Llamar al abrir la cámara."""
        if self._last_time is None:
            self._last_time = time.time()

    def tick(self, in_count: int, out_count: int, fov_count: int, mode: str) -> None:
        """Cierra el período actual si ha pasado INTERVAL_SECONDS."""
        if self._last_time is None:
            self._last_time = time.time()
            return
        now = time.time()
        if now - self._last_time < INTERVAL_SECONDS:
            return

        hora_str = datetime.now().strftime("%H:%M")
        hora     = datetime.now().hour

        if mode == "fov":
            delta = fov_count - self._fov_start
            self.interval_data.append((hora_str, delta, 0, fov_count, 0))
            self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + delta
            self._fov_start = fov_count
        else:
            in_d  = in_count  - self._in_start
            out_d = out_count - self._out_start
            self.interval_data.append((hora_str, in_d, out_d, in_count, out_count))
            self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + in_d
            self._in_start  = in_count
            self._out_start = out_count

        self._last_time = now

    def reset(self) -> None:
        self.interval_data.clear()
        self.hourly_entries.clear()
        self._last_time = time.time()
        self._in_start = self._out_start = self._fov_start = 0

    def export_csv(
        self, filepath: str, in_count: int, out_count: int, fov_count: int, mode: str
    ) -> None:
        """Escribe el historial completo (con el período parcial actual) a CSV."""
        data = list(self.interval_data)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if mode == "fov":
                partial = fov_count - self._fov_start
                if partial > 0:
                    data.append((datetime.now().strftime("%H:%M"), partial, 0, fov_count, 0))
                writer.writerow(["hora", "personas_intervalo", "personas_total"])
                for row in data:
                    writer.writerow([row[0], row[1], row[3]])
            else:
                in_p  = in_count  - self._in_start
                out_p = out_count - self._out_start
                if in_p > 0 or out_p > 0:
                    data.append((
                        datetime.now().strftime("%H:%M"),
                        in_p, out_p, in_count, out_count,
                    ))
                writer.writerow([
                    "hora", "entradas_intervalo", "salidas_intervalo",
                    "entradas_total", "salidas_total",
                ])
                writer.writerows(data)

        logger.info("CSV exportado: %s (%d filas, modo=%s)", filepath, len(data), mode)
