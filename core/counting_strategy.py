"""Contrato abstracto para estrategias de conteo de personas.

Define la interfaz que todas las estrategias de conteo deben cumplir.
Aplicación del principio Open/Closed: nuevos modos se agregan creando
una subclase, sin modificar el orquestador (PersonCounter).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import supervision as sv


class CountingStrategy(ABC):
    """Interfaz que toda estrategia de conteo debe satisfacer.

    PersonCounter depende únicamente de esta abstracción, nunca de
    implementaciones concretas (Dependency Inversion Principle).
    """

    @abstractmethod
    def update(self, detections: sv.Detections) -> None:
        """Procesa las detecciones del frame actual y actualiza el estado interno.

        Args:
            detections: Detecciones con tracker_id asignado por ByteTrack.
        """

    @abstractmethod
    def get_counts(self) -> dict:
        """Retorna el estado de conteo actual.

        Returns:
            Diccionario con al menos las claves ``in_count`` y ``out_count``.
            FovStrategy agrega ``fov_count``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reinicia todos los contadores y el estado interno de tracking."""

    @abstractmethod
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja los elementos visuales del modo sobre el frame.

        Args:
            frame: Imagen BGR sobre la que se dibuja.

        Returns:
            El mismo frame con los overlays aplicados.
        """

    def purge_stale_tracks(self, stale_ids: set[int]) -> None:
        """Elimina del estado interno las entradas de tracks inactivos.

        Llamado periódicamente por PersonCounter para evitar el crecimiento
        indefinido de los diccionarios internos. No afecta los contadores
        acumulados, solo la memoria de estado por track.

        La implementación por defecto es no-op. Las subclases sobreescriben
        si mantienen estructuras indexadas por tracker_id.

        Args:
            stale_ids: IDs que no han aparecido en los últimos
                       ``STALE_TRACK_THRESHOLD_S`` segundos.
        """
