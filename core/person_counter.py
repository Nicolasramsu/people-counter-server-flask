"""Orquestador de detección, tracking y conteo de personas.

Coordina cuatro capas:
  1. Detección     — YOLO ejecuta la inferencia sobre cada frame.
  2. Tracking      — ByteTrack asigna IDs persistentes a cada detección.
  3. Conteo        — La estrategia activa (CountingStrategy) decide qué cuenta.
  4. Presentación  — FrameAnnotator superpone las anotaciones visuales.

Las responsabilidades auxiliares viven en módulos dedicados dentro de ``core/``:
  strategies/           → lógica de conteo por modo
  depth_filter.py       → filtrado por distancia (solo con OAK-D W)
  frame_annotator.py    → anotaciones visuales
  interval_stats.py     → estadísticas por período y exportación CSV
  stale_track_cleaner.py → limpieza de tracks inactivos
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    AREA_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_INFER_SIZE,
    DEFAULT_LINE_POSITION_H,
    DEFAULT_LINE_POSITION_V,
    DEFAULT_MODEL,
    DEFAULT_ROI,
    INFERENCE_DEVICE,
    PERSON_CLASS_ID,
    TRACKER_MATCHING_THRESHOLD,
)
from core.counting_strategy import CountingStrategy
from core.depth_filter import DepthFilter
from core.frame_annotator import FrameAnnotator
from core.interval_stats import IntervalStats
from core.stale_track_cleaner import StaleTrackCleaner
from core.strategies import FovStrategy, LineStrategy, PolygonStrategy

logger = logging.getLogger("ContadorPersonas")


class PersonCounter:
    """Orquestador de detección, tracking y conteo de personas."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        line_position: float = DEFAULT_LINE_POSITION_H,
        infer_size: int = DEFAULT_INFER_SIZE,
    ) -> None:
        self.model_name    = model_name
        self.confidence    = confidence
        self.line_position = line_position
        self.line_position_vertical = DEFAULT_LINE_POSITION_V
        self.infer_size    = infer_size

        self.use_horizontal_line = True
        self.use_vertical_line   = False
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = DEFAULT_ROI
        self.counting_mode = "line"
        self.frame_width   = 640
        self.frame_height  = 480

        self.persons_in_frame = 0
        self.fps = 0.0
        self._fps_buffer: deque = deque(maxlen=30)

        self.model:     YOLO | None            = None
        self.tracker:   sv.ByteTrack | None    = None
        self._strategy: CountingStrategy | None = None

        self._stats     = IntervalStats()
        self._annotator = FrameAnnotator()
        self._depth     = DepthFilter()
        self._stale     = StaleTrackCleaner()

    # ------------------------------------------------------------------
    # Propiedades de conteo
    # ------------------------------------------------------------------

    @property
    def in_count(self) -> int:
        return self._strategy.get_counts().get("in_count", 0) if self._strategy else 0

    @property
    def out_count(self) -> int:
        return self._strategy.get_counts().get("out_count", 0) if self._strategy else 0

    @property
    def fov_count(self) -> int:
        return self._strategy.get_counts().get("fov_count", 0) if self._strategy else 0

    @property
    def interval_data(self) -> list:
        return self._stats.interval_data

    @property
    def hourly_entries(self) -> dict:
        return self._stats.hourly_entries

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        logger.info("Cargando modelo YOLO: %s (device=%s)", self.model_name, INFERENCE_DEVICE)
        self.model = YOLO(self.model_name)
        self.model.to(INFERENCE_DEVICE)
        logger.info("Modelo cargado correctamente.")

    def setup_line(self, frame_width: int, frame_height: int) -> None:
        """Configura estrategia y tracker. Llamar una vez al abrir la cámara."""
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self._strategy = self._build_strategy()
        self._create_tracker()
        self._stats.start()

    # ------------------------------------------------------------------
    # Reconfiguración en caliente
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self.counting_mode = mode
        if self.model is not None:
            self._strategy = self._build_strategy()
        logger.info("Modo de conteo cambiado a: %s", mode)

    def set_roi(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.roi_x1, self.roi_y1 = min(x1, x2), min(y1, y2)
        self.roi_x2, self.roi_y2 = max(x1, x2), max(y1, y2)
        if isinstance(self._strategy, PolygonStrategy):
            self._strategy.set_from_normalized_rect(
                self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2
            )
        logger.info(
            "ROI: (%.2f,%.2f)→(%.2f,%.2f)",
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2,
        )

    def update_line(self, position: float, position_vertical: float | None = None) -> None:
        self.line_position = position
        if position_vertical is not None:
            self.line_position_vertical = position_vertical
        if isinstance(self._strategy, LineStrategy):
            self._strategy.update_config(
                position_h=self.line_position,
                position_v=self.line_position_vertical,
            )

    def set_line_enabled(
        self, horizontal: bool | None = None, vertical: bool | None = None
    ) -> None:
        if horizontal is not None:
            self.use_horizontal_line = horizontal
        if vertical is not None:
            self.use_vertical_line = vertical
        if isinstance(self._strategy, LineStrategy):
            self._strategy.update_config(
                use_horizontal=self.use_horizontal_line,
                use_vertical=self.use_vertical_line,
            )

    def update_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Procesamiento de frames
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, depth_frame: np.ndarray | None = None
    ) -> np.ndarray:
        dfs = [depth_frame] if depth_frame is not None else None
        return self.process_batch([frame], depth_frames=dfs)[0]

    def process_batch(self, frames: list, depth_frames: list | None = None) -> list:
        """Inferencia YOLO + tracking + conteo + anotación sobre un lote de frames."""
        t_start = time.perf_counter()
        if self.model is None:
            return frames

        self._stale.maybe_purge(self._strategy)

        results_list = self.model(
            frames,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            imgsz=self.infer_size,
            device=INFERENCE_DEVICE,
            verbose=False,
        )

        annotated = []
        for i, (frame, results) in enumerate(zip(frames, results_list)):
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            df = depth_frames[i] if depth_frames and i < len(depth_frames) else None
            if df is not None:
                detections = self._depth.apply(detections, df)

            self._stale.update(detections)
            if self._strategy is not None:
                self._strategy.update(detections)

            self.persons_in_frame = len(detections)
            self._stats.tick(self.in_count, self.out_count, self.fov_count, self.counting_mode)

            labels = self._build_labels(detections, df)
            annotated.append(self._annotator.annotate(frame, detections, labels, self._strategy))

        elapsed = time.perf_counter() - t_start
        self._fps_buffer.append(len(frames) / elapsed if elapsed > 0 else 0.0)
        self.fps = sum(self._fps_buffer) / len(self._fps_buffer)
        return annotated

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Reinicia todos los contadores y el tracker (acción explícita del usuario)."""
        if self._strategy is not None:
            self._strategy.reset()
        self.persons_in_frame = 0
        self._stats.reset()
        self._stale.clear()
        self._create_tracker()
        logger.info("Contadores reiniciados.")

    def export_csv(self, filepath: str) -> None:
        self._stats.export_csv(
            filepath, self.in_count, self.out_count, self.fov_count, self.counting_mode
        )

    # ------------------------------------------------------------------
    # Privados
    # ------------------------------------------------------------------

    def _create_tracker(self) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=TRACKER_MATCHING_THRESHOLD,
            frame_rate=30,
        )

    def _build_strategy(self) -> CountingStrategy:
        if self.counting_mode == "fov":
            return FovStrategy()
        if self.counting_mode == "roi":
            s = PolygonStrategy(self.frame_width, self.frame_height)
            s.set_from_normalized_rect(self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)
            return s
        return LineStrategy(
            self.frame_width, self.frame_height,
            line_position_h=self.line_position,
            line_position_v=self.line_position_vertical,
            use_horizontal=self.use_horizontal_line,
            use_vertical=self.use_vertical_line,
            area_threshold=AREA_THRESHOLD,
        )

    def _build_labels(
        self, detections: sv.Detections, depth_frame: np.ndarray | None
    ) -> list[str]:
        if detections.tracker_id is None:
            return []
        if depth_frame is not None:
            return [
                f"#{tid} {conf:.0%} {self._depth.label(box, depth_frame)}"
                for tid, conf, box in zip(
                    detections.tracker_id, detections.confidence, detections.xyxy
                )
            ]
        return [
            f"#{tid} {conf:.0%}"
            for tid, conf in zip(detections.tracker_id, detections.confidence)
        ]
