"""Orquestador de detección, tracking y conteo de personas.

Responsabilidades de esta clase (Single Responsibility por capa):
  - Cargar y ejecutar el modelo YOLO.
  - Gestionar el tracker ByteTrack.
  - Delegar la lógica de conteo a la estrategia activa (CountingStrategy).
  - Calcular estadísticas de intervalo y exportar CSV.
  - Proveer la interfaz pública que usa el servidor Flask.

La lógica de conteo específica de cada modo vive en:
  ``core/strategies/line_strategy.py``
  ``core/strategies/polygon_strategy.py``
  ``core/strategies/fov_strategy.py``
"""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    AREA_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_INFER_SIZE,
    DEFAULT_LINE_POSITION_H,
    DEFAULT_LINE_POSITION_V,
    DEFAULT_ROI,
    INFERENCE_DEVICE,
    INTERVAL_SECONDS,
    MAX_DETECTION_DEPTH_M,
    MIN_DETECTION_DEPTH_M,
    PERSON_CLASS_ID,
    STALE_TRACK_CLEANUP_INTERVAL_S,
    STALE_TRACK_THRESHOLD_S,
    TRACKER_MATCHING_THRESHOLD,
    DEFAULT_MODEL,
)
from core.counting_strategy import CountingStrategy
from core.strategies import FovStrategy, LineStrategy, PolygonStrategy

logger = logging.getLogger("ContadorPersonas")


class PersonCounter:
    """Orquestador de detección, tracking y conteo de personas."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        line_position: float = DEFAULT_LINE_POSITION_H,
        line_orientation: str = "horizontal",
        infer_size: int = DEFAULT_INFER_SIZE,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.line_position = line_position
        self.line_position_vertical = DEFAULT_LINE_POSITION_V
        self.line_orientation = line_orientation
        self.infer_size = infer_size

        # Parámetros de línea
        self.use_horizontal_line = True
        self.use_vertical_line = False

        # Parámetros de ROI (coordenadas normalizadas 0–1)
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = DEFAULT_ROI

        # Modo activo: "line" | "roi" | "fov"
        self.counting_mode = "line"

        # Dimensiones del frame (se fijan en setup_line)
        self.frame_width = 640
        self.frame_height = 480

        # Métricas de frame
        self.persons_in_frame = 0
        self.fps = 0.0
        self._fps_buffer: deque = deque(maxlen=30)

        # Estadísticas por intervalo y hora
        self.interval_data: list = []
        self._last_interval_time: float | None = None
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0
        self.hourly_entries: dict = {}

        # Motor YOLO y tracker
        self.model: YOLO | None = None
        self.tracker: sv.ByteTrack | None = None

        # Estrategia de conteo activa (se instancia en setup_line)
        self._strategy: CountingStrategy | None = None

        # Control de limpieza de estado stale
        # Reemplaza al antiguo _periodic_tracker_reset que causaba doble conteo.
        self._track_last_seen: dict[int, float] = {}
        self._last_cleanup_time: float = time.monotonic()

        # Anotadores visuales (cajas, trazas, etiquetas)
        self.box_annotator   = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)

    # ------------------------------------------------------------------
    # Propiedades de conteo (delegan a la estrategia activa)
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

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Carga el modelo YOLO en el dispositivo configurado."""
        logger.info("Cargando modelo YOLO: %s (device=%s)", self.model_name, INFERENCE_DEVICE)
        self.model = YOLO(self.model_name)
        self.model.to(INFERENCE_DEVICE)
        logger.info("Modelo cargado correctamente.")

    def setup_line(self, frame_width: int, frame_height: int) -> None:
        """Configura la estrategia y el tracker para las dimensiones del frame.

        Debe llamarse una vez al abrir la fuente de video, antes del primer frame.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._strategy = self._build_strategy()
        self._create_tracker()
        if self._last_interval_time is None:
            self._last_interval_time = time.time()

    # ------------------------------------------------------------------
    # Reconfiguración en caliente
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Cambia el modo de conteo activo y construye la estrategia correspondiente."""
        self.counting_mode = mode
        if self.model is not None:   # setup_line ya fue llamado
            self._strategy = self._build_strategy()
        logger.info("Modo de conteo cambiado a: %s", mode)

    def set_roi(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Define la zona de interés en coordenadas normalizadas (0–1)."""
        self.roi_x1 = min(x1, x2)
        self.roi_y1 = min(y1, y2)
        self.roi_x2 = max(x1, x2)
        self.roi_y2 = max(y1, y2)
        if isinstance(self._strategy, PolygonStrategy):
            self._strategy.set_from_normalized_rect(
                self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2
            )
        logger.info(
            "ROI actualizado: (%.2f, %.2f) → (%.2f, %.2f)",
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2,
        )

    def update_line(
        self,
        position: float,
        orientation: str | None = None,
        position_vertical: float | None = None,
    ) -> None:
        """Actualiza la posición de las líneas de conteo."""
        self.line_position = position
        if position_vertical is not None:
            self.line_position_vertical = position_vertical
        if orientation is not None:
            self.line_orientation = orientation
        if isinstance(self._strategy, LineStrategy):
            self._strategy.update_config(
                position_h=self.line_position,
                position_v=self.line_position_vertical,
            )

    def set_line_enabled(
        self,
        horizontal: bool | None = None,
        vertical: bool | None = None,
    ) -> None:
        """Habilita o deshabilita líneas individuales."""
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
        """Procesa un frame individual. Wrapper de ``process_batch``.

        Args:
            frame:       Imagen BGR capturada por la cámara.
            depth_frame: Mapa de profundidad uint16 en milímetros, alineado al
                         frame de color píxel a píxel. Opcional: si es None
                         el filtro de profundidad no se aplica. Solo disponible
                         con backend OAK-D W.
        """
        depth_frames = [depth_frame] if depth_frame is not None else None
        return self.process_batch([frame], depth_frames=depth_frames)[0]

    def process_batch(
        self, frames: list, depth_frames: list | None = None
    ) -> list:
        """Procesa una lista de frames en una sola llamada YOLO (batch GPU).

        Args:
            frames:       Lista de imágenes BGR.
            depth_frames: Lista de mapas de profundidad uint16 (mm), uno por
                          frame. Puede ser None o tener menos elementos que
                          ``frames``; los frames sin depth correspondiente no
                          se filtran por profundidad.

        Returns:
            Lista de frames anotados en el mismo orden.
        """
        t_start = time.perf_counter()

        if self.model is None:
            return frames

        self._maybe_cleanup_stale_tracks()

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

            # Filtrar por profundidad DESPUÉS del tracker (para no romper
            # la continuidad de tracks) y ANTES de la estrategia (para no
            # contar personas en el fondo del escenario).
            df: np.ndarray | None = (
                depth_frames[i]
                if depth_frames is not None and i < len(depth_frames)
                else None
            )
            if df is not None:
                detections = self._filter_by_depth(detections, df)

            self._update_track_activity(detections)

            if self._strategy is not None:
                self._strategy.update(detections)

            self.persons_in_frame = len(detections)
            self._check_interval()

            annotated.append(self._annotate_frame(frame, detections, df))

        elapsed = time.perf_counter() - t_start
        fps_batch = len(frames) / elapsed if elapsed > 0 else 0.0
        self._fps_buffer.append(fps_batch)
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
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0
        self._last_interval_time = time.time()
        self.interval_data.clear()
        self.hourly_entries.clear()
        self._track_last_seen.clear()
        # Recrear el tracker solo aquí (reset explícito del usuario),
        # nunca de forma automática para no generar IDs duplicados.
        self._create_tracker()
        logger.info("Contadores reiniciados.")

    def export_csv(self, filepath: str) -> None:
        """Exporta los datos de conteo por intervalo a un archivo CSV."""
        data = list(self.interval_data)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if self.counting_mode == "fov":
                fov_partial = self.fov_count - self._interval_fov_start
                if fov_partial > 0:
                    data.append((
                        datetime.now().strftime("%H:%M"),
                        fov_partial, 0, self.fov_count, 0,
                    ))
                writer.writerow(["hora", "personas_intervalo", "personas_total"])
                for row in data:
                    writer.writerow([row[0], row[1], row[3]])
            else:
                in_partial  = self.in_count  - self._interval_in_start
                out_partial = self.out_count - self._interval_out_start
                if in_partial > 0 or out_partial > 0:
                    data.append((
                        datetime.now().strftime("%H:%M"),
                        in_partial, out_partial,
                        self.in_count, self.out_count,
                    ))
                writer.writerow([
                    "hora", "entradas_intervalo", "salidas_intervalo",
                    "entradas_total", "salidas_total",
                ])
                writer.writerows(data)

        logger.info(
            "CSV exportado: %s (%d filas, modo=%s)",
            filepath, len(data), self.counting_mode,
        )

    # ------------------------------------------------------------------
    # Métodos privados — tracker
    # ------------------------------------------------------------------

    def _create_tracker(self) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=TRACKER_MATCHING_THRESHOLD,
            frame_rate=30,
        )

    def _build_strategy(self) -> CountingStrategy:
        """Instancia la estrategia correcta según ``self.counting_mode``."""
        if self.counting_mode == "fov":
            return FovStrategy()

        if self.counting_mode == "roi":
            strategy = PolygonStrategy(self.frame_width, self.frame_height)
            strategy.set_from_normalized_rect(
                self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2
            )
            return strategy

        # "line" (modo por defecto)
        return LineStrategy(
            self.frame_width, self.frame_height,
            line_position_h=self.line_position,
            line_position_v=self.line_position_vertical,
            use_horizontal=self.use_horizontal_line,
            use_vertical=self.use_vertical_line,
            area_threshold=AREA_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Métodos privados — limpieza de estado stale
    # ------------------------------------------------------------------

    def _update_track_activity(self, detections: sv.Detections) -> None:
        """Registra el timestamp del último avistamiento de cada track activo."""
        if detections.tracker_id is None:
            return
        now = time.monotonic()
        for tid in detections.tracker_id:
            self._track_last_seen[int(tid)] = now

    def _maybe_cleanup_stale_tracks(self) -> None:
        """Purga estado interno de tracks inactivos sin recrear el tracker.

        Reemplaza al antiguo _periodic_tracker_reset. La diferencia clave:
        el tracker ByteTrack NO se destruye ni se recrea. Solo se eliminan
        las entradas de los diccionarios de estado de las estrategias para
        tracks que llevan más de STALE_TRACK_THRESHOLD_S sin aparecer.

        Esto libera memoria sin causar el bug de doble conteo.
        """
        now = time.monotonic()
        if (now - self._last_cleanup_time) < STALE_TRACK_CLEANUP_INTERVAL_S:
            return

        stale_ids = {
            tid
            for tid, last_seen in self._track_last_seen.items()
            if (now - last_seen) > STALE_TRACK_THRESHOLD_S
        }

        if stale_ids and self._strategy is not None:
            self._strategy.purge_stale_tracks(stale_ids)
            for tid in stale_ids:
                self._track_last_seen.pop(tid, None)
            logger.debug("Limpieza de tracks stale: %d eliminados.", len(stale_ids))

        self._last_cleanup_time = now

    # ------------------------------------------------------------------
    # Métodos privados — estadísticas por intervalo
    # ------------------------------------------------------------------

    def _check_interval(self) -> None:
        now = time.time()
        if self._last_interval_time is None:
            self._last_interval_time = now
            return
        if now - self._last_interval_time < INTERVAL_SECONDS:
            return

        hora_str = datetime.now().strftime("%H:%M")
        hora = datetime.now().hour

        if self.counting_mode == "fov":
            fov_interval = self.fov_count - self._interval_fov_start
            self.interval_data.append((hora_str, fov_interval, 0, self.fov_count, 0))
            self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + fov_interval
            self._interval_fov_start = self.fov_count
        else:
            in_interval  = self.in_count  - self._interval_in_start
            out_interval = self.out_count - self._interval_out_start
            self.interval_data.append(
                (hora_str, in_interval, out_interval, self.in_count, self.out_count)
            )
            self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + in_interval
            self._interval_in_start  = self.in_count
            self._interval_out_start = self.out_count

        self._last_interval_time = now

    # ------------------------------------------------------------------
    # Métodos privados — filtro de profundidad
    # ------------------------------------------------------------------

    def _filter_by_depth(
        self, detections: sv.Detections, depth_frame: np.ndarray
    ) -> sv.Detections:
        """Descarta detecciones fuera del rango de profundidad configurado.

        Muestrea la profundidad en el centro del bounding box de cada
        detección. Valores 0 del sensor (medición inválida) siempre se
        descartan.

        Args:
            detections:  Detecciones ya actualizadas por el tracker.
            depth_frame: Mapa uint16 en milímetros, misma resolución que
                         el frame de color.

        Returns:
            Subconjunto de detecciones dentro del rango válido.
        """
        if len(detections) == 0 or detections.xyxy is None:
            return detections

        fh, fw = depth_frame.shape[:2]
        mask = []
        for box in detections.xyxy:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            # Clamp por si el bbox toca el borde del frame
            cx = max(0, min(cx, fw - 1))
            cy = max(0, min(cy, fh - 1))
            depth_mm = int(depth_frame[cy, cx])
            if depth_mm == 0:
                # Medición inválida del sensor estéreo → descartar
                mask.append(False)
                continue
            depth_m = depth_mm / 1000.0
            mask.append(MIN_DETECTION_DEPTH_M <= depth_m <= MAX_DETECTION_DEPTH_M)

        return detections[np.array(mask, dtype=bool)]

    def _depth_label(self, box: np.ndarray, depth_frame: np.ndarray) -> str:
        """Retorna la profundidad en el centro del bbox como string legible."""
        fh, fw = depth_frame.shape[:2]
        cx = max(0, min(int((box[0] + box[2]) / 2), fw - 1))
        cy = max(0, min(int((box[1] + box[3]) / 2), fh - 1))
        d_mm = int(depth_frame[cy, cx])
        return f"{d_mm / 1000:.1f}m" if d_mm > 0 else "?"

    # ------------------------------------------------------------------
    # Métodos privados — anotación de frames
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        depth_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dibuja bounding boxes, trazas, etiquetas y overlay del modo activo."""
        labels = []
        if detections.tracker_id is not None:
            if depth_frame is not None:
                labels = [
                    f"#{tid} {conf:.0%} {self._depth_label(box, depth_frame)}"
                    for tid, conf, box in zip(
                        detections.tracker_id, detections.confidence, detections.xyxy
                    )
                ]
            else:
                labels = [
                    f"#{tid} {conf:.0%}"
                    for tid, conf in zip(detections.tracker_id, detections.confidence)
                ]

        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        if labels:
            frame = self.label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        if self._strategy is not None:
            frame = self._strategy.draw_overlay(frame)

        # Punto central de cada detección
        if len(detections) > 0 and detections.xyxy is not None:
            for box in detections.xyxy:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (0, 180, 180), 1)

        return frame
