"""Motor de detección, tracking y conteo de personas.

Usa YOLOv8 para detección y ByteTrack para seguimiento de objetos.
Soporta tres modos de conteo independientes:
  - ``line``  : cruce de una o dos líneas (horizontal y/o vertical).
  - ``roi``   : entrada/salida de una zona rectangular de interés.
  - ``fov``   : campo de visión — acumula cada ID único visto al menos una vez.
"""

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
    DEFAULT_MODEL,
    DEFAULT_ROI,
    INFERENCE_DEVICE,
    INTERVAL_SECONDS,
    PERSON_CLASS_ID,
    TRACKER_RESET_INTERVAL_S,
)

logger = logging.getLogger("ContadorPersonas")


class PersonCounter:
    """Motor de detección, tracking y conteo de personas.

    Esta clase no tiene ninguna dependencia con la interfaz gráfica;
    puede usarse de forma independiente o en pruebas unitarias.
    """

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

        # Líneas activas
        self.use_horizontal_line = True
        self.use_vertical_line = False

        # ROI (coordenadas normalizadas 0–1)
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = DEFAULT_ROI
        self.use_roi_mode = False

        # Resolución de inferencia YOLO
        self.infer_size = infer_size

        # Modo de conteo: "line" | "roi" | "fov"
        self.counting_mode = "line"

        # Contadores — modo línea / ROI
        self.in_count = 0
        self.out_count = 0
        self._in_offset = 0
        self._out_offset = 0

        # Contadores — modo FOV
        self.fov_count = 0
        self._fov_offset = 0
        self._seen_ids: set = set()

        # Métricas de frame
        self.persons_in_frame = 0
        self.fps = 0.0

        # Estadísticas por intervalo y por hora
        self.interval_data: list = []
        self._last_interval_time: float | None = None
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0
        self.hourly_entries: dict = {}

        # Internos del pipeline de detección
        self.model: YOLO | None = None
        self.tracker: sv.ByteTrack | None = None
        self.line_zone: sv.LineZone | None = None
        self.line_zone_horizontal: sv.LineZone | None = None
        self.line_zone_vertical: sv.LineZone | None = None
        self.frame_width = 640
        self.frame_height = 480
        self._fps_buffer: deque = deque(maxlen=30)
        self._tracker_reset_time = time.time()
        self._tracker_reset_interval = TRACKER_RESET_INTERVAL_S

        # Estado de tracking por ID de objeto
        self._tracker_side_h: dict = {}    # {tracker_id: "up" | "down"}
        self._tracker_side_v: dict = {}    # {tracker_id: "left" | "right"}
        self._area_threshold = AREA_THRESHOLD
        self._tracker_in_roi: dict = {}    # {tracker_id: bool}

        # Anotadores de supervision
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Carga el modelo YOLO en el dispositivo configurado en ``INFERENCE_DEVICE``."""
        logger.info("Cargando modelo YOLO: %s (device=%s)", self.model_name, INFERENCE_DEVICE)
        self.model = YOLO(self.model_name)
        self.model.to(INFERENCE_DEVICE)
        logger.info("Modelo cargado correctamente.")

    def setup_line(self, frame_width: int, frame_height: int) -> None:
        """Configura la zona de conteo y el tracker para las dimensiones del video.

        Debe llamarse una sola vez al abrir cada fuente de video, antes de
        empezar a procesar frames.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._create_line_zone()
        self._create_tracker()
        if self._last_interval_time is None:
            self._last_interval_time = time.time()

    # ------------------------------------------------------------------
    # Configuración dinámica (accesible desde la GUI en tiempo real)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Cambia el modo de conteo activo.

        Args:
            mode: ``"line"``, ``"roi"`` o ``"fov"``.
        """
        self.counting_mode = mode
        self.use_roi_mode = (mode == "roi")
        logger.info("Modo de conteo cambiado a: %s", mode)

    def set_roi(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Define la zona rectangular de interés en coordenadas normalizadas (0–1)."""
        self.roi_x1 = min(x1, x2)
        self.roi_y1 = min(y1, y2)
        self.roi_x2 = max(x1, x2)
        self.roi_y2 = max(y1, y2)
        logger.info("ROI actualizado: (%.2f, %.2f) -> (%.2f, %.2f)", x1, y1, x2, y2)

    def update_line(
        self,
        position: float,
        orientation: str | None = None,
        position_vertical: float | None = None,
    ) -> None:
        """Actualiza la posición de las líneas de conteo y las recrea."""
        self.line_position = position
        if position_vertical is not None:
            self.line_position_vertical = position_vertical
        if orientation is not None:
            self.line_orientation = orientation
        self._create_line_zone()

    def set_line_enabled(self, horizontal: bool | None = None, vertical: bool | None = None) -> None:
        """Habilita o deshabilita líneas individuales y las recrea."""
        if horizontal is not None:
            self.use_horizontal_line = horizontal
        if vertical is not None:
            self.use_vertical_line = vertical
        self._create_line_zone()

    def update_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Procesamiento de frames
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame individual. Wrapper de :meth:`process_batch`."""
        return self.process_batch([frame])[0]

    def process_batch(self, frames: list) -> list:
        """Procesa una lista de frames en una sola llamada a YOLO (batch GPU).

        Args:
            frames: Lista de imágenes BGR (``np.ndarray``).

        Returns:
            Lista de frames anotados en el mismo orden.
        """
        t_start = time.perf_counter()

        if self.model is None:
            return frames

        if time.time() - self._tracker_reset_time > self._tracker_reset_interval:
            self._periodic_tracker_reset()

        results_list = self.model(
            frames,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            imgsz=self.infer_size,
            device=INFERENCE_DEVICE,
            verbose=False,
        )

        annotated = []
        for frame, results in zip(frames, results_list):
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            if self.counting_mode == "fov":
                self._update_fov_count(detections)
            elif self.counting_mode == "roi":
                self._detect_roi_crossings(detections)
            else:
                self._detect_crossings_with_area_threshold(detections)

            self.persons_in_frame = len(detections)
            self._check_interval()

            annotated.append(self._annotate_frame(frame, detections))

        elapsed = time.perf_counter() - t_start
        fps_batch = len(frames) / elapsed if elapsed > 0 else 0
        self._fps_buffer.append(fps_batch)
        self.fps = sum(self._fps_buffer) / len(self._fps_buffer)

        return annotated

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Reinicia todos los contadores y el estado interno del tracker."""
        self.in_count = 0
        self.out_count = 0
        self._in_offset = 0
        self._out_offset = 0
        self.fov_count = 0
        self._fov_offset = 0
        self._seen_ids = set()
        self.persons_in_frame = 0
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0
        self._last_interval_time = time.time()
        self.interval_data.clear()
        self.hourly_entries.clear()
        self._tracker_side_h.clear()
        self._tracker_side_v.clear()
        self._tracker_in_roi.clear()
        self.line_zone = None
        self.line_zone_horizontal = None
        self.line_zone_vertical = None
        self._create_line_zone()
        self._create_tracker()
        logger.info("Contadores reiniciados.")

    def export_csv(self, filepath: str) -> None:
        """Exporta los datos de conteo por intervalo a un archivo CSV.

        Args:
            filepath: Ruta completa del archivo ``.csv`` de salida.
        """
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
                in_partial = self.in_count - self._interval_in_start
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
    # Métodos privados — líneas y tracker
    # ------------------------------------------------------------------

    def _create_line_zone(self) -> None:
        """Recrea las zonas de conteo conservando los conteos anteriores."""
        if self.line_zone_horizontal is not None:
            self._in_offset += self.line_zone_horizontal.in_count
            self._out_offset += self.line_zone_horizontal.out_count
        if self.line_zone_vertical is not None:
            self._in_offset += self.line_zone_vertical.in_count
            self._out_offset += self.line_zone_vertical.out_count

        if self.use_horizontal_line:
            y = int(self.frame_height * self.line_position)
            self.line_zone_horizontal = sv.LineZone(
                start=sv.Point(0, y),
                end=sv.Point(self.frame_width, y),
                triggering_anchors=[sv.Position.CENTER],
            )
        else:
            self.line_zone_horizontal = None

        if self.use_vertical_line:
            x = int(self.frame_width * self.line_position_vertical)
            self.line_zone_vertical = sv.LineZone(
                start=sv.Point(x, 0),
                end=sv.Point(x, self.frame_height),
                triggering_anchors=[sv.Position.CENTER],
            )
        else:
            self.line_zone_vertical = None

        self.line_zone = self.line_zone_horizontal or self.line_zone_vertical

    def _create_tracker(self) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self._tracker_reset_time = time.time()

    def _periodic_tracker_reset(self) -> None:
        logger.info("Reset periódico del tracker.")
        if self.counting_mode == "fov":
            self._fov_offset += len(self._seen_ids)
            self._seen_ids = set()
        self._tracker_side_h.clear()
        self._tracker_side_v.clear()
        self._create_tracker()

    # ------------------------------------------------------------------
    # Métodos privados — detección de cruces
    # ------------------------------------------------------------------

    def _detect_crossings_with_area_threshold(self, detections: sv.Detections) -> None:
        """Detecta cruces de línea usando umbral de área del bounding box."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0:
                continue

            if self.line_zone_horizontal is not None:
                self._check_horizontal_crossing(tid, x1, y1, x2, y2, bbox_area)
            if self.line_zone_vertical is not None:
                self._check_vertical_crossing(tid, x1, y1, x2, y2, bbox_area)

    def _check_horizontal_crossing(
        self, tid, x1: float, y1: float, x2: float, y2: float, bbox_area: float
    ) -> None:
        """Evalúa si el tracker cruzó la línea horizontal y actualiza contadores."""
        y_line = int(self.frame_height * self.line_position)

        if y2 < y_line:
            current_side = "up"
        elif y1 > y_line:
            current_side = "down"
        else:
            area_up = (x2 - x1) * (y_line - y1) if y1 < y_line else 0
            area_down = (x2 - x1) * (y2 - y_line) if y2 > y_line else 0
            if area_up / bbox_area >= self._area_threshold:
                current_side = "up"
            elif area_down / bbox_area >= self._area_threshold:
                current_side = "down"
            else:
                current_side = self._tracker_side_h.get(tid)

        prev_side = self._tracker_side_h.get(tid)
        if current_side is not None:
            if prev_side is not None and prev_side != current_side:
                if current_side == "down":
                    self.in_count += 1
                    logger.debug("Cruce H: Tracker %s - IN (up->down)", tid)
                else:
                    self.out_count += 1
                    logger.debug("Cruce H: Tracker %s - OUT (down->up)", tid)
            self._tracker_side_h[tid] = current_side

    def _check_vertical_crossing(
        self, tid, x1: float, y1: float, x2: float, y2: float, bbox_area: float
    ) -> None:
        """Evalúa si el tracker cruzó la línea vertical y actualiza contadores."""
        x_line = int(self.frame_width * self.line_position_vertical)

        if x2 < x_line:
            current_side = "left"
        elif x1 > x_line:
            current_side = "right"
        else:
            area_left = (y2 - y1) * (x_line - x1) if x1 < x_line else 0
            area_right = (y2 - y1) * (x2 - x_line) if x2 > x_line else 0
            if area_left / bbox_area >= self._area_threshold:
                current_side = "left"
            elif area_right / bbox_area >= self._area_threshold:
                current_side = "right"
            else:
                current_side = self._tracker_side_v.get(tid)

        prev_side = self._tracker_side_v.get(tid)
        if current_side is not None:
            if prev_side is not None and prev_side != current_side:
                if current_side == "right":
                    self.in_count += 1
                    logger.debug("Cruce V: Tracker %s - IN (left->right)", tid)
                else:
                    self.out_count += 1
                    logger.debug("Cruce V: Tracker %s - OUT (right->left)", tid)
            self._tracker_side_v[tid] = current_side

    def _detect_roi_crossings(self, detections: sv.Detections) -> None:
        """Detecta entradas y salidas de la zona rectangular de interés."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        roi_x1_px = int(self.frame_width * self.roi_x1)
        roi_y1_px = int(self.frame_height * self.roi_y1)
        roi_x2_px = int(self.frame_width * self.roi_x2)
        roi_y2_px = int(self.frame_height * self.roi_y2)

        if (roi_x2_px - roi_x1_px) * (roi_y2_px - roi_y1_px) == 0:
            return

        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]

            inter_x1 = max(x1, roi_x1_px)
            inter_y1 = max(y1, roi_y1_px)
            inter_x2 = min(x2, roi_x2_px)
            inter_y2 = min(y2, roi_y2_px)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                bbox_area = (x2 - x1) * (y2 - y1)
                is_inside = bbox_area > 0 and (inter_area / bbox_area) >= self._area_threshold
            else:
                is_inside = False

            was_inside = self._tracker_in_roi.get(tid, False)
            if is_inside and not was_inside:
                self.in_count += 1
                logger.debug("ROI: Tracker %s - ENTRADA", tid)
            elif not is_inside and was_inside:
                self.out_count += 1
                logger.debug("ROI: Tracker %s - SALIDA", tid)

            self._tracker_in_roi[tid] = is_inside

    def _update_fov_count(self, detections: sv.Detections) -> None:
        """Acumula IDs únicos vistos (modo campo de visión)."""
        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self._seen_ids.add(tid)
        self.fov_count = self._fov_offset + len(self._seen_ids)

    def _check_interval(self) -> None:
        """Registra un intervalo de estadísticas si transcurrió el tiempo configurado."""
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
            in_interval = self.in_count - self._interval_in_start
            out_interval = self.out_count - self._interval_out_start
            self.interval_data.append(
                (hora_str, in_interval, out_interval, self.in_count, self.out_count)
            )
            self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + in_interval
            self._interval_in_start = self.in_count
            self._interval_out_start = self.out_count

        self._last_interval_time = now

    # ------------------------------------------------------------------
    # Métodos privados — anotación de frames
    # ------------------------------------------------------------------

    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Dibuja bounding boxes, trazas, etiquetas y overlays sobre el frame."""
        labels = []
        if detections.tracker_id is not None:
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

        if self.counting_mode == "roi":
            frame = self._draw_roi_overlay(frame)
        elif self.counting_mode == "line":
            frame = self._draw_line_overlay(frame)

        # Punto de centro de cada detección
        if len(detections) > 0 and detections.xyxy is not None:
            for box in detections.xyxy:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (0, 180, 180), 1)

        return frame

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        x1 = int(self.frame_width * self.roi_x1)
        y1 = int(self.frame_height * self.roi_y1)
        x2 = int(self.frame_width * self.roi_x2)
        y2 = int(self.frame_height * self.roi_y2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame, f"IN: {self.in_count}  OUT: {self.out_count}",
            (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        return frame

    def _draw_line_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.line_zone_horizontal is not None:
            y = int(self.frame_height * self.line_position)
            cv2.line(frame, (0, y), (self.frame_width, y), (0, 255, 255), 3)
            cv2.putText(
                frame, f"IN: {self.in_count}  OUT: {self.out_count}",
                (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        if self.line_zone_vertical is not None:
            x = int(self.frame_width * self.line_position_vertical)
            cv2.line(frame, (x, 0), (x, self.frame_height), (255, 0, 255), 3)
            if self.line_zone_horizontal is None:
                cv2.putText(
                    frame, f"IN: {self.in_count}  OUT: {self.out_count}",
                    (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
                )

        return frame
