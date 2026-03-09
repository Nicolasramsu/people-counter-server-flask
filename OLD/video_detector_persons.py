import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import threading
import time
import csv
import os
import logging
import queue
from datetime import datetime
from collections import deque
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contador.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ContadorPersonas")

PERSON_CLASS_ID = 0


# ===========================================================================
# PersonCounter - Motor de deteccion (sin GUI)
# ===========================================================================
class PersonCounter:
    """Motor de deteccion, tracking y conteo de personas."""

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence: float = 0.3,
        line_position: float = 0.7,
        line_orientation: str = "horizontal",
        frame_skip: int = 1,
        batch_size: int = 1,
        infer_size: int = 640,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.line_position = line_position
        self.line_position_vertical = 0.5  # Posición de línea vertical
        self.line_orientation = line_orientation
        
        # Habilitar/deshabilitar líneas individuales
        self.use_horizontal_line = True
        self.use_vertical_line = False
        
        # Zona rectangular de interés (ROI) - coordenadas normalizadas 0-1
        self.roi_x1 = 0.15  # izquierda
        self.roi_y1 = 0.02  # arriba
        self.roi_x2 = 1  # derecha
        self.roi_y2 = 1  # abajo
        self.use_roi_mode = False

        # --- Optimizaciones de velocidad ---
        # frame_skip=2 -> procesa 1 de cada 2 frames (2x velocidad)
        # frame_skip=4 -> procesa 1 de cada 4 frames (4x velocidad)
        self.frame_skip = max(1, frame_skip)
        # batch_size: cuantos frames enviar a YOLO en una sola llamada GPU
        self.batch_size = max(1, batch_size)
        # infer_size: resolucion de inferencia YOLO (320=rapido, 640=normal, 1280=preciso)
        self.infer_size = infer_size

        # Modo de conteo: "line" (cruce de linea) | "roi" (zona rectangular) | "fov" (campo de vision)
        self.counting_mode = "line"

        # Estadisticas modo linea
        self.in_count = 0
        self.out_count = 0
        self._in_offset = 0
        self._out_offset = 0
        self.persons_in_frame = 0
        self.fps = 0.0

        # Estadisticas modo campo de vision (FOV)
        self.fov_count = 0
        self._fov_offset = 0
        self._seen_ids = set()

        # Datos por intervalo (cada 15 min simulados o reales)
        self.interval_data = []
        self._last_interval_time = None
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0

        # Datos por hora para el grafico
        self.hourly_entries = {}

        # Internos
        self.model = None
        self.tracker = None
        self.line_zone = None
        self.line_zone_horizontal = None
        self.line_zone_vertical = None
        self.frame_width = 640
        self.frame_height = 480
        self._fps_buffer = deque(maxlen=30)
        self._tracker_reset_time = time.time()
        self._tracker_reset_interval = 30 * 60
        
        # Tracking manual para detección de cruce con 30% de área
        self._tracker_side_h = {}  # {tracker_id: "up" o "down"}
        self._tracker_side_v = {}  # {tracker_id: "left" o "right"}
        self._area_threshold = 0.30  # 30% del área del bbox debe cruzar
        
        # Tracking para modo ROI
        self._tracker_in_roi = {}  # {tracker_id: True/False} - si está dentro del ROI

        # Anotadores
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

    def load_model(self):
        """Carga el modelo YOLO."""
        logger.info("Cargando modelo YOLO: %s", self.model_name)
        self.model = YOLO(self.model_name)
        self.model.to("cuda:0")
        logger.info("Modelo cargado correctamente.")

    def setup_line(self, frame_width: int, frame_height: int):
        """Configura la linea de conteo y el tracker."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._create_line_zone()
        self._create_tracker()
        if self._last_interval_time is None:
            self._last_interval_time = time.time()

    def _create_line_zone(self):
        """Crea las líneas de conteo (horizontal y/o vertical)."""
        # Guardar conteos anteriores si existían
        if self.line_zone_horizontal is not None:
            self._in_offset += self.line_zone_horizontal.in_count
            self._out_offset += self.line_zone_horizontal.out_count
        if self.line_zone_vertical is not None:
            self._in_offset += self.line_zone_vertical.in_count
            self._out_offset += self.line_zone_vertical.out_count
            
        # Línea horizontal
        if self.use_horizontal_line:
            y = int(self.frame_height * self.line_position)
            start_h = sv.Point(0, y)
            end_h = sv.Point(self.frame_width, y)
            self.line_zone_horizontal = sv.LineZone(
                start=start_h,
                end=end_h,
                triggering_anchors=[sv.Position.CENTER],
            )
        else:
            self.line_zone_horizontal = None
            
        # Línea vertical
        if self.use_vertical_line:
            x = int(self.frame_width * self.line_position_vertical)
            start_v = sv.Point(x, 0)
            end_v = sv.Point(x, self.frame_height)
            self.line_zone_vertical = sv.LineZone(
                start=start_v,
                end=end_v,
                triggering_anchors=[sv.Position.CENTER],
            )
        else:
            self.line_zone_vertical = None
            
        # Para compatibilidad con código que usa self.line_zone
        self.line_zone = self.line_zone_horizontal or self.line_zone_vertical

    def _create_tracker(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self._tracker_reset_time = time.time()

    def _periodic_tracker_reset(self):
        logger.info("Reset periodico del tracker.")
        if self.counting_mode == "fov":
            self._fov_offset += len(self._seen_ids)
            self._seen_ids = set()
        # Limpiar tracking de posiciones
        self._tracker_side_h.clear()
        self._tracker_side_v.clear()
        self._create_tracker()

    def set_mode(self, mode: str):
        self.counting_mode = mode
        if mode == "roi":
            self.use_roi_mode = True
        else:
            self.use_roi_mode = False
        logger.info("Modo de conteo cambiado a: %s", mode)
    
    def set_roi(self, x1: float, y1: float, x2: float, y2: float):
        """Define la zona rectangular de interés (coordenadas normalizadas 0-1)."""
        self.roi_x1 = min(x1, x2)
        self.roi_y1 = min(y1, y2)
        self.roi_x2 = max(x1, x2)
        self.roi_y2 = max(y1, y2)
        logger.info("ROI actualizado: (%.2f, %.2f) -> (%.2f, %.2f)", x1, y1, x2, y2)

    def update_line(self, position: float, orientation: str = None, position_vertical: float = None):
        """Actualiza la posición de las líneas."""
        self.line_position = position
        if position_vertical is not None:
            self.line_position_vertical = position_vertical
        if orientation is not None:
            self.line_orientation = orientation
        self._create_line_zone()
    
    def set_line_enabled(self, horizontal: bool = None, vertical: bool = None):
        """Habilita o deshabilita líneas individuales."""
        if horizontal is not None:
            self.use_horizontal_line = horizontal
        if vertical is not None:
            self.use_vertical_line = vertical
        self._create_line_zone()

    def update_confidence(self, confidence: float):
        self.confidence = confidence

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame individual (wrapper de process_batch para compatibilidad)."""
        return self.process_batch([frame])[0]

    def process_batch(self, frames: list) -> list:
        """Procesa una lista de frames en una sola llamada a YOLO (mas eficiente en GPU).

        Retorna la lista de frames anotados en el mismo orden.
        """
        t_start = time.perf_counter()

        if self.model is None:
            return frames

        if time.time() - self._tracker_reset_time > self._tracker_reset_interval:
            self._periodic_tracker_reset()

        # --- Inferencia en batch: una sola llamada para todos los frames ---
        results_list = self.model(
            frames,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            imgsz=self.infer_size,
            device=0,   # fuerza GPU 0
            verbose=False
        )

        annotated = []
        for frame, results in zip(frames, results_list):
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            if self.counting_mode == "fov":
                self._update_fov_count(detections)
            elif self.counting_mode == "roi":
                # Detección de entrada/salida de zona rectangular
                self._detect_roi_crossings(detections)
            else:
                # Detección personalizada de cruce con 30% de área (modo línea)
                self._detect_crossings_with_area_threshold(detections)

            self.persons_in_frame = len(detections)
            self._check_interval()

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
                # Dibujar rectángulo ROI
                x1 = int(self.frame_width * self.roi_x1)
                y1 = int(self.frame_height * self.roi_y1)
                x2 = int(self.frame_width * self.roi_x2)
                y2 = int(self.frame_height * self.roi_y2)
                
                # Rectángulo con relleno semi-transparente
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                # Borde del rectángulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Texto con contadores
                text = f"IN: {self.in_count}  OUT: {self.out_count}"
                cv2.putText(frame, text, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                
            elif self.counting_mode == "line":
                # Dibujar líneas activas
                if self.line_zone_horizontal is not None:
                    y = int(self.frame_height * self.line_position)
                    cv2.line(frame, (0, y), (self.frame_width, y), (0, 255, 255), 3)
                    # Texto con contadores
                    text = f"IN: {self.in_count}  OUT: {self.out_count}"
                    cv2.putText(frame, text, (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2)
                if self.line_zone_vertical is not None:
                    x = int(self.frame_width * self.line_position_vertical)
                    cv2.line(frame, (x, 0), (x, self.frame_height), (255, 0, 255), 3)
                    # Si solo hay línea vertical, mostrar contadores ahí
                    if self.line_zone_horizontal is None:
                        text = f"IN: {self.in_count}  OUT: {self.out_count}"
                        cv2.putText(frame, text, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (255, 0, 255), 2)

            if len(detections) > 0 and detections.xyxy is not None:
                for box in detections.xyxy:
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 7, (0, 180, 180), 1)

            annotated.append(frame)

        # FPS calculado sobre el batch completo
        elapsed = time.perf_counter() - t_start
        fps_batch = len(frames) / elapsed if elapsed > 0 else 0
        self._fps_buffer.append(fps_batch)
        self.fps = sum(self._fps_buffer) / len(self._fps_buffer)

        return annotated

    def _detect_crossings_with_area_threshold(self, detections):
        if detections.tracker_id is None or len(detections) == 0:
            return
            
        for i, tid in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            if bbox_area == 0:
                continue
            
            # Detectar cruce de línea horizontal
            if self.line_zone_horizontal is not None:
                y_line = int(self.frame_height * self.line_position)
                
                # Calcular qué porcentaje del bbox está arriba vs abajo de la línea
                if y2 < y_line:
                    # Completamente arriba
                    current_side_h = "up"
                elif y1 > y_line:
                    # Completamente abajo
                    current_side_h = "down"
                else:
                    # El bbox intersecta la línea
                    area_up = (x2 - x1) * (y_line - y1) if y1 < y_line else 0
                    area_down = (x2 - x1) * (y2 - y_line) if y2 > y_line else 0
                    
                    # Determinar lado basado en 30% de área
                    if area_up / bbox_area >= self._area_threshold:
                        current_side_h = "up"
                    elif area_down / bbox_area >= self._area_threshold:
                        current_side_h = "down"
                    else:
                        # No cumple el umbral de 30% en ningún lado, mantener lado anterior
                        current_side_h = self._tracker_side_h.get(tid, None)
                
                # Detectar cruce
                prev_side_h = self._tracker_side_h.get(tid)
                if current_side_h is not None:
                    if prev_side_h is not None and prev_side_h != current_side_h:
                        if current_side_h == "down":
                            self.in_count += 1
                            logger.debug(f"Cruce H: Tracker {tid} - IN (up->down)")
                        else:
                            self.out_count += 1
                            logger.debug(f"Cruce H: Tracker {tid} - OUT (down->up)")
                    self._tracker_side_h[tid] = current_side_h
            
            # Detectar cruce de línea vertical
            if self.line_zone_vertical is not None:
                x_line = int(self.frame_width * self.line_position_vertical)
                
                # Calcular qué porcentaje del bbox está a la izquierda vs derecha
                if x2 < x_line:
                    # Completamente a la izquierda
                    current_side_v = "left"
                elif x1 > x_line:
                    # Completamente a la derecha
                    current_side_v = "right"
                else:
                    # El bbox intersecta la línea
                    area_left = (y2 - y1) * (x_line - x1) if x1 < x_line else 0
                    area_right = (y2 - y1) * (x2 - x_line) if x2 > x_line else 0
                    
                    # Determinar lado basado en 30% de área
                    if area_left / bbox_area >= self._area_threshold:
                        current_side_v = "left"
                    elif area_right / bbox_area >= self._area_threshold:
                        current_side_v = "right"
                    else:
                        # No cumple el umbral, mantener lado anterior
                        current_side_v = self._tracker_side_v.get(tid, None)
                
                # Detectar cruce
                prev_side_v = self._tracker_side_v.get(tid)
                if current_side_v is not None:
                    if prev_side_v is not None and prev_side_v != current_side_v:
                        if current_side_v == "right":
                            self.in_count += 1
                            logger.debug(f"Cruce V: Tracker {tid} - IN (left->right)")
                        else:
                            self.out_count += 1
                            logger.debug(f"Cruce V: Tracker {tid} - OUT (right->left)")
                    self._tracker_side_v[tid] = current_side_v
    
    def _detect_roi_crossings(self, detections):
        if detections.tracker_id is None or len(detections) == 0:
            return
        
        # Coordenadas del ROI en píxeles
        roi_x1_px = int(self.frame_width * self.roi_x1)
        roi_y1_px = int(self.frame_height * self.roi_y1)
        roi_x2_px = int(self.frame_width * self.roi_x2)
        roi_y2_px = int(self.frame_height * self.roi_y2)
        roi_area = (roi_x2_px - roi_x1_px) * (roi_y2_px - roi_y1_px)
        
        if roi_area == 0:
            return
        
        for i, tid in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Calcular intersección del bbox con el ROI
            inter_x1 = max(x1, roi_x1_px)
            inter_y1 = max(y1, roi_y1_px)
            inter_x2 = min(x2, roi_x2_px)
            inter_y2 = min(y2, roi_y2_px)
            
            # Área de intersección
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                bbox_area = (x2 - x1) * (y2 - y1)
                
                if bbox_area == 0:
                    continue
                
                # Porcentaje del bbox dentro del ROI
                overlap_ratio = inter_area / bbox_area
                
                # Considerar que está dentro si 30% o más del bbox está en el ROI
                is_inside = overlap_ratio >= self._area_threshold
            else:
                # No hay intersección
                is_inside = False
            
            # Detectar cambios de estado (entrada/salida)
            was_inside = self._tracker_in_roi.get(tid, False)
            
            if is_inside and not was_inside:
                # Entrada al ROI
                self.in_count += 1
                logger.debug(f"ROI: Tracker {tid} - ENTRADA (overlap={overlap_ratio:.1%})")
            elif not is_inside and was_inside:
                # Salida del ROI
                self.out_count += 1
                logger.debug(f"ROI: Tracker {tid} - SALIDA")
            
            # Actualizar estado
            self._tracker_in_roi[tid] = is_inside
    
    def _update_fov_count(self, detections):
        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self._seen_ids.add(tid)
        self.fov_count = self._fov_offset + len(self._seen_ids)

    def _check_interval(self):
        now = time.time()
        if self._last_interval_time is None:
            self._last_interval_time = now
            return
        if now - self._last_interval_time >= 15 * 60:
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

    def reset_counters(self):
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

    def export_csv(self, filepath: str):
        data = list(self.interval_data)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if self.counting_mode == "fov":
                fov_partial = self.fov_count - self._interval_fov_start
                if fov_partial > 0:
                    hora_str = datetime.now().strftime("%H:%M")
                    data.append((hora_str, fov_partial, 0, self.fov_count, 0))
                writer.writerow(["hora", "personas_intervalo", "personas_total"])
                for row in data:
                    writer.writerow([row[0], row[1], row[3]])
            else:
                in_partial = self.in_count - self._interval_in_start
                out_partial = self.out_count - self._interval_out_start
                if in_partial > 0 or out_partial > 0:
                    hora_str = datetime.now().strftime("%H:%M")
                    data.append((hora_str, in_partial, out_partial, self.in_count, self.out_count))
                writer.writerow(["hora", "entradas_intervalo", "salidas_intervalo",
                                 "entradas_total", "salidas_total"])
                writer.writerows(data)

        logger.info("CSV exportado: %s (%d filas, modo=%s)", filepath, len(data), self.counting_mode)


# ===========================================================================
# CounterApp - Interfaz grafica Tkinter
# ===========================================================================
class CounterApp(tk.Tk):
    """Ventana principal de la aplicacion."""

    DISPLAY_W = 640
    DISPLAY_H = 480
    UPDATE_MS = 33  # ~30 fps GUI

    def __init__(self):
        super().__init__()
        self.title("Contador de Personas - Análisis de Video")
        self.resizable(True, True)
        self.configure(bg="#2b2b2b")
        # Ancho: video(640) + panel derecho(290) + márgenes(~30) = ~960
        # Alto: video(480) + barra progreso(~25) + márgenes(~20) = ~530
        self.minsize(960, 540)

        # Estado
        self._running = False
        self._paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # Set = pausado
        self._video_thread = None
        self._lock = threading.Lock()
        self._current_frame = None
        self._photo_image = None

        # Archivo de video seleccionado
        self._video_path = None

        # Progreso del video
        self._video_progress = 0.0   # 0.0 a 1.0
        self._video_pos_str = "00:00"
        self._video_dur_str = "00:00"
        self._video_total_frames = 0
        self._video_current_frame = 0

        # Motor de deteccion
        self.counter = PersonCounter()

        self._canvas_image_id = None

        # Auto-guardado
        self._autosave_interval = 5 * 60 * 1000
        self._csv_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Variables de optimizacion ---
        # Cola pipeline: hilo lector -> hilo procesador (max 4 frames en buffer para menor latencia)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self._reader_thread = None

        self._build_gui()

        self._model_loaded = False
        threading.Thread(target=self._load_model_async, daemon=True).start()

        # Inicializar estado de controles de línea
        self._on_lines_enabled_change()

        self.after(self._autosave_interval, self._auto_save)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Aplicacion iniciada.")

    # -----------------------------------------------------------------------
    # Construccion de la GUI
    # -----------------------------------------------------------------------
    def _build_gui(self):

        BG = "#2b2b2b"
        PANEL_W = 290   # ancho fijo del panel derecho (sliders necesitan ~240 px mínimo)
        SLIDER_LEN = 220  # longitud fija para todos los sliders

        cfg_lbl   = {"bg": BG, "fg": "#cccccc", "font": ("Segoe UI", 9), "anchor": "w"}
        radio_kw  = {"bg": BG, "fg": "white", "selectcolor": "#444444",
                     "font": ("Segoe UI", 9), "activebackground": BG, "activeforeground": "white"}
        btn_kw    = {"font": ("Segoe UI", 9), "pady": 2, "relief": "flat", "cursor": "hand2"}
        slider_kw = {"orient": "horizontal", "bg": BG, "fg": "white",
                     "highlightthickness": 0, "troughcolor": "#555555",
                     "length": SLIDER_LEN, "showvalue": True}
        self._lbl_style = {"bg": BG, "font": ("Consolas", 12), "anchor": "w"}

        # ── contenedor raíz ──────────────────────────────────────────────
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=6, pady=6)

        # ══════════════════════════════════════════════════════════════════
        # COLUMNA IZQUIERDA: video + barra de progreso
        # ══════════════════════════════════════════════════════════════════
        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="y")

        self.video_canvas = tk.Canvas(
            left, width=self.DISPLAY_W, height=self.DISPLAY_H,
            bg="black", highlightthickness=0,
        )
        self.video_canvas.pack()
        self._draw_placeholder("Seleccione un video para comenzar")

        # Barra de progreso
        prog_frame = tk.Frame(left, bg=BG)
        prog_frame.pack(fill="x", pady=(4, 0))

        self.lbl_time = tk.Label(prog_frame, text="00:00 / 00:00",
                                  bg=BG, fg="#888888", font=("Consolas", 9))
        self.lbl_time.pack(side="left", padx=(0, 5))

        self.progress_bar = ttk.Progressbar(
            prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True)

        # ══════════════════════════════════════════════════════════════════
        # COLUMNA DERECHA: panel con scroll vertical
        # ══════════════════════════════════════════════════════════════════
        right_outer = tk.Frame(root, bg=BG, width=PANEL_W)
        right_outer.pack(side="left", fill="both", expand=True, padx=(8, 0))
        right_outer.pack_propagate(False)  # respeta el ancho fijo

        # Canvas + scrollbar
        vscroll = tk.Scrollbar(right_outer, orient="vertical")
        vscroll.pack(side="right", fill="y")

        scroll_canvas = tk.Canvas(right_outer, bg=BG, highlightthickness=0,
                                   yscrollcommand=vscroll.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=scroll_canvas.yview)

        # Frame interno que contiene todos los widgets
        inner = tk.Frame(scroll_canvas, bg=BG)
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        # Actualizar scrollregion al cambiar tamaño del contenido
        def _on_inner_configure(e):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        # El frame interno se ensancha con el canvas
        def _on_canvas_configure(e):
            scroll_canvas.itemconfig(win_id, width=e.width)
        scroll_canvas.bind("<Configure>", _on_canvas_configure)

        # Rueda del ratón
        def _on_mousewheel(e):
            scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Atajo: P para pack de todos los widgets dentro de `inner`
        def _pack(widget, **kw):
            widget.pack(fill="x", padx=8, **kw)

        # ── Sección: Estadísticas ─────────────────────────────────────────
        stats_lf = tk.LabelFrame(inner, text="Estadísticas", bg=BG, fg="white",
                                  font=("Segoe UI", 9, "bold"))
        stats_lf.pack(fill="x", padx=5, pady=(0, 4))

        self.lbl_entradas  = tk.Label(stats_lf, text="ENTRADAS:   0", fg="#00ff88", **self._lbl_style)
        self.lbl_entradas.pack(fill="x", padx=8, pady=(4, 0))
        self.lbl_salidas   = tk.Label(stats_lf, text="SALIDAS:    0", fg="#ff6644", **self._lbl_style)
        self.lbl_salidas.pack(fill="x", padx=8)
        self.lbl_en_cuadro = tk.Label(stats_lf, text="EN CUADRO:  0", fg="#ffdd44", **self._lbl_style)
        self.lbl_en_cuadro.pack(fill="x", padx=8)
        self.lbl_fps       = tk.Label(stats_lf, text="FPS proc.: --", fg="#888888", **self._lbl_style)
        self.lbl_fps.pack(fill="x", padx=8)
        self.lbl_status    = tk.Label(stats_lf, text="Estado: Cargando modelo...",
                                       fg="#ffaa00", bg=BG, font=("Segoe UI", 8))
        self.lbl_status.pack(fill="x", padx=8, pady=(0, 5))

        # ── Sección: Controles ────────────────────────────────────────────
        btn_lf = tk.LabelFrame(inner, text="Controles", bg=BG, fg="white",
                                font=("Segoe UI", 9, "bold"))
        btn_lf.pack(fill="x", padx=5, pady=(0, 4))

        BW = 24  # ancho de botones en caracteres

        self.btn_select = tk.Button(btn_lf, text="📂  Seleccionar video",
                                     command=self._select_video,
                                     bg="#336699", fg="white", activebackground="#4488bb",
                                     width=BW, **btn_kw)
        self.btn_select.pack(padx=8, pady=(5, 1))

        self.lbl_file = tk.Label(btn_lf, text="Ningún video seleccionado",
                                  bg=BG, fg="#666666", font=("Segoe UI", 8),
                                  wraplength=PANEL_W - 30, justify="center")
        self.lbl_file.pack(padx=8, pady=(0, 3))

        self.btn_start = tk.Button(btn_lf, text="▶  Analizar",
                                    command=self._start,
                                    bg="#228833", fg="white", activebackground="#33aa44",
                                    width=BW, **btn_kw)
        self.btn_start.pack(padx=8, pady=1)

        self.btn_pause = tk.Button(btn_lf, text="⏸  Pausar",
                                    command=self._toggle_pause, state="disabled",
                                    bg="#886622", fg="white", activebackground="#aa8833",
                                    width=BW, **btn_kw)
        self.btn_pause.pack(padx=8, pady=1)

        self.btn_stop = tk.Button(btn_lf, text="⏹  Detener",
                                   command=self._stop, state="disabled",
                                   bg="#aa3333", fg="white", activebackground="#cc4444",
                                   width=BW, **btn_kw)
        self.btn_stop.pack(padx=8, pady=1)

        self.btn_reset = tk.Button(btn_lf, text="↺  Reiniciar contadores",
                                    command=self._reset_counters,
                                    bg="#555555", fg="white", activebackground="#777777",
                                    width=BW, **btn_kw)
        self.btn_reset.pack(padx=8, pady=1)

        self.btn_export = tk.Button(btn_lf, text="📁  Exportar CSV",
                                     command=self._export_csv,
                                     bg="#555555", fg="white", activebackground="#777777",
                                     width=BW, **btn_kw)
        self.btn_export.pack(padx=8, pady=(1, 6))

        # ── Sección: Configuración ────────────────────────────────────────
        cfg_lf = tk.LabelFrame(inner, text="Configuración", bg=BG, fg="white",
                                font=("Segoe UI", 9, "bold"))
        cfg_lf.pack(fill="x", padx=5, pady=(0, 4))

        # Modo de conteo
        tk.Label(cfg_lf, text="Modo de conteo:", **cfg_lbl).pack(fill="x", padx=8, pady=(5, 0))
        self.mode_var = tk.StringVar(value="line")
        
        mode_f1 = tk.Frame(cfg_lf, bg=BG)
        mode_f1.pack(fill="x", padx=8, pady=(2, 0))
        tk.Radiobutton(mode_f1, text="—  Cruce de línea", variable=self.mode_var,
                       value="line", command=self._on_mode_change, **radio_kw
                       ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(mode_f1, text="▭  Zona rectangular", variable=self.mode_var,
                       value="roi", command=self._on_mode_change, **radio_kw
                       ).pack(side="left")
        
        mode_f2 = tk.Frame(cfg_lf, bg=BG)
        mode_f2.pack(fill="x", padx=8, pady=(2, 4))
        tk.Radiobutton(mode_f2, text="□  Campo de visión", variable=self.mode_var,
                       value="fov", command=self._on_mode_change, **radio_kw
                       ).pack(side="left")

        tk.Frame(cfg_lf, bg="#444444", height=1).pack(fill="x", padx=8, pady=(0, 4))
        
        # ── Configuración específica por modo ──
        
        # Configuración de LÍNEAS (solo visible en modo "line")
        self._line_config_frame = tk.Frame(cfg_lf, bg=BG)
        self._line_config_frame.pack(fill="x")

        # Modelo YOLO
        tk.Label(cfg_lf, text="Modelo YOLO:", **cfg_lbl).pack(fill="x", padx=8, pady=(2, 0))
        self.model_var = tk.StringVar(value="yolov8s.pt")
        model_combo = ttk.Combobox(cfg_lf, textvariable=self.model_var,
                                    values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                                    width=14, state="readonly")
        model_combo.pack(padx=8, anchor="w", pady=(1, 0))
        tk.Label(cfg_lf, text="n=rápido  s=normal  m=preciso",
                 bg=BG, fg="#666666", font=("Segoe UI", 7)).pack(anchor="w", padx=10)

        tk.Frame(cfg_lf, bg="#444444", height=1).pack(fill="x", padx=8, pady=(4, 4))

        # Confianza
        tk.Label(cfg_lf, text="Confianza:", **cfg_lbl).pack(fill="x", padx=8)
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.slider_conf = tk.Scale(cfg_lf, from_=0.1, to=0.9, resolution=0.05,
                                     variable=self.confidence_var,
                                     command=self._on_confidence_change, **slider_kw)
        self.slider_conf.pack(padx=8)

        # Líneas de conteo
        tk.Label(self._line_config_frame, text="Líneas de conteo:", **cfg_lbl).pack(fill="x", padx=8, pady=(4, 0))
        lines_f = tk.Frame(self._line_config_frame, bg=BG)
        lines_f.pack(fill="x", padx=8, pady=(2, 4))
        self.use_h_line_var = tk.BooleanVar(value=True)
        self.use_v_line_var = tk.BooleanVar(value=False)
        self._cb_h_line = tk.Checkbutton(lines_f, text="Horizontal",
                                          variable=self.use_h_line_var,
                                          command=self._on_lines_enabled_change,
                                          bg=BG, fg="white", selectcolor="#444444",
                                          font=("Segoe UI", 9),
                                          activebackground=BG, activeforeground="white")
        self._cb_h_line.pack(side="left", padx=(0, 10))
        self._cb_v_line = tk.Checkbutton(lines_f, text="Vertical",
                                          variable=self.use_v_line_var,
                                          command=self._on_lines_enabled_change,
                                          bg=BG, fg="white", selectcolor="#444444",
                                          font=("Segoe UI", 9),
                                          activebackground=BG, activeforeground="white")
        self._cb_v_line.pack(side="left")
        
        # Posición línea horizontal
        self._lbl_line_pos = tk.Label(self._line_config_frame, text="Posición línea horizontal:", **cfg_lbl)
        self._lbl_line_pos.pack(fill="x", padx=8, pady=(4, 0))
        self.line_pos_var = tk.DoubleVar(value=0.7)
        self.slider_line = tk.Scale(self._line_config_frame, from_=0.0, to=1.0, resolution=0.05,
                                     variable=self.line_pos_var,
                                     command=self._on_line_change, **slider_kw)
        self.slider_line.pack(padx=8)
        
        # Posición línea vertical
        self._lbl_line_pos_v = tk.Label(self._line_config_frame, text="Posición línea vertical:", **cfg_lbl)
        self._lbl_line_pos_v.pack(fill="x", padx=8, pady=(4, 0))
        self.line_pos_v_var = tk.DoubleVar(value=0.5)
        self.slider_line_v = tk.Scale(self._line_config_frame, from_=0.0, to=1.0, resolution=0.05,
                                       variable=self.line_pos_v_var,
                                       command=self._on_line_v_change, **slider_kw)
        self.slider_line_v.pack(padx=8)

        tk.Frame(cfg_lf, bg="#444444", height=1).pack(fill="x", padx=8, pady=(2, 4))

        # Skip de frames
        self._lbl_skip = tk.Label(cfg_lf, text="Skip de frames: 1 (sin skip)", **cfg_lbl)
        self._lbl_skip.pack(fill="x", padx=8)
        self.frame_skip_var = tk.IntVar(value=1)
        tk.Scale(cfg_lf, from_=1, to=8, resolution=1,
                 variable=self.frame_skip_var,
                 command=self._on_skip_change, **slider_kw).pack(padx=8)

        # Batch size
        self._lbl_batch = tk.Label(cfg_lf, text="Batch size: 1", **cfg_lbl)
        self._lbl_batch.pack(fill="x", padx=8, pady=(4, 0))
        self.batch_size_var = tk.IntVar(value=1)
        tk.Scale(cfg_lf, from_=1, to=8, resolution=1,
                 variable=self.batch_size_var,
                 command=self._on_batch_change, **slider_kw).pack(padx=8)

        # Resolución de inferencia
        self._lbl_infer = tk.Label(cfg_lf, text="Resolución inferencia: 640", **cfg_lbl)
        self._lbl_infer.pack(fill="x", padx=8, pady=(4, 0))
        self.infer_size_var = tk.IntVar(value=640)
        tk.Scale(cfg_lf, from_=320, to=1280, resolution=160,
                 variable=self.infer_size_var,
                 command=self._on_infer_size_change, **slider_kw).pack(padx=8, pady=(0, 6))

        # ── Sección: Gráfico tráfico por hora ─────────────────────────────
        chart_lf = tk.LabelFrame(inner, text="Tráfico por hora", bg=BG, fg="white",
                                  font=("Segoe UI", 9, "bold"))
        chart_lf.pack(fill="x", padx=5, pady=(0, 4))

        self.chart_canvas = tk.Canvas(chart_lf, bg="#1e1e1e", highlightthickness=0,
                                       height=120)
        self.chart_canvas.pack(fill="x", padx=5, pady=5)

    # -----------------------------------------------------------------------
    # Carga asincrona del modelo
    # -----------------------------------------------------------------------
    def _load_model_async(self):
        try:
            self.counter.load_model()
            self._model_loaded = True
            self.after(0, lambda: self.lbl_status.configure(
                text="Estado: Modelo listo. Seleccione un video.", fg="#00ff88",
            ))
        except Exception as e:
            logger.error("Error cargando modelo: %s", e)
            self.after(0, lambda: self.lbl_status.configure(
                text=f"Error modelo: {e}", fg="#ff4444",
            ))

    # -----------------------------------------------------------------------
    # Seleccion de video
    # -----------------------------------------------------------------------
    def _select_video(self):
        """Abre dialogo para seleccionar un archivo de video."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if filepath:
            self._video_path = filepath
            filename = os.path.basename(filepath)
            # Mostrar nombre del archivo (truncado si es muy largo)
            display_name = filename if len(filename) <= 30 else f"...{filename[-27:]}"
            self.lbl_file.configure(text=display_name, fg="#88ccff")
            self.lbl_status.configure(
                text="Video seleccionado. Listo para analizar.", fg="#00ff88"
            )
            # Previsualizar primer frame
            self._preview_first_frame(filepath)
            logger.info("Video seleccionado: %s", filepath)

    def _preview_first_frame(self, filepath: str):
        """Muestra el primer frame del video en el canvas como previsualización."""
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display = cv2.resize(frame_rgb, (self.DISPLAY_W, self.DISPLAY_H))
                img = Image.fromarray(display)
                self._photo_image = ImageTk.PhotoImage(image=img)
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.video_canvas.create_image(
                        0, 0, anchor="nw", image=self._photo_image
                    )
                else:
                    self.video_canvas.itemconfig(self._canvas_image_id, image=self._photo_image)

            # Calcular duracion
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration_sec = int(total_frames / fps)
            self._video_dur_str = self._seconds_to_mmss(duration_sec)
            self.lbl_time.configure(text=f"00:00 / {self._video_dur_str}")
        cap.release()

    # -----------------------------------------------------------------------
    # Controles
    # -----------------------------------------------------------------------
    def _start(self):
        """Inicia el análisis del video seleccionado."""
        if not self._model_loaded:
            messagebox.showwarning("Modelo", "El modelo aún se está cargando. Espere.")
            return
        if not self._video_path:
            messagebox.showwarning("Video", "Primero seleccione un archivo de video.")
            return
        if not os.path.exists(self._video_path):
            messagebox.showerror("Error", "El archivo de video no existe o fue movido.")
            return
        if self._running:
            return

        # Recargar modelo si cambió la selección
        selected_model = self.model_var.get()
        if selected_model != self.counter.model_name:
            self.lbl_status.configure(text="Cargando modelo...", fg="#ffaa00")
            self.update()
            self.counter.model_name = selected_model
            try:
                self.counter.load_model()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
                return

        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._pause_event.clear()
        self._video_progress = 0.0

        # Limpiar cola del pipeline
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="normal", text="⏸  Pausar")
        self.btn_select.configure(state="disabled")
        self.lbl_status.configure(text="Estado: Analizando...", fg="#00ff88")

        # Hilo lector (I/O puro, sin inferencia)
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Hilo procesador (YOLO + tracking)
        self._video_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self._video_thread.start()

        self._update_gui()
        logger.info("Análisis iniciado: %s", self._video_path)

    def _stop(self):
        """Detiene el análisis."""
        if not self._running:
            return
        self._running = False
        self._paused = False
        self._pause_event.clear()
        self._stop_event.set()
        # Desbloquear la cola por si el reader está bloqueado en put()
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._canvas_image_id = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_pause.configure(state="disabled", text="⏸  Pausar")
        self.btn_select.configure(state="normal")
        self.lbl_status.configure(text="Estado: Detenido", fg="#ffaa00")
        self._draw_placeholder("Análisis detenido")
        logger.info("Análisis detenido por el usuario.")

    def _toggle_pause(self):
        """Alterna entre pausar y reanudar el análisis."""
        if not self._running:
            return
        if self._paused:
            # Reanudar
            self._paused = False
            self._pause_event.clear()
            self.btn_pause.configure(text="⏸  Pausar")
            self.lbl_status.configure(text="Estado: Analizando...", fg="#00ff88")
        else:
            # Pausar
            self._paused = True
            self._pause_event.set()
            self.btn_pause.configure(text="▶  Reanudar")
            self.lbl_status.configure(text="Estado: Pausado", fg="#ffaa00")

    def _reset_counters(self):
        if messagebox.askyesno("Reiniciar", "¿Reiniciar todos los contadores a cero?"):
            self.counter.reset_counters()
            self._update_stats_labels()
            self._draw_chart()
            logger.info("Contadores reiniciados por el usuario.")

    def _export_csv(self):
        default_name = f"conteo_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=default_name,
            title="Guardar conteo como CSV",
        )
        if filepath:
            try:
                self.counter.export_csv(filepath)
                messagebox.showinfo("Exportado", f"Datos guardados en:\n{filepath}")
            except Exception as e:
                logger.error("Error exportando CSV: %s", e)
                messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    def _on_confidence_change(self, _val=None):
        self.counter.update_confidence(self.confidence_var.get())

    def _on_line_change(self, _val=None):
        self.counter.update_line(self.line_pos_var.get())
    
    def _on_line_v_change(self, _val=None):
        self.counter.update_line(self.line_pos_var.get(), position_vertical=self.line_pos_v_var.get())
    
    def _on_roi_change(self, _val=None):
        self.counter.set_roi(
            self.roi_x1_var.get(),
            self.roi_y1_var.get(),
            self.roi_x2_var.get(),
            self.roi_y2_var.get()
        )
    
    def _on_lines_enabled_change(self):
        h_enabled = self.use_h_line_var.get()
        v_enabled = self.use_v_line_var.get()
        self.counter.set_line_enabled(horizontal=h_enabled, vertical=v_enabled)
        
        # Habilitar/deshabilitar sliders según checkboxes
        if h_enabled:
            self.slider_line.configure(state="normal", troughcolor="#555555")
            self._lbl_line_pos.configure(fg="#cccccc")
        else:
            self.slider_line.configure(state="disabled", troughcolor="#333333")
            self._lbl_line_pos.configure(fg="#555555")
            
        if v_enabled:
            self.slider_line_v.configure(state="normal", troughcolor="#555555")
            self._lbl_line_pos_v.configure(fg="#cccccc")
        else:
            self.slider_line_v.configure(state="disabled", troughcolor="#333333")
            self._lbl_line_pos_v.configure(fg="#555555")

    def _on_skip_change(self, _val=None):
        skip = self.frame_skip_var.get()
        self.counter.frame_skip = skip
        label = f"Skip de frames: {skip}" + (" (sin skip)" if skip == 1 else f" (~{skip}x más rápido)")
        self._lbl_skip.configure(text=label)

    def _on_batch_change(self, _val=None):
        batch = self.batch_size_var.get()
        self.counter.batch_size = batch
        self._lbl_batch.configure(text=f"Batch size: {batch}")

    def _on_infer_size_change(self, _val=None):
        size = self.infer_size_var.get()
        self.counter.infer_size = size
        self._lbl_infer.configure(text=f"Resolución inferencia: {size}")

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.counter.set_mode(mode)
        
        # Ocultar todos los frames de configuración
        self._line_config_frame.pack_forget()
        self._roi_config_frame.pack_forget()

        if mode == "fov":
            # Modo campo de visión - sin configuración adicional
            self.lbl_entradas.configure(text="PERSONAS:  0", fg="#44bbff")
            self.lbl_salidas.configure(text="SALIDAS:   N/A", fg="#555555")
        elif mode == "roi":
            # Modo zona rectangular - mostrar configuración ROI
            self._roi_config_frame.pack(fill="x")
            self.lbl_entradas.configure(text="ENTRADAS:  0", fg="#00ff88")
            self.lbl_salidas.configure(text="SALIDAS:   0", fg="#ff6644")
        else:
            # Modo línea - mostrar configuración de líneas
            self._line_config_frame.pack(fill="x")
            self.lbl_entradas.configure(text="ENTRADAS:  0", fg="#00ff88")
            self.lbl_salidas.configure(text="SALIDAS:   0", fg="#ff6644")
            # Los sliders se habilitan según los checkboxes
            self._on_lines_enabled_change()

    # -----------------------------------------------------------------------
    # Hilo de procesamiento de video
    # -----------------------------------------------------------------------
    def _reader_loop(self):
        """Hilo lector: lee frames del video y los pone en la cola del pipeline.

        Solo hace I/O (cv2.read). No hace inferencia. Esto desacopla la lectura
        del disco/decodificacion del procesamiento GPU, eliminando el tiempo de
        espera entre frames.
        """
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            logger.error("Reader: no se pudo abrir %s", self._video_path)
            self._frame_queue.put(None)  # Senial de fin
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.counter.setup_line(w, h)
        self._video_total_frames = total_frames
        self._video_dur_str = self._seconds_to_mmss(int(total_frames / video_fps))

        logger.info("Reader: %dx%d, %.1f fps, %d frames", w, h, video_fps, total_frames)

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break  # Fin del video

                frame_idx += 1

                # Skip de frames: descartar frames intermedios para ir mas rapido.
                # El tracker ByteTrack esta disenado para tolerar frames omitidos.
                if (frame_idx - 1) % self.counter.frame_skip != 0:
                    # NO actualizar progreso aquí - solo se actualiza con frames procesados
                    continue

                # Encolar (bloquea si la cola esta llena, aplicando backpressure)
                self._frame_queue.put((frame_idx, frame, video_fps, total_frames))

        except Exception as e:
            logger.error("Reader: error en frame %d: %s", frame_idx, e, exc_info=True)
        finally:
            cap.release()
            self._frame_queue.put(None)  # Senial de fin al procesador
            logger.info("Reader: finalizado (%d frames leidos).", frame_idx)

    def _processor_loop(self):
        """Hilo procesador: consume frames de la cola, corre YOLO en batch y anota.

        Agrupa hasta batch_size frames antes de llamar a YOLO, lo que es mas
        eficiente en GPU porque amortiza el overhead de cada llamada.
        """
        batch_frames = []
        batch_meta = []  # (frame_idx, video_fps, total_frames) por frame

        try:
            while not self._stop_event.is_set():
                # Intentar llenar el batch
                try:
                    item = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    # Fin del video: procesar lo que quede en el batch parcial
                    if batch_frames:
                        self._run_batch(batch_frames, batch_meta)
                    break

                frame_idx, frame, video_fps, total_frames = item
                batch_frames.append(frame)
                batch_meta.append((frame_idx, video_fps, total_frames))

                # Procesar batch solo cuando este lleno para mantener sincronización
                if len(batch_frames) >= self.counter.batch_size:
                    self._run_batch(batch_frames, batch_meta)
                    batch_frames = []
                    batch_meta = []

        except Exception as e:
            logger.error("Processor: error: %s", e, exc_info=True)
        finally:
            logger.info("Processor: finalizado.")
            self._set_status_threadsafe("✓ Video procesado completamente", "#00ff88")
            self.after(0, self._on_video_finished)

    def _run_batch(self, frames: list, meta: list):
        """Ejecuta YOLO sobre un batch de frames y actualiza cada frame mostrado en GUI."""
        try:
            annotated_list = self.counter.process_batch(frames)
        except Exception as e:
            logger.error("Processor: error en batch: %s", e, exc_info=True)
            annotated_list = frames  # Mostrar frame sin anotar si hay error

        # Actualizar y mostrar cada frame del batch secuencialmente
        # Esto evita saltos visuales mostrando todos los frames procesados
        for i, (display_frame, (frame_idx, video_fps, total_frames)) in enumerate(zip(annotated_list, meta)):
            # Actualizar progreso para este frame específico
            self._video_current_frame = frame_idx
            self._video_progress = frame_idx / total_frames if total_frames > 0 else 0
            self._video_pos_str = self._seconds_to_mmss(int(frame_idx / video_fps))
            
            # Mostrar el frame
            display = cv2.resize(display_frame, (self.DISPLAY_W, self.DISPLAY_H))
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._current_frame = display
            
            # Pequeña pausa entre frames para que la GUI los muestre suavemente
            # Solo si no es el último frame del batch
            if i < len(annotated_list) - 1:
                time.sleep(0.01)  # 10ms entre frames del mismo batch

    def _on_video_finished(self):
        """Llamado cuando el video termina de procesarse (éxito o error)."""
        self._running = False
        self._paused = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_pause.configure(state="disabled", text="⏸  Pausar")
        self.btn_select.configure(state="normal")

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------
    @staticmethod
    def _seconds_to_mmss(seconds: int) -> str:
        """Convierte segundos a formato MM:SS."""
        m, s = divmod(max(0, seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _set_status_threadsafe(self, text: str, color: str):
        try:
            self.after(0, lambda: self.lbl_status.configure(text=text, fg=color))
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Actualizacion de GUI (hilo principal)
    # -----------------------------------------------------------------------
    def _update_gui(self):
        """Actualiza video, progreso y stats en el hilo principal de Tkinter."""
        if not self._running:
            return

        # Frame de video
        with self._lock:
            frame = self._current_frame

        if frame is not None:
            try:
                img = Image.fromarray(frame)
                self._photo_image = ImageTk.PhotoImage(image=img)
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.video_canvas.create_image(
                        0, 0, anchor="nw", image=self._photo_image
                    )
                else:
                    self.video_canvas.itemconfig(self._canvas_image_id, image=self._photo_image)
            except Exception:
                pass

        # Barra de progreso y tiempo
        progress_pct = self._video_progress * 100
        self.progress_bar["value"] = progress_pct
        self.lbl_time.configure(
            text=f"{self._video_pos_str} / {self._video_dur_str}"
        )

        # Estadísticas
        self._update_stats_labels()

        # Gráfico
        self._draw_chart()

        self.after(self.UPDATE_MS, self._update_gui)

    def _update_stats_labels(self):
        c = self.counter
        if c.counting_mode == "fov":
            self.lbl_entradas.configure(text=f"PERSONAS:  {c.fov_count}")
            self.lbl_salidas.configure(text="SALIDAS:   N/A")
        else:
            self.lbl_entradas.configure(text=f"ENTRADAS:  {c.in_count}")
            self.lbl_salidas.configure(text=f"SALIDAS:   {c.out_count}")
        self.lbl_en_cuadro.configure(text=f"EN CUADRO:  {c.persons_in_frame}")
        self.lbl_fps.configure(text=f"FPS proc.: {c.fps:.1f}")

    def _draw_placeholder(self, text: str):
        self.video_canvas.delete("all")
        self._canvas_image_id = None
        self.video_canvas.create_text(
            self.DISPLAY_W // 2, self.DISPLAY_H // 2,
            text=text, fill="#888888", font=("Segoe UI", 14),
        )

    # -----------------------------------------------------------------------
    # Grafico de trafico por hora
    # -----------------------------------------------------------------------
    def _draw_chart(self):
        canvas = self.chart_canvas
        canvas.delete("all")

        data = self.counter.hourly_entries
        if not data:
            canvas.create_text(
                canvas.winfo_width() // 2 or 110, 50,
                text="Sin datos aún", fill="#666666", font=("Segoe UI", 9),
            )
            return

        cw = canvas.winfo_width() or 220
        ch = canvas.winfo_height() or 100
        margin_bottom = 20
        margin_top = 10
        chart_h = ch - margin_bottom - margin_top

        min_hour = min(data.keys())
        max_hour = max(data.keys())
        hours = list(range(min_hour, max_hour + 1))
        if not hours:
            return

        max_val = max(data.values()) if data.values() else 1
        bar_w = max(8, (cw - 20) // max(len(hours), 1) - 4)

        for i, h in enumerate(hours):
            val = data.get(h, 0)
            bar_h = int((val / max_val) * chart_h) if max_val > 0 else 0
            x = 10 + i * (bar_w + 4)
            y_top = margin_top + chart_h - bar_h
            y_bottom = margin_top + chart_h

            canvas.create_rectangle(
                x, y_top, x + bar_w, y_bottom,
                fill="#22aa55", outline="#33cc66",
            )

            if val > 0:
                canvas.create_text(
                    x + bar_w // 2, y_top - 6,
                    text=str(val), fill="#aaaaaa", font=("Consolas", 7),
                )

            canvas.create_text(
                x + bar_w // 2, y_bottom + 10,
                text=f"{h}h", fill="#888888", font=("Consolas", 7),
            )

    # -----------------------------------------------------------------------
    # Auto-guardado y cierre
    # -----------------------------------------------------------------------
    def _auto_save(self):
        try:
            if self.counter.in_count > 0 or self.counter.fov_count > 0:
                filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self._csv_dir, filename)
                self.counter.export_csv(filepath)
                logger.info("Auto-guardado: %s", filepath)
        except Exception as e:
            logger.error("Error en auto-guardado: %s", e)
        self.after(self._autosave_interval, self._auto_save)

    def _on_close(self):
        logger.info("Cerrando aplicacion...")
        self._running = False
        self._stop_event.set()
        self._pause_event.clear()
        # Desbloquear cola para que los hilos puedan terminar
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            if self.counter.in_count > 0 or self.counter.fov_count > 0:
                filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self._csv_dir, filename)
                self.counter.export_csv(filepath)
                logger.info("Datos guardados al cerrar: %s", filepath)
        except Exception as e:
            logger.error("Error guardando al cerrar: %s", e)

        for t in (self._reader_thread, self._video_thread):
            if t is not None and t.is_alive():
                t.join(timeout=3)

        self.destroy()
        logger.info("Aplicacion cerrada.")


# ===========================================================================
# Punto de entrada
# ===========================================================================
def main():
    app = CounterApp()
    app.mainloop()


if __name__ == "__main__":
    main()