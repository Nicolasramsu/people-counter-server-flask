"""
Contador de Personas - Stand Feria
===================================
Aplicacion con GUI Tkinter para conteo de personas en tiempo real.
Usa YOLO + ByteTrack + LineZone para detectar y contar cruces de linea.
Disenado para sesiones prolongadas (8-10 horas) con reconexion automatica,
auto-guardado CSV y reset periodico del tracker.
"""

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
import shutil
from datetime import datetime, timedelta
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
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.line_position = line_position
        self.line_orientation = line_orientation

        # Modo de conteo: "line" (cruce de linea) | "fov" (campo de vision)
        self.counting_mode = "line"

        # Estadisticas modo linea
        self.in_count = 0
        self.out_count = 0
        self._in_offset = 0   # Acumulado de LineZones anteriores
        self._out_offset = 0
        self.persons_in_frame = 0
        self.fps = 0.0

        # Estadisticas modo campo de vision (FOV)
        self.fov_count = 0          # Personas unicas vistas desde el inicio/reset
        self._fov_offset = 0        # Acumulado antes de resets del tracker
        self._seen_ids = set()      # IDs vistos en el ciclo actual del tracker

        # Datos por intervalo (cada 15 min) - compartido entre modos
        # Formato: (hora_str, val_intervalo, val2_intervalo, total, total2)
        #   linea: (hora, in_intervalo, out_intervalo, in_total, out_total)
        #   fov:   (hora, personas_intervalo, 0, personas_total, 0)
        self.interval_data = []
        self._last_interval_time = None
        self._interval_in_start = 0
        self._interval_out_start = 0
        self._interval_fov_start = 0

        # Datos por hora para el grafico
        self.hourly_entries = {}  # {hora_int: entradas}

        # Internos
        self.model = None
        self.tracker = None
        self.line_zone = None
        self.frame_width = 640
        self.frame_height = 480
        self._fps_buffer = deque(maxlen=30)
        self._tracker_reset_time = time.time()
        self._tracker_reset_interval = 30 * 60  # 30 minutos

        # Anotadores
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

    def load_model(self):
        """Carga el modelo YOLO."""
        logger.info("Cargando modelo YOLO: %s", self.model_name)
        self.model = YOLO(self.model_name)
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
        """Crea la zona de linea segun la orientacion y posicion."""
        if self.line_orientation == "horizontal":
            y = int(self.frame_height * self.line_position)
            start = sv.Point(0, y)
            end = sv.Point(self.frame_width, y)
        else:
            x = int(self.frame_width * self.line_position)
            start = sv.Point(x, 0)
            end = sv.Point(x, self.frame_height)
        # Guardar conteos actuales como offset antes de recrear
        if self.line_zone is not None:
            self._in_offset += self.line_zone.in_count
            self._out_offset += self.line_zone.out_count
        # Usar solo el centroide como punto de cruce.
        # El default (4 esquinas del bounding box) dispara el conteo cuando
        # cualquier esquina toca la linea, lo que con camaras cercanas hace
        # que se cuente antes de que la persona haya realmente pasado.
        self.line_zone = sv.LineZone(
            start=start,
            end=end,
            triggering_anchors=[sv.Position.CENTER],
        )

    def _create_tracker(self):
        """Crea un nuevo tracker ByteTrack."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.confidence,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self._tracker_reset_time = time.time()

    def _periodic_tracker_reset(self):
        """Reset automatico cada 30 min para liberar memoria.

        En modo FOV, antes de reiniciar el tracker se preserva el conteo de
        IDs unicos vistos en este ciclo sumandolos al offset acumulado.
        Los IDs en ByteTrack se reinician desde 1, por lo que sin este
        mecanismo se perderia la cuenta entre ciclos.
        """
        logger.info("Reset periodico del tracker (cada 30 min).")
        if self.counting_mode == "fov":
            self._fov_offset += len(self._seen_ids)
            self._seen_ids = set()
        self._create_tracker()

    def set_mode(self, mode: str):
        """Cambia el modo de conteo sin resetear los acumulados."""
        self.counting_mode = mode
        logger.info("Modo de conteo cambiado a: %s", mode)

    def update_line(self, position: float, orientation: str):
        """Actualiza la posicion/orientacion de la linea en vivo."""
        self.line_position = position
        self.line_orientation = orientation
        self._create_line_zone()

    def update_confidence(self, confidence: float):
        """Actualiza el umbral de confianza."""
        self.confidence = confidence

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame: deteccion, tracking, conteo y anotacion."""
        t_start = time.perf_counter()

        if self.model is None:
            return frame

        # Reset periodico del tracker para liberar memoria
        if time.time() - self._tracker_reset_time > self._tracker_reset_interval:
            self._periodic_tracker_reset()

        # Deteccion YOLO
        results = self.model(
            frame, classes=[PERSON_CLASS_ID], conf=self.confidence, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Tracking
        detections = self.tracker.update_with_detections(detections)

        # --- Conteo segun el modo activo ---
        if self.counting_mode == "fov":
            self._update_fov_count(detections)
        else:
            self.line_zone.trigger(detections=detections)
            self.in_count = self._in_offset + self.line_zone.in_count
            self.out_count = self._out_offset + self.line_zone.out_count

        self.persons_in_frame = len(detections)

        # Registrar intervalo cada 15 min
        self._check_interval()

        # Etiquetas de tracker ID
        labels = []
        if detections.tracker_id is not None:
            labels = [
                f"#{tid} {conf:.0%}"
                for tid, conf in zip(detections.tracker_id, detections.confidence)
            ]

        # Anotaciones comunes
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        if labels:
            frame = self.label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        # Linea solo en modo linea
        if self.counting_mode == "line":
            frame = self.line_annotator.annotate(frame=frame, line_counter=self.line_zone)

        # Centroide visible en ambos modos
        if len(detections) > 0 and detections.xyxy is not None:
            for box in detections.xyxy:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (0, 180, 180), 1)

        # FPS
        elapsed = time.perf_counter() - t_start
        self._fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
        self.fps = sum(self._fps_buffer) / len(self._fps_buffer)

        return frame

    def _update_fov_count(self, detections):
        """Registra IDs nuevos y actualiza fov_count (modo campo de vision).

        Cada ID de tracker que aparece por primera vez desde el ultimo reset
        representa una persona unica detectada. El conteo es monotonicamente
        creciente: nunca baja aunque la persona salga del cuadro.
        """
        if detections.tracker_id is not None:
            for tid in detections.tracker_id:
                self._seen_ids.add(tid)
        self.fov_count = self._fov_offset + len(self._seen_ids)

    def _check_interval(self):
        """Registra datos cada 15 minutos."""
        now = time.time()
        if self._last_interval_time is None:
            self._last_interval_time = now
            return
        if now - self._last_interval_time >= 15 * 60:
            hora_str = datetime.now().strftime("%H:%M")
            hora = datetime.now().hour

            if self.counting_mode == "fov":
                fov_interval = self.fov_count - self._interval_fov_start
                self.interval_data.append(
                    (hora_str, fov_interval, 0, self.fov_count, 0)
                )
                self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + fov_interval
                self._interval_fov_start = self.fov_count
                logger.info("Intervalo FOV: %s | Personas: %d", hora_str, fov_interval)
            else:
                in_interval = self.in_count - self._interval_in_start
                out_interval = self.out_count - self._interval_out_start
                self.interval_data.append(
                    (hora_str, in_interval, out_interval, self.in_count, self.out_count)
                )
                self.hourly_entries[hora] = self.hourly_entries.get(hora, 0) + in_interval
                self._interval_in_start = self.in_count
                self._interval_out_start = self.out_count
                logger.info(
                    "Intervalo linea: %s | Entradas: %d | Salidas: %d",
                    hora_str, in_interval, out_interval,
                )

            self._last_interval_time = now

    def reset_counters(self):
        """Reinicia todos los contadores y el tracker."""
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
        # Forzar line_zone a None para que _create_line_zone no acumule offset
        self.line_zone = None
        self._create_line_zone()
        self._create_tracker()
        logger.info("Contadores reiniciados.")

    def export_csv(self, filepath: str):
        """Exporta los datos a un archivo CSV.

        El esquema de columnas varia segun el modo activo:
          - linea: hora | entradas_intervalo | salidas_intervalo | entradas_total | salidas_total
          - fov:   hora | personas_intervalo | personas_total
        """
        data = list(self.interval_data)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if self.counting_mode == "fov":
                # Agregar intervalo parcial final
                fov_partial = self.fov_count - self._interval_fov_start
                if fov_partial > 0:
                    hora_str = datetime.now().strftime("%H:%M")
                    data.append((hora_str, fov_partial, 0, self.fov_count, 0))

                writer.writerow(["hora", "personas_intervalo", "personas_total"])
                for row in data:
                    writer.writerow([row[0], row[1], row[3]])
            else:
                # Agregar intervalo parcial final
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
        self.title("Contador de Personas - Stand Feria")
        self.resizable(False, False)
        self.configure(bg="#2b2b2b")

        # Estado
        self._running = False
        self._stop_event = threading.Event()
        self._video_thread = None
        self._cap = None
        self._camera_index = 0
        self._lock = threading.Lock()
        self._current_frame = None  # Frame procesado (numpy)
        self._photo_image = None  # ImageTk para mostrar

        # Motor de deteccion
        self.counter = PersonCounter()

        # Canvas image item (evita parpadeo al usar itemconfig en vez de delete+create)
        self._canvas_image_id = None

        # Auto-guardado
        self._autosave_interval = 5 * 60 * 1000  # 5 minutos en ms
        self._csv_dir = os.path.dirname(os.path.abspath(__file__))

        # Construir GUI
        self._build_gui()

        # Cargar modelo en hilo para no bloquear la GUI
        self._model_loaded = False
        threading.Thread(target=self._load_model_async, daemon=True).start()

        # Auto-guardado periodico
        self.after(self._autosave_interval, self._auto_save)

        # Cierre limpio
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Aplicacion iniciada.")

    # -----------------------------------------------------------------------
    # Construccion de la GUI
    # -----------------------------------------------------------------------
    def _build_gui(self):
        """Construye todo el layout de la ventana."""
        main_frame = tk.Frame(self, bg="#2b2b2b")
        main_frame.pack(padx=5, pady=5)

        # --- Panel izquierdo: video ---
        self.video_canvas = tk.Canvas(
            main_frame, width=self.DISPLAY_W, height=self.DISPLAY_H,
            bg="black", highlightthickness=0,
        )
        self.video_canvas.grid(row=0, column=0, padx=(0, 5))
        self._draw_placeholder("Presione 'Iniciar' para comenzar")

        # --- Panel derecho: controles ---
        right_panel = tk.Frame(main_frame, bg="#2b2b2b", width=250)
        right_panel.grid(row=0, column=1, sticky="ns")
        right_panel.grid_propagate(False)
        right_panel.configure(width=250)

        # Estadisticas
        stats_frame = tk.LabelFrame(
            right_panel, text="Estadisticas", bg="#2b2b2b", fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        stats_frame.pack(fill="x", padx=5, pady=(0, 5))

        self._lbl_style = {"bg": "#2b2b2b", "font": ("Consolas", 13), "anchor": "w"}

        self.lbl_entradas = tk.Label(stats_frame, text="ENTRADAS:  0", fg="#00ff88", **self._lbl_style)
        self.lbl_entradas.pack(fill="x", padx=8, pady=(5, 0))

        self.lbl_salidas = tk.Label(stats_frame, text="SALIDAS:   0", fg="#ff6644", **self._lbl_style)
        self.lbl_salidas.pack(fill="x", padx=8)

        self.lbl_en_cuadro = tk.Label(stats_frame, text="EN CUADRO:  0", fg="#ffdd44", **self._lbl_style)
        self.lbl_en_cuadro.pack(fill="x", padx=8)

        self.lbl_fps = tk.Label(stats_frame, text="FPS: --", fg="#aaaaaa", **self._lbl_style)
        self.lbl_fps.pack(fill="x", padx=8, pady=(0, 5))

        self.lbl_status = tk.Label(
            stats_frame, text="Estado: Cargando modelo...",
            fg="#ffaa00", bg="#2b2b2b", font=("Segoe UI", 9),
        )
        self.lbl_status.pack(fill="x", padx=8, pady=(0, 5))

        # Botones de accion
        btn_frame = tk.LabelFrame(
            right_panel, text="Controles", bg="#2b2b2b", fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        btn_frame.pack(fill="x", padx=5, pady=(0, 5))

        btn_style = {"font": ("Segoe UI", 10), "width": 20, "pady": 3}

        self.btn_start = tk.Button(
            btn_frame, text="\u25B6  Iniciar", command=self._start,
            bg="#228833", fg="white", activebackground="#33aa44", **btn_style,
        )
        self.btn_start.pack(padx=8, pady=(5, 2))

        self.btn_stop = tk.Button(
            btn_frame, text="\u25A0  Detener", command=self._stop, state="disabled",
            bg="#aa3333", fg="white", activebackground="#cc4444", **btn_style,
        )
        self.btn_stop.pack(padx=8, pady=2)

        self.btn_reset = tk.Button(
            btn_frame, text="\u21BA  Reiniciar contadores", command=self._reset_counters,
            bg="#555555", fg="white", activebackground="#777777", **btn_style,
        )
        self.btn_reset.pack(padx=8, pady=2)

        self.btn_export = tk.Button(
            btn_frame, text="\U0001F4C1  Exportar CSV", command=self._export_csv,
            bg="#555555", fg="white", activebackground="#777777", **btn_style,
        )
        self.btn_export.pack(padx=8, pady=(2, 5))

        # Configuracion
        config_frame = tk.LabelFrame(
            right_panel, text="Configuracion", bg="#2b2b2b", fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        config_frame.pack(fill="x", padx=5, pady=(0, 5))

        cfg_lbl = {"bg": "#2b2b2b", "fg": "#cccccc", "font": ("Segoe UI", 9), "anchor": "w"}
        radio_style = {"bg": "#2b2b2b", "fg": "white", "selectcolor": "#444444",
                       "font": ("Segoe UI", 9), "activebackground": "#2b2b2b",
                       "activeforeground": "white"}

        # --- Modo de conteo ---
        tk.Label(config_frame, text="Modo de conteo:", **cfg_lbl).pack(fill="x", padx=8, pady=(5, 0))
        mode_frame = tk.Frame(config_frame, bg="#2b2b2b")
        mode_frame.pack(fill="x", padx=8, pady=(2, 5))
        self.mode_var = tk.StringVar(value="line")
        tk.Radiobutton(
            mode_frame, text="\u2014  Cruce de linea", variable=self.mode_var,
            value="line", command=self._on_mode_change, **radio_style,
        ).pack(side="left", padx=(0, 8))
        tk.Radiobutton(
            mode_frame, text="\u25A1  Campo de vision", variable=self.mode_var,
            value="fov", command=self._on_mode_change, **radio_style,
        ).pack(side="left")

        # Separador visual entre modo y resto de config
        tk.Frame(config_frame, bg="#444444", height=1).pack(fill="x", padx=8, pady=(0, 4))

        # Camara
        tk.Label(config_frame, text="Camara:", **cfg_lbl).pack(fill="x", padx=8, pady=(2, 0))
        self.camera_var = tk.IntVar(value=0)
        cam_combo = ttk.Combobox(
            config_frame, textvariable=self.camera_var, values=[0, 1, 2, 3, 4],
            width=5, state="readonly",
        )
        cam_combo.pack(padx=8, anchor="w")
        cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        # Slider confianza
        tk.Label(config_frame, text="Confianza:", **cfg_lbl).pack(fill="x", padx=8, pady=(5, 0))
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.slider_conf = tk.Scale(
            config_frame, from_=0.1, to=0.9, resolution=0.05, orient="horizontal",
            variable=self.confidence_var, command=self._on_confidence_change,
            bg="#2b2b2b", fg="white", highlightthickness=0, troughcolor="#555555",
            length=200,
        )
        self.slider_conf.pack(padx=8)

        # Slider posicion linea (solo relevante en modo linea)
        self._lbl_line_pos = tk.Label(config_frame, text="Posicion linea:", **cfg_lbl)
        self._lbl_line_pos.pack(fill="x", padx=8, pady=(5, 0))
        self.line_pos_var = tk.DoubleVar(value=0.7)
        self.slider_line = tk.Scale(
            config_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
            variable=self.line_pos_var, command=self._on_line_change,
            bg="#2b2b2b", fg="white", highlightthickness=0, troughcolor="#555555",
            length=200,
        )
        self.slider_line.pack(padx=8)

        # Orientacion (solo relevante en modo linea)
        self._lbl_orientation = tk.Label(config_frame, text="Orientacion linea:", **cfg_lbl)
        self._lbl_orientation.pack(fill="x", padx=8, pady=(5, 0))
        orient_frame = tk.Frame(config_frame, bg="#2b2b2b")
        orient_frame.pack(fill="x", padx=8, pady=(0, 5))
        self.orientation_var = tk.StringVar(value="horizontal")
        self._rb_horizontal = tk.Radiobutton(
            orient_frame, text="Horizontal", variable=self.orientation_var,
            value="horizontal", command=self._on_line_change, **radio_style,
        )
        self._rb_horizontal.pack(side="left", padx=(0, 10))
        self._rb_vertical = tk.Radiobutton(
            orient_frame, text="Vertical", variable=self.orientation_var,
            value="vertical", command=self._on_line_change, **radio_style,
        )
        self._rb_vertical.pack(side="left")

        # Grafico de trafico
        chart_frame = tk.LabelFrame(
            right_panel, text="Trafico por hora", bg="#2b2b2b", fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        chart_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self.chart_canvas = tk.Canvas(
            chart_frame, bg="#1e1e1e", highlightthickness=0, height=120,
        )
        self.chart_canvas.pack(fill="both", expand=True, padx=5, pady=5)

    # -----------------------------------------------------------------------
    # Carga asincrona del modelo
    # -----------------------------------------------------------------------
    def _load_model_async(self):
        """Carga el modelo YOLO en un hilo separado."""
        try:
            self.counter.load_model()
            self._model_loaded = True
            self.after(0, lambda: self.lbl_status.configure(
                text="Estado: Modelo listo", fg="#00ff88",
            ))
        except Exception as e:
            logger.error("Error cargando modelo: %s", e)
            self.after(0, lambda: self.lbl_status.configure(
                text=f"Error modelo: {e}", fg="#ff4444",
            ))

    # -----------------------------------------------------------------------
    # Controles
    # -----------------------------------------------------------------------
    def _start(self):
        """Inicia la captura de video y deteccion."""
        if not self._model_loaded:
            messagebox.showwarning("Modelo", "El modelo aun se esta cargando. Espere.")
            return
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._camera_index = self.camera_var.get()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="Estado: Ejecutando", fg="#00ff88")

        # Hilo de video
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()

        # Iniciar actualizacion de GUI
        self._update_gui()
        logger.info("Deteccion iniciada (camara %d).", self._camera_index)

    def _stop(self):
        """Detiene la deteccion."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        self._canvas_image_id = None  # Permite redibujar placeholder al reiniciar
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Estado: Detenido", fg="#ffaa00")
        self._draw_placeholder("Presione 'Iniciar' para comenzar")
        logger.info("Deteccion detenida.")

    def _reset_counters(self):
        """Reinicia los contadores."""
        if messagebox.askyesno("Reiniciar", "Reiniciar todos los contadores a cero?"):
            self.counter.reset_counters()
            self._update_stats_labels()
            self._draw_chart()
            logger.info("Contadores reiniciados por el usuario.")

    def _export_csv(self):
        """Exporta CSV con dialogo de ubicacion."""
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

    def _on_camera_change(self, _event=None):
        """Cambia de camara (requiere reiniciar captura)."""
        new_idx = self.camera_var.get()
        if new_idx != self._camera_index and self._running:
            self._stop()
            self._camera_index = new_idx
            self.after(500, self._start)

    def _on_confidence_change(self, _val=None):
        """Actualiza confianza en vivo."""
        self.counter.update_confidence(self.confidence_var.get())

    def _on_line_change(self, _val=None):
        """Actualiza posicion y orientacion de la linea en vivo."""
        self.counter.update_line(self.line_pos_var.get(), self.orientation_var.get())

    def _on_mode_change(self):
        """Cambia el modo de conteo y ajusta la GUI en consecuencia."""
        mode = self.mode_var.get()
        self.counter.set_mode(mode)

        if mode == "fov":
            # Deshabilitar controles exclusivos del modo linea
            self.slider_line.configure(state="disabled", troughcolor="#333333")
            self._rb_horizontal.configure(state="disabled")
            self._rb_vertical.configure(state="disabled")
            self._lbl_line_pos.configure(fg="#555555")
            self._lbl_orientation.configure(fg="#555555")
            # Actualizar etiqueta de estadisticas
            self.lbl_entradas.configure(text="PERSONAS:  0", fg="#44bbff")
            self.lbl_salidas.configure(text="SALIDAS:   N/A", fg="#555555")
        else:
            # Rehabilitar controles de linea
            self.slider_line.configure(state="normal", troughcolor="#555555")
            self._rb_horizontal.configure(state="normal")
            self._rb_vertical.configure(state="normal")
            self._lbl_line_pos.configure(fg="#cccccc")
            self._lbl_orientation.configure(fg="#cccccc")
            # Restaurar etiquetas originales
            self.lbl_entradas.configure(text="ENTRADAS:  0", fg="#00ff88")
            self.lbl_salidas.configure(text="SALIDAS:   0", fg="#ff6644")

    # -----------------------------------------------------------------------
    # Hilo de video
    # -----------------------------------------------------------------------
    def _video_loop(self):
        """Loop de captura y procesamiento de video (hilo daemon)."""
        cap = None
        reconnect_delay = 3  # segundos

        try:
            while not self._stop_event.is_set():
                # Abrir camara si es necesario
                if cap is None or not cap.isOpened():
                    cap = self._open_camera(self._camera_index)
                    if cap is None:
                        self._set_status_threadsafe("Camara desconectada. Reintentando...", "#ff4444")
                        self._show_disconnected_frame()
                        time.sleep(reconnect_delay)
                        continue
                    # Configurar linea con el tamano del frame
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.counter.setup_line(w, h)
                    self._set_status_threadsafe("Estado: Ejecutando", "#00ff88")

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Fallo de lectura de frame. Reconectando camara...")
                    cap.release()
                    cap = None
                    continue

                # Procesar frame
                try:
                    processed = self.counter.process_frame(frame)
                except Exception as e:
                    logger.error("Error procesando frame: %s", e, exc_info=True)
                    processed = frame

                # Redimensionar para la GUI
                display = cv2.resize(processed, (self.DISPLAY_W, self.DISPLAY_H))
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                with self._lock:
                    self._current_frame = display

        except Exception as e:
            logger.error("Error critico en hilo de video: %s", e, exc_info=True)
        finally:
            if cap is not None:
                cap.release()
            self._cap = None
            logger.info("Hilo de video finalizado.")

    def _open_camera(self, index: int):
        """Intenta abrir la camara. Devuelve None si falla.

        Prueba primero DirectShow (DSHOW) en Windows, que es mas estable para
        camaras USB como la Logitech C922. Si falla, intenta el backend por defecto.
        Ademas:
          - Fija el buffer interno a 1 frame para evitar acumulacion de frames atrasados.
          - Lee varios frames de calentamiento hasta obtener uno valido, ya que la
            C922 necesita un momento para inicializar el sensor tras ser abierta.
        """
        WARMUP_ATTEMPTS = 20
        WARMUP_DELAY   = 0.05  # segundos entre intentos

        # En Windows, DSHOW es mas confiable para webcams USB; usar como primer intento
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Reducir buffer interno para no leer frames acumulados (causa de parpadeo)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Calentamiento: esperar hasta obtener un frame valido
                warmed = False
                for attempt in range(WARMUP_ATTEMPTS):
                    ret, _ = cap.read()
                    if ret:
                        warmed = True
                        break
                    time.sleep(WARMUP_DELAY)

                if not warmed:
                    logger.warning(
                        "Camara %d (backend %d): no produjo frames en calentamiento.",
                        index, backend,
                    )
                    cap.release()
                    continue

                backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "AUTO"
                logger.info("Camara %d abierta con backend %s.", index, backend_name)
                return cap

            except Exception as e:
                logger.error("Error abriendo camara %d con backend %d: %s", index, backend, e)

        logger.error("No se pudo abrir la camara %d con ningun backend.", index)
        return None

    def _show_disconnected_frame(self):
        """Genera un frame negro con mensaje de desconexion."""
        frame = np.zeros((self.DISPLAY_H, self.DISPLAY_W, 3), dtype=np.uint8)
        cv2.putText(
            frame, "Camara desconectada", (120, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        cv2.putText(
            frame, "Reintentando...", (170, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1,
        )
        with self._lock:
            self._current_frame = frame

    def _set_status_threadsafe(self, text: str, color: str):
        """Actualiza el label de estado desde cualquier hilo."""
        try:
            self.after(0, lambda: self.lbl_status.configure(text=text, fg=color))
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Actualizacion de GUI (hilo principal)
    # -----------------------------------------------------------------------
    def _update_gui(self):
        """Actualiza video y stats en el hilo principal de Tkinter."""
        if not self._running:
            return

        # Actualizar frame de video
        with self._lock:
            frame = self._current_frame

        if frame is not None:
            try:
                img = Image.fromarray(frame)
                self._photo_image = ImageTk.PhotoImage(image=img)
                if self._canvas_image_id is None:
                    # Primera vez: crear el item de imagen en el canvas
                    self._canvas_image_id = self.video_canvas.create_image(
                        0, 0, anchor="nw", image=self._photo_image
                    )
                else:
                    # Actualizaciones siguientes: modificar el item existente sin borrar
                    # Esto elimina el flash negro que causaba el parpadeo visible
                    self.video_canvas.itemconfig(self._canvas_image_id, image=self._photo_image)
            except Exception:
                pass

        # Actualizar estadisticas
        self._update_stats_labels()

        # Actualizar grafico
        self._draw_chart()

        # Re-programar
        self.after(self.UPDATE_MS, self._update_gui)

    def _update_stats_labels(self):
        """Actualiza las etiquetas de estadisticas segun el modo activo."""
        c = self.counter
        if c.counting_mode == "fov":
            self.lbl_entradas.configure(text=f"PERSONAS:  {c.fov_count}")
            self.lbl_salidas.configure(text="SALIDAS:   N/A")
        else:
            self.lbl_entradas.configure(text=f"ENTRADAS:  {c.in_count}")
            self.lbl_salidas.configure(text=f"SALIDAS:   {c.out_count}")
        self.lbl_en_cuadro.configure(text=f"EN CUADRO:  {c.persons_in_frame}")
        self.lbl_fps.configure(text=f"FPS: {c.fps:.1f}")

    def _draw_placeholder(self, text: str):
        """Dibuja un mensaje centrado en el canvas de video."""
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            self.DISPLAY_W // 2, self.DISPLAY_H // 2,
            text=text, fill="#888888", font=("Segoe UI", 14),
        )

    # -----------------------------------------------------------------------
    # Grafico de trafico por hora (canvas simple)
    # -----------------------------------------------------------------------
    def _draw_chart(self):
        """Dibuja barras de trafico por hora en el canvas."""
        canvas = self.chart_canvas
        canvas.delete("all")

        data = self.counter.hourly_entries
        if not data:
            canvas.create_text(
                canvas.winfo_width() // 2 or 110, 60,
                text="Sin datos aun", fill="#666666", font=("Segoe UI", 9),
            )
            return

        cw = canvas.winfo_width() or 220
        ch = canvas.winfo_height() or 120
        margin_bottom = 20
        margin_top = 10
        chart_h = ch - margin_bottom - margin_top

        # Rango de horas con datos
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

            # Barra
            canvas.create_rectangle(
                x, y_top, x + bar_w, y_bottom,
                fill="#22aa55", outline="#33cc66",
            )

            # Valor encima de la barra
            if val > 0:
                canvas.create_text(
                    x + bar_w // 2, y_top - 6,
                    text=str(val), fill="#aaaaaa", font=("Consolas", 7),
                )

            # Hora debajo
            canvas.create_text(
                x + bar_w // 2, y_bottom + 10,
                text=f"{h}h", fill="#888888", font=("Consolas", 7),
            )

    # -----------------------------------------------------------------------
    # Auto-guardado y cierre
    # -----------------------------------------------------------------------
    def _auto_save(self):
        """Auto-guardado del CSV cada 5 minutos."""
        try:
            if self.counter.in_count > 0 or self.counter.out_count > 0:
                filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self._csv_dir, filename)
                self.counter.export_csv(filepath)
                logger.info("Auto-guardado: %s", filepath)
        except Exception as e:
            logger.error("Error en auto-guardado: %s", e)
        # Re-programar
        self.after(self._autosave_interval, self._auto_save)

    def _on_close(self):
        """Cierre limpio: guardar datos y liberar recursos."""
        logger.info("Cerrando aplicacion...")
        # Detener video
        self._running = False
        self._stop_event.set()

        # Guardar datos finales
        try:
            if self.counter.in_count > 0 or self.counter.out_count > 0:
                filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                filepath = os.path.join(self._csv_dir, filename)
                self.counter.export_csv(filepath)
                logger.info("Datos guardados al cerrar: %s", filepath)
        except Exception as e:
            logger.error("Error guardando al cerrar: %s", e)

        # Esperar a que el hilo de video termine
        if self._video_thread is not None and self._video_thread.is_alive():
            self._video_thread.join(timeout=3)

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
