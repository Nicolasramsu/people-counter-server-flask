"""Configuración global y constantes de la aplicación.

Este módulo centraliza todos los parámetros para evitar valores mágicos
dispersos en el código. Modifica este archivo para ajustar el comportamiento
del sistema sin tocar la lógica principal.
"""

# ---------------------------------------------------------------------------
# Detección (YOLO)
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 0

DEFAULT_MODEL = "yolov8s.pt"
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

DEFAULT_CONFIDENCE = 0.3
DEFAULT_INFER_SIZE = 640

# Dispositivo de inferencia: 0 = CUDA GPU 0, "cpu" = solo CPU
INFERENCE_DEVICE = 0

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
AREA_THRESHOLD = 0.30           # Porcentaje mínimo del bbox para registrar cruce

# Matching threshold del tracker: 0.8 era demasiado estricto para multitudes.
# Con 0.6 el tracker re-asocia mejor tras oclusiones parciales.
TRACKER_MATCHING_THRESHOLD = 0.60

# Limpieza periódica de estado interno por tracks inactivos.
# No recrea el tracker (evita el doble conteo del bug original).
STALE_TRACK_THRESHOLD_S    = 10 * 60   # Track sin aparecer 10 min → stale
STALE_TRACK_CLEANUP_INTERVAL_S = 5 * 60  # Revisar cada 5 min

# ---------------------------------------------------------------------------
# Líneas de conteo (posición relativa 0.0–1.0 del frame)
# ---------------------------------------------------------------------------
DEFAULT_LINE_POSITION_H = 0.5
DEFAULT_LINE_POSITION_V = 0.5

# Tiempo mínimo (segundos) entre dos cruces registrados del mismo track.
# Previene el doble conteo por oscilación del bbox cerca de la línea.
LINE_CROSSING_COOLDOWN_S = 3.0

# ROI / Zona poligonal por defecto (x1, y1, x2, y2) en coordenadas normalizadas
DEFAULT_ROI = (0.0, 0.0, 1.0, 1.0)

# Tiempo mínimo que un track debe permanecer dentro del polígono para contarse.
# Filtra personas que cruzan el área sin detenerse.
POLYGON_DWELL_TIME_S = 2.0

# Tiempo mínimo antes de recontar a la misma persona si re-entra al polígono.
POLYGON_REENTRY_COOLDOWN_S = 10 * 60   # 10 minutos

# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------
INTERVAL_SECONDS = 15 * 60     # Período de agregación de datos (15 min)
AUTOSAVE_INTERVAL_S = 5 * 60   # Auto-guardado periódico (5 min)

# ---------------------------------------------------------------------------
# Cámara — backend OpenCV (genérico)
# ---------------------------------------------------------------------------
# Opciones para CAMERA_SOURCE:
#   Entero (0, 1, 2…)         → webcam USB/V4L2 por índice
#   "rtsp://user:pass@ip/…"   → stream RTSP
#   String GStreamer           → pipeline personalizado (Jetson CSI, etc.)
CAMERA_SOURCE = 0

# Resolución solicitada a la cámara (puede ser ignorada por el driver)
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# Pipeline GStreamer para cámara CSI en Jetson Orin Nano.
# Descomenta y ajusta si usas una cámara CSI en lugar de USB.
# CAMERA_SOURCE = (
#     "nvarguscamerasrc ! "
#     "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
#     "nvvidconv flip-method=0 ! "
#     "video/x-raw,format=BGRx ! videoconvert ! appsink"
# )

# Calidad de compresión JPEG para el stream de video (0–100)
MJPEG_QUALITY = 80

# ---------------------------------------------------------------------------
# Cámara — backend OAK-D W (Luxonis DepthAI)
# ---------------------------------------------------------------------------
# Selección de backend:
#   "opencv"  → usa cv2.VideoCapture (CAMERA_SOURCE arriba)
#   "oak"     → usa DepthAI con Luxonis OAK-D W (requiere: pip install depthai)
CAMERA_BACKEND = "opencv"

# Framerate del sensor de color y del par estéreo
OAK_RGB_FPS    = 30
OAK_STEREO_FPS = 30

# ---------------------------------------------------------------------------
# Filtro de profundidad (activo solo con CAMERA_BACKEND="oak")
# ---------------------------------------------------------------------------
# Rango de profundidad válida para reconocer una detección como persona
# real dentro del stand. Detecciones fuera del rango se descartan antes
# de pasar al contador, eliminando personas del fondo.
#
# Unidades: metros.  El sensor OAK-D W tiene rango útil de ~0.3 m a ~15 m.
# Ajustar MAX_DETECTION_DEPTH_M según el tamaño real del stand.
MIN_DETECTION_DEPTH_M = 0.3   # por debajo → ruido del sensor estéreo
MAX_DETECTION_DEPTH_M = 8.0   # por encima → fondo irrelevante

# ---------------------------------------------------------------------------
# Servidor web
# ---------------------------------------------------------------------------
SERVER_HOST = "localhost"        # Escuchar en todas las interfaces
SERVER_PORT = 8000

# Intervalo de broadcast de estadísticas por WebSocket (segundos)
WS_BROADCAST_INTERVAL = 1.0