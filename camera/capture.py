"""Captura de cámara en tiempo real y ejecución del motor de detección.

Este módulo gestiona la lectura continua de frames desde la cámara,
los pasa al motor de detección (PersonCounter) y mantiene el último
frame anotado disponible en formato JPEG para el stream de video.
"""

import logging
import threading
import time

import cv2
import numpy as np

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_SOURCE,
    CAMERA_WIDTH,
    MJPEG_QUALITY,
)
from core.person_counter import PersonCounter

logger = logging.getLogger("ContadorPersonas")


class CameraCapture:
    """Gestiona la captura en tiempo real y la detección de personas.

    Corre en un hilo de fondo dedicado. El hilo principal (servidor web)
    lee el último frame JPEG disponible y las estadísticas a través de
    los atributos del ``PersonCounter`` compartido.

    Ejemplo de uso::

        counter = PersonCounter()
        counter.load_model()

        camera = CameraCapture(counter)
        camera.start()
        ...
        camera.stop()
    """

    def __init__(self, counter: PersonCounter) -> None:
        """
        Args:
            counter: Motor de detección ya inicializado (modelo cargado).
        """
        self.counter = counter
        self._running = False
        self._thread: threading.Thread | None = None

        # Último frame anotado en bytes JPEG (None hasta el primer frame)
        # La asignación de referencias en Python es atómica bajo el GIL,
        # por lo que los lectores del servidor pueden acceder sin lock.
        self._jpeg_frame: bytes | None = None

        # Callbacks invocados tras cada frame procesado
        self._frame_callbacks: list = []

    # ------------------------------------------------------------------
    # Control del ciclo de captura
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia el hilo de captura. No tiene efecto si ya está corriendo."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Captura de cámara iniciada.")

    def stop(self) -> None:
        """Detiene el hilo de captura y espera a que finalice."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Captura de cámara detenida.")

    # ------------------------------------------------------------------
    # Acceso al frame actual
    # ------------------------------------------------------------------

    def get_jpeg_frame(self) -> bytes | None:
        """Retorna el último frame anotado en bytes JPEG, o ``None`` si aún no hay."""
        return self._jpeg_frame

    def add_frame_callback(self, callback) -> None:
        """Registra una función a invocar tras cada frame procesado.

        Útil para notificar al broadcaster de WebSocket que hay datos nuevos.
        El callback se invoca en el hilo de captura; debe ser rápido y
        no bloqueante.
        """
        self._frame_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Hilo de captura
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        cap = self._open_camera()
        if cap is None:
            logger.error("No se pudo abrir la cámara. Verifica CAMERA_SOURCE en config.py.")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS

        self.counter.setup_line(w, h)
        logger.info("Cámara abierta: %dx%d @ %.1f fps", w, h, actual_fps)

        frame_idx = 0
        consecutive_errors = 0

        try:
            while self._running:
                ret, frame = cap.read()

                if not ret:
                    consecutive_errors += 1
                    logger.warning("Frame perdido (%d consecutivos).", consecutive_errors)
                    if consecutive_errors >= 30:
                        logger.error("Demasiados errores consecutivos. Deteniendo captura.")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0
                frame_idx += 1

                # Inferencia y anotación
                try:
                    annotated = self.counter.process_frame(frame)
                except Exception as e:
                    logger.error("Error en inferencia (frame %d): %s", frame_idx, e, exc_info=True)
                    annotated = frame   # mostrar frame sin anotar si falla

                # Codificar a JPEG para el stream MJPEG
                ok, jpeg = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY],
                )
                if ok:
                    self._jpeg_frame = jpeg.tobytes()

                # Notificar a los listeners (p.ej. broadcaster WebSocket)
                for cb in self._frame_callbacks:
                    try:
                        cb()
                    except Exception:
                        pass

        except Exception as e:
            logger.error("Error inesperado en el hilo de captura: %s", e, exc_info=True)
        finally:
            cap.release()
            logger.info("Cámara liberada. Total de frames procesados: %d", frame_idx)

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Abre la cámara según CAMERA_SOURCE y aplica la resolución configurada."""
        source = CAMERA_SOURCE

        # Pipeline GStreamer (string que no es RTSP)
        if isinstance(source, str) and not source.lower().startswith("rtsp"):
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return None

        # Solicitar resolución y framerate (el driver puede ignorarlo)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        return cap