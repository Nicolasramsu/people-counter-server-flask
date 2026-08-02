"""Backend de captura para Luxonis OAK-D W via DepthAI.

Implementa la misma interfaz pública que ``CameraCapture`` (duck typing),
por lo que ``main.py`` puede intercambiar ambos backends sin modificar
el resto del sistema.

Pipeline DepthAI configurado:
  ┌─────────────┐    ┌──────────────┐
  │ ColorCamera │───►│  XLinkOut    │ → "color" (BGR, 1080p)
  └─────────────┘    └──────────────┘
  ┌─────────────┐ ┐
  │ MonoCamera  │  ├──►┌─────────────┐    ┌──────────────┐
  │   (Left)    │  │   │ StereoDepth │───►│  XLinkOut    │ → "depth" (uint16 mm)
  │ MonoCamera  │  ├──►│  alineado   │    └──────────────┘
  │   (Right)   │ ┘   └─────────────┘
  └─────────────┘

El mapa de profundidad se alinea al sensor de color (CAM_A) y se
redimensiona a la misma resolución, de modo que depth[y, x] corresponde
exactamente al píxel color[y, x].

Los frames de color y profundidad se pasan juntos a ``PersonCounter``
para habilitar el filtrado por distancia.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from config import MJPEG_QUALITY, OAK_RGB_FPS, OAK_STEREO_FPS
from core.person_counter import PersonCounter

logger = logging.getLogger("ContadorPersonas")

# Resolución del sensor RGB de la OAK-D W en modo 1080p
_RGB_W = 1920
_RGB_H = 1080

# Resolución de los sensores mono (OV9282)
_MONO_RES_STR = "800"   # "400" para menor uso de ancho de banda USB


class OakCapture:
    """Captura frames RGB + profundidad de una OAK-D W usando DepthAI.

    Misma interfaz pública que ``CameraCapture``:
      - ``start()`` / ``stop()``
      - ``get_jpeg_frame()``
      - ``add_frame_callback()``
    """

    def __init__(self, counter: PersonCounter) -> None:
        self.counter = counter
        self._running = False
        self._thread: threading.Thread | None = None

        # Último frame anotado en bytes JPEG (atómico bajo el GIL de Python)
        self._jpeg_frame: bytes | None = None
        self._frame_callbacks: list = []

    # ------------------------------------------------------------------
    # Control del ciclo de captura
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia el hilo de captura DepthAI. Sin efecto si ya está corriendo."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="OakCameraThread"
        )
        self._thread.start()
        logger.info("Captura OAK-D W iniciada.")

    def stop(self) -> None:
        """Detiene el hilo de captura y espera a que finalice."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=8)
        logger.info("Captura OAK-D W detenida.")

    # ------------------------------------------------------------------
    # Acceso al frame actual (lectura desde el hilo Flask/WebSocket)
    # ------------------------------------------------------------------

    def get_jpeg_frame(self) -> bytes | None:
        """Retorna el último frame anotado en bytes JPEG, o None si aún no hay."""
        return self._jpeg_frame

    def add_frame_callback(self, callback) -> None:
        """Registra un callback invocado tras cada frame procesado."""
        self._frame_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Hilo de captura
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            import depthai as dai
        except ImportError:
            logger.critical(
                "El paquete 'depthai' no está instalado. "
                "Ejecuta: pip install depthai"
            )
            return

        pipeline = self._build_pipeline(dai)

        try:
            with dai.Device(pipeline) as device:
                logger.info(
                    "OAK-D W conectada. MxID: %s",
                    device.getMxId(),
                )

                q_color = device.getOutputQueue(name="color", maxSize=4, blocking=False)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

                self.counter.setup_line(_RGB_W, _RGB_H)
                logger.info("Pipeline DepthAI activo: %dx%d @ %d fps", _RGB_W, _RGB_H, OAK_RGB_FPS)

                self._run_frame_loop(q_color, q_depth)

        except Exception as e:
            logger.error("Error en la captura OAK-D W: %s", e, exc_info=True)

    def _run_frame_loop(self, q_color, q_depth) -> None:
        """Bucle principal de lectura de frames y paso al motor de detección."""
        frame_idx = 0
        last_depth: np.ndarray | None = None   # último depth válido disponible

        while self._running:
            in_color = q_color.tryGet()

            if in_color is None:
                time.sleep(0.005)
                continue

            color_frame: np.ndarray = in_color.getCvFrame()

            # Intentar obtener depth del mismo ciclo; si no hay, usar el último
            in_depth = q_depth.tryGet()
            if in_depth is not None:
                last_depth = in_depth.getFrame().astype(np.uint16)

            frame_idx += 1

            try:
                annotated = self.counter.process_frame(color_frame, depth_frame=last_depth)
            except Exception as exc:
                logger.error("Error en inferencia (frame %d): %s", frame_idx, exc, exc_info=True)
                annotated = color_frame

            ok, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY],
            )
            if ok:
                self._jpeg_frame = jpeg.tobytes()

            for cb in self._frame_callbacks:
                try:
                    cb()
                except Exception:
                    pass

        logger.info("Bucle OAK-D W finalizado. Total frames: %d", frame_idx)

    # ------------------------------------------------------------------
    # Construcción del pipeline DepthAI
    # ------------------------------------------------------------------

    def _build_pipeline(self, dai) -> object:
        """Crea y devuelve el pipeline DepthAI con color + estéreo + depth."""
        pipeline = dai.Pipeline()

        # ── Sensor de color (IMX378, 12MP, 120° DFOV) ─────────────────
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(OAK_RGB_FPS)

        # ── Sensores mono (OV9282, 150° DFOV, par estéreo) ────────────
        mono_res = (
            dai.MonoCameraProperties.SensorResolution.THE_800_P
            if _MONO_RES_STR == "800"
            else dai.MonoCameraProperties.SensorResolution.THE_400_P
        )

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(mono_res)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setFps(OAK_STEREO_FPS)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(mono_res)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setFps(OAK_STEREO_FPS)

        # ── Nodo StereoDepth ──────────────────────────────────────────
        # HIGH_DENSITY: mayor cobertura del mapa (mejor para multitudes).
        # El depth se alinea al sensor de color (CAM_A) y se redimensiona
        # a la misma resolución que el frame RGB.
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(_RGB_W, _RGB_H)
        stereo.setLeftRightCheck(True)    # reduce "speckle" (ruido puntual)
        stereo.setSubpixel(False)         # no necesario para filtrado por rango

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # ── Salidas XLink ─────────────────────────────────────────────
        xout_color = pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        cam_rgb.video.link(xout_color.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return pipeline
