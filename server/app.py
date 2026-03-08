"""Factory de la aplicación Flask.

Responsabilidades:
  - Crear la app Flask con sus rutas REST y stream MJPEG.
  - Registrar el endpoint WebSocket con flask-sock.
  - Iniciar el hilo de broadcast periódico de estadísticas.
  - Servir los archivos estáticos del dashboard web.
"""

import json
import logging
import pathlib
import threading
import time

from flask import Flask
from flask_sock import Sock

from config import WS_BROADCAST_INTERVAL
from server.routes import build_stats, register_routes
from server.websocket_manager import ConnectionManager

logger = logging.getLogger("ContadorPersonas")

_STATIC_DIR = str(pathlib.Path(__file__).parent.parent / "static")


def create_app(counter, camera, csv_dir: str | None = None) -> Flask:
    """Crea y configura la aplicación Flask.

    Args:
        counter: Instancia de ``PersonCounter`` con el modelo ya cargado.
        camera:  Instancia de ``CameraCapture`` ya iniciada.
        csv_dir: Directorio para CSV de auto-guardado (no usado internamente,
                 disponible para extensiones futuras).

    Returns:
        Aplicación Flask lista para ejecutar con ``app.run()``.
    """
    app = Flask(
        __name__,
        static_folder=_STATIC_DIR,
        static_url_path="",           # archivos estáticos en la raíz "/"
    )
    sock = Sock(app)
    ws_manager = ConnectionManager()

    # ------------------------------------------------------------------
    # Rutas REST + MJPEG
    # ------------------------------------------------------------------
    register_routes(app, counter, camera)

    # ------------------------------------------------------------------
    # WebSocket  (flask-sock asigna un hilo por conexión)
    # ------------------------------------------------------------------
    @sock.route("/ws")
    def ws_endpoint(ws):
        """Canal WebSocket: envía estadísticas al conectar y mantiene viva la sesión.

        El broadcaster (hilo de fondo) llama a ``ws.send()`` mientras este
        hilo permanece bloqueado en ``ws.receive()``.  flask-sock /
        simple-websocket soporta send/receive concurrentes desde distintos
        hilos (full-duplex).
        """
        ws_manager.add(ws)
        try:
            # Estado inicial inmediato al conectarse
            ws.send(json.dumps(build_stats(counter)))
            # Mantener la conexión viva; se acepta cualquier mensaje del cliente
            while True:
                ws.receive(timeout=30)   # despierta cada 30 s o en mensaje del cliente
        except Exception:
            pass   # ConnectionClosed u otro error → salir limpiamente
        finally:
            ws_manager.remove(ws)

    # ------------------------------------------------------------------
    # Hilo de broadcast periódico
    # ------------------------------------------------------------------
    def _broadcast_loop() -> None:
        """Emite estadísticas a todos los clientes WS cada WS_BROADCAST_INTERVAL s."""
        while True:
            time.sleep(WS_BROADCAST_INTERVAL)
            if ws_manager.active_count > 0:
                try:
                    ws_manager.broadcast(json.dumps(build_stats(counter)))
                except Exception as e:
                    logger.error("Error en broadcast: %s", e)

    broadcaster = threading.Thread(
        target=_broadcast_loop, daemon=True, name="WsBroadcastThread"
    )
    broadcaster.start()
    logger.info("Broadcast WebSocket iniciado (cada %.1f s).", WS_BROADCAST_INTERVAL)

    return app
