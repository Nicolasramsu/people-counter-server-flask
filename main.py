"""Punto de entrada de la aplicación Contador de Personas.

Secuencia de arranque:
  1. Configurar logging.
  2. Cargar el modelo YOLO (bloquea hasta completar, ~2-5 s en Jetson).
  3. Iniciar la captura de cámara en hilo de fondo.
  4. Crear la app Flask con las referencias compartidas.
  5. Lanzar Flask → el dashboard queda disponible en http://<IP>:8000

Ejecutar con:
    python main.py
"""

import logging
import os
import sys

from camera.capture import CameraCapture
from config import AUTOSAVE_INTERVAL_S, SERVER_HOST, SERVER_PORT
from core.person_counter import PersonCounter
from server.app import create_app
from utils.logger import setup_logging

logger = logging.getLogger("ContadorPersonas")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir  = os.path.join(base_dir, "data")
    os.makedirs(csv_dir, exist_ok=True)

    setup_logging(base_dir)
    logger.info("=" * 60)
    logger.info("Iniciando Contador de Personas")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Motor de detección + carga del modelo
    # ------------------------------------------------------------------
    counter = PersonCounter()
    logger.info("Cargando modelo YOLO… (puede tardar unos segundos)")
    try:
        counter.load_model()
    except Exception as e:
        logger.critical("No se pudo cargar el modelo YOLO: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Captura de cámara
    # ------------------------------------------------------------------
    camera = CameraCapture(counter)
    camera.start()

    # ------------------------------------------------------------------
    # 3. Auto-guardado periódico
    # ------------------------------------------------------------------
    import threading
    import time

    def _autosave_loop():
        while True:
            time.sleep(AUTOSAVE_INTERVAL_S)
            try:
                if counter.in_count > 0 or counter.fov_count > 0:
                    from datetime import datetime
                    filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                    counter.export_csv(os.path.join(csv_dir, filename))
            except Exception as exc:
                logger.error("Error en auto-guardado: %s", exc)

    threading.Thread(target=_autosave_loop, daemon=True, name="AutosaveThread").start()

    # ------------------------------------------------------------------
    # 4. Servidor web
    # ------------------------------------------------------------------
    app = create_app(counter, camera, csv_dir=csv_dir)

    logger.info("Dashboard disponible en http://%s:%d", SERVER_HOST, SERVER_PORT)
    logger.info("Usa Ctrl+C para detener.")

    try:
        app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            threaded=True,   # un hilo por request (necesario para WebSocket y MJPEG)
        )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Deteniendo captura de cámara…")
        camera.stop()

        # Guardar datos al salir
        try:
            if counter.in_count > 0 or counter.fov_count > 0:
                from datetime import datetime
                filename = f"conteo_{datetime.now().strftime('%Y-%m-%d')}.csv"
                counter.export_csv(os.path.join(csv_dir, filename))
                logger.info("Datos guardados al cerrar.")
        except Exception as exc:
            logger.error("Error al guardar datos finales: %s", exc)

        logger.info("Aplicación cerrada.")


if __name__ == "__main__":
    main()