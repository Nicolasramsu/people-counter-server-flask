"""Configuración centralizada del sistema de logging.

Uso típico
----------
En ``main.py`` (una sola vez al arrancar):

    from utils.logger import setup_logging
    setup_logging("/ruta/al/proyecto")

En cada módulo que necesite loggear:

    import logging
    logger = logging.getLogger("ContadorPersonas")
"""

import logging
import os


def setup_logging(log_dir: str) -> None:
    """Inicializa el sistema de logging con salida a archivo y consola.

    Args:
        log_dir: Directorio donde se creará el archivo ``contador.log``.
    """
    log_path = os.path.join(log_dir, "contador.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
