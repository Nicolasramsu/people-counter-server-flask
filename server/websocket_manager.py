"""Gestión de conexiones WebSocket activas (compatible con flask-sock).

Usa un ``threading.Lock`` para acceso seguro desde múltiples hilos:
  - El hilo broadcaster escribe (envía mensajes).
  - El hilo de cada conexión WebSocket lee y elimina su entrada.
"""

import logging
import threading

logger = logging.getLogger("ContadorPersonas")


class ConnectionManager:
    """Registra clientes WebSocket y hace broadcast desde cualquier hilo."""

    def __init__(self) -> None:
        self._connections: set = set()
        self._lock = threading.Lock()

    def add(self, ws) -> None:
        """Registra una nueva conexión WebSocket."""
        with self._lock:
            self._connections.add(ws)
        logger.info("Cliente WS conectado. Total activos: %d", self.active_count)

    def remove(self, ws) -> None:
        """Elimina una conexión del registro."""
        with self._lock:
            self._connections.discard(ws)
        logger.info("Cliente WS desconectado. Total activos: %d", self.active_count)

    def broadcast(self, message: str) -> None:
        """Envía ``message`` a todos los clientes conectados.

        Los clientes que fallen al recibir el mensaje se eliminan
        del registro automáticamente.
        """
        with self._lock:
            snapshot = list(self._connections)

        stale = []
        for ws in snapshot:
            try:
                ws.send(message)
            except Exception:
                stale.append(ws)

        if stale:
            with self._lock:
                for ws in stale:
                    self._connections.discard(ws)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._connections)
