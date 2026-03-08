# Contador de Personas — Servidor Web Multi-Camara

Sistema headless de conteo de personas en tiempo real mediante visión por computadora. Diseñado para correr en una **NVIDIA Jetson Orin Nano** (o cualquier PC con Linux/Windows), expone un dashboard web accesible desde cualquier dispositivo en la red local.

---

## Caracteristicas

- Deteccion con **YOLOv8** y tracking con **ByteTrack** via la libreria `supervision`
- Soporte para **hasta 3 camaras USB simultaneas**, cada una en su propio hilo
- Dos modos de conteo:
  - **`line`** — cuenta entradas y salidas cruzando una linea virtual
  - **`fov`** — cuenta personas unicas vistas en el campo de vision
- Dashboard web responsive (dark theme), accesible desde celular via WiFi local
- API REST JSON para integracion con sistemas externos
- Auto-guardado de datos en CSV cada 5 minutos
- Reconexion automatica de camara ante fallos
- Reset de tracker cada 30 minutos para evitar acumulacion de IDs

---

## Requisitos

| Dependencia | Version minima |
|---|---|
| Python | 3.9+ |
| OpenCV | 4.x |
| Ultralytics (YOLOv8) | 8.x |
| Supervision | 0.x |
| Flask | 2.x |
| NumPy | 1.x |

> En Jetson se recomienda `opencv-python-headless` en lugar de `opencv-python`.

---

## Instalacion

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd contador-personas

# 2. Crear y activar el entorno virtual
python3 -m venv venv

# Linux / Jetson
source venv/bin/activate

# Windows (bash)
source venv/Scripts/activate

# 3. Instalar dependencias
pip install flask opencv-python ultralytics supervision numpy

# En Jetson (sin GUI):
pip install flask opencv-python-headless ultralytics supervision numpy
```

El modelo `yolov8n.pt` se descarga automaticamente de internet la primera vez que se ejecuta el servidor.

---

## Uso

```bash
python server.py
```

El servidor queda disponible en:

- **Desarrollo (local):** `http://localhost:5000`
- **Jetson / red local:** `http://<ip-del-dispositivo>:5000`

Para encontrar la IP del dispositivo en Linux:

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

---

## Configuracion

Todos los parametros se encuentran al inicio de `server.py`:

```python
CAMERAS = [
    {"id": 0, "name": "Entrada Principal",   "index": 0},
    {"id": 1, "name": "Entrada Lateral Izq", "index": 1},
    {"id": 2, "name": "Entrada Lateral Der", "index": 2},
]

MODEL_NAME        = "yolov8n.pt"   # "yolov8s.pt" para mayor precision
CONFIDENCE        = 0.35           # Umbral de deteccion (0.0 - 1.0)
LINE_POS          = 0.5            # Posicion de la linea (0.0 = inicio, 1.0 = fin)
LINE_ORIENT       = "horizontal"   # "horizontal" o "vertical"
COUNTING_MODE     = "line"         # "line" o "fov"
SERVER_PORT       = 5000
AUTOSAVE_INTERVAL = 300            # Segundos entre guardados automaticos de CSV
```

> **Tip:** Para encontrar el indice de una camara en Linux usar `ls /dev/video*`. En Windows, los indices empiezan en 0 y se puede probar con `0`, `1`, `2`, etc.

---

## API REST

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/api/counts` | JSON con conteos de todas las camaras y totales agregados |
| `GET` | `/api/snapshot/<id>` | Ultimo frame anotado como JPEG (503 si aun no hay frame) |
| `POST` | `/api/config/<id>` | Reconfigurar una camara en caliente (JSON body) |
| `POST` | `/api/reset/<id>` | Resetear contadores de la camara `id` |
| `POST` | `/api/reset/all` | Resetear contadores de todas las camaras |

### Ejemplo: obtener conteos

```bash
curl http://localhost:5000/api/counts
```

```json
{
  "cameras": [
    {
      "id": 0,
      "name": "Entrada Principal",
      "in_count": 42,
      "out_count": 38,
      "fov_count": 0,
      "mode": "line"
    }
  ],
  "totals": {
    "in": 42,
    "out": 38
  }
}
```

### Ejemplo: resetear una camara

```bash
curl -X POST http://localhost:5000/api/reset/0
```

---

## Salida CSV

Los archivos se generan automaticamente en el mismo directorio que `server.py`:

```
conteo_cam0_2025-01-15.csv
conteo_cam1_2025-01-15.csv
```

**Modo `line`:**

| hora | entradas_intervalo | salidas_intervalo | entradas_total | salidas_total |
|------|--------------------|-------------------|----------------|---------------|

**Modo `fov`:**

| hora | personas_intervalo | personas_total |
|------|--------------------|----------------|

---

## Arquitectura

```
server.py
├── PersonCounter        # Motor de deteccion (YOLOv8 + ByteTrack + supervision)
│   ├── process_frame()  # Detecta, trackea y cuenta en cada frame
│   └── export_csv()     # Exporta historial a CSV
├── CameraWorker         # Un hilo por camara
│   ├── _loop()          # Captura, procesa y reconecta en caso de error
│   └── Metodos publicos # get_stats(), get_snapshot(), reset(), reconfigure()
└── Flask API            # Dashboard + endpoints REST
```

Cada `CameraWorker` carga su propia instancia de YOLO (los modelos no son thread-safe para compartir). La sincronizacion se realiza mediante un unico `threading.Lock` por worker.

---

## Despliegue en Jetson Orin Nano

```bash
# Instalar dependencias del sistema
sudo apt install python3-pip python3-venv

# Configurar e iniciar el servidor
python3 -m venv venv
source venv/bin/activate
pip install flask opencv-python-headless ultralytics supervision numpy

python3 server.py
```

Para que el servidor inicie automaticamente al encender la Jetson, se puede usar `systemd` o `cron @reboot`.

---

## Estructura del repositorio

```
.
├── server.py        # Servidor principal (unico archivo fuente)
├── CLAUDE.md        # Instrucciones para Claude Code
├── README.md
└── .gitignore
```

> `yolov8n.pt` y los CSV generados en runtime **no se incluyen** en el repositorio (ver `.gitignore`).
