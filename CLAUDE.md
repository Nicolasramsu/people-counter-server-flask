# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the server

```bash
# Activate venv (Windows dev)
source venv/Scripts/activate   # bash
# or: venv\Scripts\activate.bat (cmd)

python server.py
# Dashboard available at http://localhost:5000
# On Jetson: http://<device-ip>:5000
```

No build step, no test suite. The only runtime dependency is `yolov8n.pt` in the same directory (downloaded automatically by ultralytics on first run if missing).

## Architecture

Everything lives in `server.py` (~1 100 lines). Three layers:

### 1. `PersonCounter` (detection engine, no GUI)
Wraps YOLOv8 + ByteTrack + supervision annotators. Stateful: maintains cumulative `in_count`/`out_count` (line mode) or `fov_count` (FOV mode), plus interval history for CSV export.

Key design: counters use an **offset pattern** — when the tracker or line zone is recreated, current counts are saved to `_in_offset`/`_out_offset` so the public counts stay monotonic.

Tracker is automatically reset every 30 minutes (`_tracker_reset_interval`) to prevent ID accumulation.

### 2. `CameraWorker` (one thread per camera)
Owns a `PersonCounter` instance (YOLO models are **not** thread-safe to share). The `_loop` method handles:
- Camera open/warmup with backend fallback (V4L2 → AUTO on Linux; DSHOW → AUTO on Windows, detected via `platform.system()`)
- Automatic reconnect on read failure (3 s delay)
- Hot-swap of physical camera index via `_pending_cam_index`

All public methods (`get_stats`, `get_snapshot`, `reset`, `reconfigure`) use `self._lock`. The `process_frame` call inside `_loop` also holds the lock, so stats reads never race with frame processing.

### 3. Flask REST API + inline dashboard
Dashboard HTML is a string constant (`DASHBOARD_HTML`). The frontend polls `/api/counts` every second and patches the DOM in-place to avoid full re-renders.

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/api/counts` | JSON with all camera stats + aggregated totals |
| GET | `/api/snapshot/<id>` | Latest annotated frame as JPEG (503 if no frame yet) |
| POST | `/api/config/<id>` | Reconfigure a camera live (JSON body) |
| POST | `/api/reset/<id>` | Reset counters (id = integer or `"all"`) |
| POST | `/api/reset/all` | Reset all cameras |

## Configuration

All tunable constants are at the top of `server.py`:

```python
CAMERAS           # list of {id, name, index} dicts
MODEL_NAME        # "yolov8n.pt" (fast) or larger variant
CONFIDENCE        # detection threshold (0.35 default)
LINE_POS          # 0.0–1.0, fraction of frame for counting line
LINE_ORIENT       # "horizontal" or "vertical"
COUNTING_MODE     # "line" (in/out) or "fov" (unique persons seen)
SERVER_PORT       # 5000
AUTOSAVE_INTERVAL # CSV auto-save period in seconds (300 = 5 min)
```

## Threading model

- Main thread: Flask (threaded=True, use_reloader=False)
- One `cam-{id}` daemon thread per camera
- One `autosave` daemon thread

The `CameraWorker._lock` is the only synchronization primitive. Flask request handlers must acquire it through the public worker methods — never access `counter` or `_last_annotated_frame` directly.

## CSV output

Auto-saved to the script's directory as `conteo_cam{id}_{YYYY-MM-DD}.csv`. Manual export can be triggered by calling `counter.export_csv(filepath)`. Schema differs by mode:
- **line mode**: `hora, entradas_intervalo, salidas_intervalo, entradas_total, salidas_total`
- **fov mode**: `hora, personas_intervalo, personas_total`
