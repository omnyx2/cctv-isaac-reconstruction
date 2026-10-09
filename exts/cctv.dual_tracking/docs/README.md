# cctv.dual_tracking

Isaac Sim extension of the CCTV_Project: pick a raw CCTV video and watch it
reconstructed live in the viewport.

## Run

From the project directory:

```
C:\isaacsim\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
```

The **CCTV Dual Tracking** window:

| Control | |
|---|---|
| Scenario | a calibrated config from `config/` (video + camera calibration) |
| Browse... | a video file or a folder of frames. If a config uses that video, it is selected; otherwise the video runs with the selected scenario's calibration (geometry is only right for the same camera) |
| Start / Pause / Restart / Stop | replay control |
| Image | the real video with YOLO tracks; the reconstruction is in the viewport (CCTV camera) |
| VLM | latest vision-language scene analysis (see the VLM Console) |
| VLM Console | opens the console window below |

## VLM Console

A second window, **CCTV VLM Console**, controls the vision-language model (Ollama). It runs beside YOLO and never
blocks the replay. Every change applies from the next analysis on.

| Section | |
|---|---|
| Connection | Ollama URL, check / refresh, model chosen from the pulled models |
| Run settings | periodic analysis on/off, interval (s), image long side (px), temperature, timeout, JSON answer, log to `data/results/vlm/console_<time>.jsonl` |
| Prompt | presets (`scene_json`, `accident_yesno`, `describe`) or your own text; Apply prompt. Turn *JSON answer* off for free-text prompts. An answer starting with yes/no sets the accident flag |
| Test now | analyze the current replay frame, or an image file (no replay needed), right away |
| Status and timing | frames offered / taken / skipped by the interval / replaced while busy / manual, completed, errors, latency last / mean / p95, analyses per minute, busy |
| Results | latest summary and parsed fields (red when accident), raw answer, history (newest first) |

A scenario whose config has `vlm.enabled: true` turns periodic analysis on with its settings.

## How it runs

`session.py` does what `main.py` does, driven by Kit's update events instead of a
loop: the perception process (YOLO + ByteTrack, a separate Python process using
Isaac Sim's bundled interpreter and torch from `omni.isaac.ml_archive`) runs ahead;
on each app update the session takes the next result only if it is ready and due
at the video frame rate, steps the world model and moves the vehicle prims.
Nothing blocks the UI.

## Settings (`--/exts/cctv.dual_tracking/<name>=<value>`)

| Setting | Default | |
|---|---|---|
| `project_root` | the project containing `exts/` | |
| `autostart_config` | "" | config to start right away (project-relative) |
| `snapshot_dir` / `snapshot_every` | "" / 125 | save real view + viewport every N frames |
| `quit_on_finish` | false | quit Isaac Sim at the end of the video |
| `vlm_periodic` | false | periodic VLM analysis on from the start |
| `vlm_log` | false | log every VLM result to `data/results/vlm/extension_log.jsonl` |
