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
| Browse... | a video file or a folder of frames. If a config uses that video, it is selected; otherwise its first frame is shown and Start calibrates it from its first seconds (~30 s, below), then replays it from the start with that calibration. A failed calibration is reported with its conditions; no other scenario's calibration is used |
| Auto-calibrate / Road / s | calibrate videos without a config (on by default; off: replay with the selected scenario's calibration); the road standard of the country ("any", DE_autobahn, CN_expressway, KR_expressway, KR_urban, US_interstate; remembered, set it once - "any" can mistake one standard's dash ratio for another's); seconds of video to perceive |
| Calibration conditions | progress, and every condition of every calibration source (why one was admitted or rejected) |
| Start / Pause / Restart / Stop | replay control |
| Image | the real video with YOLO tracks; the reconstruction is in the viewport (CCTV camera) |
| VLM | latest vision-language scene analysis (see the VLM Console) |
| VLM Console | opens the console window below |

## Automatic calibration

Starting a video that has no config runs `tools/calibrate_video.py` in its own process
(`calibration.py`) while the replay already plays on the borrowed calibration:

1. perceive the first N seconds (yolo11s at 1280 px, ByteTrack, camera motion)
2. select a calibration: this camera's stored calibration if the view is unchanged
   (`data/calibrations/<camera id>/`, lane or track vanishing point) -> lane markings
   (METRIC: dash period and lane width of a road standard, car box widths checked) ->
   vanishing points (PROJECTIVE, not enough to place vehicles)
3. METRIC: write `config/auto_<recording>.yaml` (camera, homography, the real lane
   markings for the USD scene) and store a new lane-marking calibration under the camera id

The extension then selects that config and restarts the replay on it at the same video
time: the stage is rebuilt with the calibrated camera and lane markings, the world model
uses the new homography. The video is from then on a scenario of its own. Without a
METRIC calibration the replay stays on the borrowed one and the window says which
conditions failed.

A frame folder's frame rate is read from the file names when they carry the capture
time (`<s>_<ns>_...`, TUMTraf); for video files from the container.

## Automatic lane markings

Every scenario gets the real lane markings in the scene, whatever its calibration came
from. A config calibrated from lane markings carries them; for any other (calibrated by
hand, borrowed, or after a failed automatic calibration) `tools/extract_lanes.py` runs
in the background: marking pieces of the video's empty-road background (dashes too
short to pass alone are kept when they line up towards the vanishing point), projected
with the scenario's own homography so they lie where its vehicles are placed. When done
they replace the straight lane lines in the running scene. They are cached per video and
calibration in `data/cache/lanes/` (main.py uses the cache too), so the next start draws
them right away.

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
| `autostart_video` | "" | Browse this video and Start (alone: calibrate and replay it; with `autostart_config`: that scenario selected first) |
| `snapshot_dir` / `snapshot_every` | "" / 125 | save real view + viewport every N frames |
| `quit_on_finish` | false | quit Isaac Sim at the end of the video |
| `vlm_periodic` | false | periodic VLM analysis on from the start |
| `vlm_log` | false | log every VLM result to `data/results/vlm/extension_log.jsonl` |
