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
| VLM | latest vision-language scene analysis, when the config enables `vlm` and an Ollama server is running |

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
