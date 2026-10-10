# Changelog

## [0.3.0] - 2026-10-10
- CCTV Setup window (opens by itself until the environment check passes): datasets folder,
  packages in Isaac Sim's Python (ultralytics --no-deps), environment check, a PyTorch cu130
  venv for YOLO when Isaac Sim's torch does not cover the GPU (DGX Spark GB10), import of the
  transfer folder, Ollama model pull. Choices are persistent settings.
- Startup: docked layout and the default scenario previewed (persistent `layout_on_start`).
- Presentation mode, whole-window captures and H.264 recording (presentation.py).

## [0.2.0] - 2026-10-09
- VLM Console window: connection and model, live run settings, prompt editor with presets,
  analyze the current frame or an image file, throughput statistics, results and history, JSON lines log.
- VLM moved out of the replay session into a controller of its own (works without a replay).
- Autostart runs once per process.

## [0.1.0] - 2026-10-08
- Scenario / video selection, replay control, real view with tracks, VLM text, snapshots, unattended mode.
