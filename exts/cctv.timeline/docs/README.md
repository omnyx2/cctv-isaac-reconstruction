# cctv.timeline

Timeline controller for `cctv.dual_tracking` (see the project README and docs/TECHNICAL_NOTES.md 15).

```
C:\isaacsim\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline
```

- Slider and marker bar: orange = annotated contact time of the scenario, red = VLM accident answer, white = playhead. Click a marker (or Go in the list) to jump there minus the lead-in.
- |<, -5s, -1s, Play/Pause, Step (one frame), +1s, +5s, speed 0.1x-2x (slow motion around the accident).
- VLM-analyze the frame after each jump; analyze this frame now.
- Scan: analyze the whole video every N seconds in the background with the VLM Console model and prompt; accident answers become markers.

A jump restarts tracking and the world model at the new position (as at a video start).
