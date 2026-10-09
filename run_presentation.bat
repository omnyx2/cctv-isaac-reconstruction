@echo off
rem Presentation mode: 1920x1080 window, real video | reconstruction | VLM console, timeline
rem below, no other panels, no viewport statistics / toolbar, no main menu (the "Layout"
rem button in CCTV Dual Tracking re-applies the docking).
rem   run_presentation.bat [extra Kit arguments]
rem Captures for slides (whole app window):
rem   --/exts/cctv.dual_tracking/snapshot_dir=<folder> --/exts/cctv.dual_tracking/snapshot_every=150
rem   --/exts/cctv.dual_tracking/record_dir=<folder> [--/exts/cctv.dual_tracking/record_every=2]
setlocal
call "%~dp0run_isaac.bat" ^
  --/app/window/width=1920 --/app/window/height=1080 ^
  --/exts/cctv.dual_tracking/presentation_layout=true ^
  --/app/viewport/defaults/hud/visible=false --/app/viewport/defaults/guide/axis/visible=false ^
  --/exts/omni.kit.viewport.menubar.camera/visible=false --/exts/omni.kit.viewport.menubar.render/visible=false ^
  --/exts/omni.kit.viewport.menubar.settings/visible=false --/exts/omni.kit.viewport.menubar.display/visible=false ^
  --/exts/omni.kit.viewport.menubar.lighting/visible=false --/exts/omni.kit.viewport.menubar.waypoint/visible=false ^
  %*
