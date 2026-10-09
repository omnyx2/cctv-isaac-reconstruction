@echo off
rem Isaac Sim with the CCTV extensions (Dual Tracking, VLM Console, Timeline).
rem   run_isaac.bat [extra Kit arguments, e.g. --/exts/cctv.dual_tracking/autostart_config=config/tum_r09_s03_north50.yaml]
rem Opens with the windows docked around a clean viewport and the default scenario previewed
rem (its scene and first frame); pick another scenario in CCTV Dual Tracking, then Start.
rem Isaac Sim location: ISAAC_SIM (default C:\isaacsim).
setlocal
if "%ISAAC_SIM%"=="" set ISAAC_SIM=C:\isaacsim
cd /d "%~dp0"
call "%ISAAC_SIM%\isaac-sim.bat" --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline ^
  --/exts/cctv.dual_tracking/layout=true ^
  --/exts/cctv.dual_tracking/default_config=config/tum_r09_s03_north50.yaml ^
  %*
