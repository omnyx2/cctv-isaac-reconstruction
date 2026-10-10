"""Regression test of the static-edge calibration on a fixed set of videos.

Runs the pipeline (PIPELINE below) on each video of VIDEOS, collects the calibration
(focal, height, pitch, vanishing point) and its error against the label-fitted TUM
camera, and writes data/results/calib_regression/<label>.json. With --compare, the
result is printed next to a stored one (default: tools/calib_regression_baseline.json,
the result of the master branch when the static-edge calibration was introduced), so a
branch is tested by running the same videos and comparing.

Usage:
    C:\\isaacsim\\python.bat tools/calib_regression.py --label <name> [--compare [file]] [--write-baseline]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import datasets_root  # noqa: E402

PYTHON = sys.executable
BASELINE = PROJECT_ROOT / "tools" / "calib_regression_baseline.json"
OUT = PROJECT_ROOT / "data" / "results" / "calib_regression"
# daytime TUM cameras (both lenses, both directions) with a label-fitted reference where available
VIDEOS = [
    "r09_s10/s040_camera_basler_north_50mm",
    "r09_s09/s050_camera_basler_south_50mm",
    "r09_s12/s050_camera_basler_south_50mm",
    "r09_s06/s050_camera_basler_south_50mm",
    "r09_s01/s040_camera_basler_north_16mm",
    "r09_s03/s040_camera_basler_north_16mm",
    "r09_s09/s040_camera_basler_north_16mm",
    "r09_s12/s040_camera_basler_north_16mm",
    "r09_s05/s050_camera_basler_south_16mm",
]
STANDARD = "DE_autobahn"


def pipeline(video: Path, out: Path):
    """The steps under test: (script, arguments). Baseline (master): edge_accumulate.py,
    vp_gaussian_process.py, calibrate_from_vp.py; here the scene dynamic logic."""
    return [
        ("tools/scene_calibration.py", ["--video", str(video), "--out", str(out), "--standard", STANDARD]),
    ]


def result_of(out: Path) -> dict:
    path = out / "calibration.json"
    if not path.exists():
        path = out / "blur9" / "calibration.json"
    if not path.exists():
        return {"error": "no calibration.json"}
    c = json.loads(path.read_text())
    row = {k: c.get(k) for k in ("vanishing_point", "focal_px", "height_m", "pitch_deg", "lane_lines", "error")}
    ref = c.get("reference")
    if ref and c.get("focal_px"):
        row["err_focal_pct"] = round((c["focal_px"] / ref["focal_px"] - 1) * 100, 2)
        row["err_height_pct"] = round((c["height_m"] / ref["height_m"] - 1) * 100, 2)
        row["err_pitch_deg"] = round(c["pitch_deg"] - ref["pitch_deg"], 3)
    return row


def fmt(row: dict) -> str:
    if not row or row.get("error") or row.get("focal_px") is None:
        return f"FAIL ({(row or {}).get('error', '')[:40]})"
    if "err_focal_pct" in row:
        return f"f {row['err_focal_pct']:+6.1f}%  h {row['err_height_pct']:+6.1f}%  pitch {row['err_pitch_deg']:+5.2f}"
    return f"f {row['focal_px']:.0f}px h {row['height_m']:.2f}m (no reference)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", required=True, help="name of this run (e.g. the branch)")
    parser.add_argument("--compare", nargs="?", const=str(BASELINE), default=None, help="result file to compare with")
    parser.add_argument("--write-baseline", action="store_true", help="store this run as the baseline")
    parser.add_argument("--reuse", action="store_true", help="only collect results already computed")
    args = parser.parse_args()

    results, start = {}, time.time()
    for name in VIDEOS:
        seq, cam = name.split("/")
        video = datasets_root() / "TUMTraf-Accid3nD" / "data" / seq / "images" / cam
        out = OUT / args.label / f"{seq}_{cam}"
        if not args.reuse:
            for script, script_args in pipeline(video, out):
                done = subprocess.run([PYTHON, str(PROJECT_ROOT / script)] + script_args, cwd=PROJECT_ROOT,
                                      capture_output=True, text=True)
                if done.returncode not in (0, 2):
                    print(f"{name}: {script} failed\n{done.stdout[-800:]}{done.stderr[-800:]}")
                    break
        results[name] = result_of(out)
        print(f"{name:42s} {fmt(results[name])}", flush=True)
    path = OUT / f"{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
    if args.write_baseline:
        BASELINE.write_text(json.dumps(results, indent=1))
    print(f"{len(VIDEOS)} videos in {time.time() - start:.0f} s -> {path}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        print(f"\n{'video':42s} {'compared (' + Path(args.compare).stem + ')':40s} {args.label}")
        for name in VIDEOS:
            print(f"{name:42s} {fmt(base.get(name)):40s} {fmt(results[name])}")
        for key in ("err_focal_pct", "err_height_pct"):
            def mean_abs(r):
                v = [abs(x[key]) for x in r.values() if x and key in x]
                return f"{sum(v) / len(v):.2f} ({len(v)} cams)" if v else "-"
            print(f"mean |{key}|: {mean_abs(base)} -> {mean_abs(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
