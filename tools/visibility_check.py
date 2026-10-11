"""Check the visual-condition logic (world/visibility_logic.py).

1. completeness: every property of `verify()` on the sub-logics and the selection, and
   the same check on deliberately broken variants (the check must catch them);
2. classification: the illumination of every video against its label
   (tools/visibility_labels.json; labels judged by eye from a middle frame);
3. optionally (--calibrations DIR...): the atmosphere of finished scene-calibration runs
   (their calibration.json carries the visibility report) against the labels.

Usage:
    C:\\isaacsim\\python.bat tools/visibility_check.py [--videos] [--calibrations DIR ...]
"""

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world import visibility_logic as vl  # noqa: E402

LABELS = PROJECT_ROOT / "tools" / "visibility_labels.json"


def print_verify(logics, title: str) -> bool:
    print(f"\n== {title}")
    ok_all = True
    for name, props in vl.verify(logics).items():
        for prop, (ok, detail) in props.items():
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:14} {prop:15} {detail}")
    return ok_all


def broken_variants():
    """Logics with a known defect: the check must fail on each."""
    yield ("negative hysteresis: the invariants do not overlap, jump* oscillates between them",
           [replace(vl.ILLUMINATION, hysteresis=-5.0), vl.ATMOSPHERE, vl.PRECIPITATION])
    yield ("hysteresis wider than the gap between bounds: twilight's invariant covers night's exit",
           [replace(vl.ILLUMINATION, hysteresis=25.0), vl.ATMOSPHERE, vl.PRECIPITATION])


def check_broken():
    print("\n== broken variants (each must FAIL somewhere)")
    caught = 0
    variants = list(broken_variants())
    for text, logics in variants:
        results = vl.verify(logics)
        failed = [f"{n}.{p}" for n, props in results.items() for p, (ok, _) in props.items() if not ok]
        caught += bool(failed)
        print(f"  {'caught' if failed else 'MISSED'}  {text}: {', '.join(failed) or '-'}")
    # a hole: a SubLogic whose partition skips [0, 0.1) - built by hand (the constructor would not)
    hole = vl.SubLogic("atmosphere", "C_far/near", "ratio", vl.Interval(0.0, vl.INF), ("fog", "haze", "clear"),
                       (0.7, 1.0), 0.05)
    hole.partition = lambda i, f=hole.partition: vl.Interval(0.1, 0.7) if i == 0 else f(i)   # type: ignore
    bad = vl.verify_sublogic(hole)
    failed = [p for p, (ok, _) in bad.items() if not ok]
    print(f"  {'caught' if failed else 'MISSED'}  partition with a hole at [0, 0.1): {', '.join(failed) or '-'}")
    # a selection that asks for more paint in fog than in clear
    orig = vl.select

    def worse_select(i, a, p):
        w = orig(i, a, p)
        return vl.Work(w.calibrate, 2, w.reason) if a == vl.FOG and w.calibrate else w
    vl.select = worse_select
    sel = vl.verify_selection()
    vl.select = orig
    failed2 = [p for p, (ok, _) in sel.items() if not ok]
    print(f"  {'caught' if failed2 else 'MISSED'}  fog reads more paint than haze: {', '.join(failed2) or '-'}")
    return caught == len(variants) and bool(failed) and bool(failed2)


def frame_files(video: str):
    return sorted(glob.glob(video + "/*.jpg") + glob.glob(video + "/*.png"))


def check_videos(labels: dict):
    import cv2
    from cctv_utils.paths import expand

    print("\n== illumination of every labelled video")
    rows, confusion = [], {}
    for name, lab in labels["videos"].items():
        files = frame_files(expand(lab["path"]))
        if len(files) < 10:
            continue
        frames = [cv2.imread(files[int(i)], cv2.IMREAD_GRAYSCALE) for i in np.linspace(0, len(files) - 1, 25)]
        m = vl.illumination_of([f for f in frames if f is not None])
        mode = vl.ILLUMINATION.step(vl.UNKNOWN, m)
        confusion[(lab["illumination"], mode)] = confusion.get((lab["illumination"], mode), 0) + 1
        rows.append((name, m, lab["illumination"], mode))
        print(f"  {'  ' if mode == lab['illumination'] else 'X '}{name:44} L50 {m:6.1f}  label {lab['illumination']:8} -> {mode}")
    modes = list(vl.ILLUMINATION.modes)
    print("\n  label \\ logic " + "".join(f"{m:>10}" for m in modes))
    for a in modes:
        print(f"  {a:14}" + "".join(f"{confusion.get((a, b), 0):10d}" for b in modes))
    right = sum(r[2] == r[3] for r in rows)
    print(f"  accuracy {right}/{len(rows)} = {right / max(len(rows), 1):.1%}")


def check_calibrations(dirs, labels: dict):
    print("\n== atmosphere of finished calibration runs")
    for d in dirs:
        c = json.loads((Path(d) / "calibration.json").read_text())
        v = c.get("visibility") or {}
        ev = (v.get("evidence") or {}).get(vl.ATMOSPHERE.quantity, {})
        name = c.get("video", d)
        label = next((lab.get("atmosphere") for n, lab in labels["videos"].items()
                      if Path(lab["path"]).as_posix().rstrip("/").endswith(Path(name).as_posix().rstrip("/")[-30:])), None)
        print(f"  {Path(d).name:40} C_far/near {ev.get('value')}  -> {v.get('modes', {}).get('atmosphere')}  "
              f"(label {label})  work: {v.get('work')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", action="store_true", help="classify the illumination of every labelled video")
    parser.add_argument("--calibrations", nargs="*", default=[], help="scene-calibration output folders")
    args = parser.parse_args()

    print(vl.describe())
    ok = print_verify(vl.SUBLOGICS, "completeness of the visual-condition logic")
    caught = check_broken()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    if args.videos:
        check_videos(labels)
    if args.calibrations:
        check_calibrations(args.calibrations, labels)
    print(f"\ncompleteness: {'PASS' if ok else 'FAIL'}; broken variants caught: {'all' if caught else 'NOT ALL'}")
    return 0 if ok and caught else 1


if __name__ == "__main__":
    sys.exit(main())
