"""Check the traffic-situation logic (world/traffic_logic.py).

1. completeness: every property of verify() on the three sub-logics and the selection,
   and the same check on deliberately broken variants (it must catch them);
2. optionally (--runs DIR ...): the window modes and the work of finished scene-
   calibration runs (their traffic.json).

Usage:
    C:\\isaacsim\\python.bat tools/traffic_check.py [--runs DIR ...]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world import sublogic  # noqa: E402
from world import traffic_logic as tl  # noqa: E402


def print_verify(results, title: str) -> bool:
    print(f"\n== {title}")
    ok_all = True
    for name, props in results.items():
        for prop, (ok, detail) in props.items():
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:10} {prop:16} {detail}")
    return ok_all


def check_broken() -> bool:
    print("\n== broken variants (each must FAIL somewhere)")
    caught = []
    for text, logic in (("density band wider than the gap (LIGHT kept at O = 0.32)", replace(tl.DENSITY, hysteresis=0.2)),
                        ("standing band negative (jump* oscillates)", replace(tl.STANDING, hysteresis=-0.01))):
        failed = [p for p, (ok, _) in sublogic.verify_sublogic(logic).items() if not ok]
        caught.append(bool(failed))
        print(f"  {'caught' if failed else 'MISSED'}  {text}: {', '.join(failed) or '-'}")
    orig = tl.select

    def admits_congested(speed, density, standing):
        w = orig(speed, density, standing)
        return tl.WindowWork(1, tl.BASE_DROP, w.reason) if density == tl.CONGESTED else w

    def drop_lower_when_slow(speed, density, standing):
        w = orig(speed, density, standing)
        return tl.WindowWork(w.admit, 0.1, w.reason) if speed == tl.SLOW and w.admit else w

    for text, fn in (("a congested window is admitted", admits_congested),
                     ("slow traffic gets a lower drop", drop_lower_when_slow)):
        tl.select = fn
        failed = [p for p, (ok, _) in tl.verify_selection().items() if not ok]
        tl.select = orig
        caught.append(bool(failed))
        print(f"  {'caught' if failed else 'MISSED'}  {text}: {', '.join(failed) or '-'}")
    return all(caught)


def show_runs(dirs):
    print("\n== traffic of finished runs")
    for d in dirs:
        p = Path(d) / "traffic.json"
        if not p.exists():
            print(f"  {d}: no traffic.json")
            continue
        t = json.loads(p.read_text())
        s = t["summary"]
        line = "".join(("." if w["drop"] == tl.BASE_DROP else "s") if w["admit"] else "x" for w in t["windows"])
        print(f"  {Path(d).name:42} {s['admitted']:5d}/{s['frames']:5d} frames  {line}")
    print("  (per window: . admitted, s admitted with the slow paint drop, x left out)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="scene-calibration output folders")
    args = parser.parse_args()
    print(tl.describe())
    ok = print_verify(tl.verify(), "completeness of the traffic logic")
    caught = check_broken()
    if args.runs:
        show_runs(args.runs)
    print(f"\ncompleteness: {'PASS' if ok else 'FAIL'}; broken variants caught: {'all' if caught else 'NOT ALL'}")
    return 0 if ok and caught else 1


if __name__ == "__main__":
    sys.exit(main())
