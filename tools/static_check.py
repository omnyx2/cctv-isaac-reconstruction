"""Check the static-object logic (world/static_objects.py): the agreement sub-logic, the
selection (total; paint and tentative names never placed), and that the check fails on
deliberately broken selections.

Usage:
    C:\\isaacsim\\python.bat tools/static_check.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world import static_objects as so  # noqa: E402


def report(results) -> bool:
    ok_all = True
    for name, props in results.items():
        for prop, (ok, detail) in props.items():
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:10} {prop:24} {detail}")
    return ok_all


def main():
    print(so.AGREEMENT.automaton().describe())
    print("\n== completeness of the static-object logic")
    ok = report(so.verify())
    print("\n== broken selections (each must FAIL somewhere)")
    orig, caught = so.select, []
    for text, fn in (("paint placed when the VLM is sure", lambda a, p, k: so.PLACE if p and a == so.CONFIRMED else orig(a, p, k)),
                     ("a tentative name placed", lambda a, p, k: so.PLACE if a == so.TENTATIVE and k == "post" else orig(a, p, k))):
        so.select = fn
        failed = [f"{n}.{q}" for n, props in so.verify().items() for q, (good, _) in props.items() if not good]
        so.select = orig
        caught.append(bool(failed))
        print(f"  {'caught' if failed else 'MISSED'}  {text}: {', '.join(failed) or '-'}")
    print(f"\ncompleteness: {'PASS' if ok else 'FAIL'}; broken selections caught: {'all' if all(caught) else 'NOT ALL'}")
    return 0 if ok and all(caught) else 1


if __name__ == "__main__":
    sys.exit(main())
