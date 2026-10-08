"""Mine near-miss (traffic conflict) moments from all cached UA-DETRAC sequences.

Per sequence: road geometry from the traffic (analysis/self_calibration.py), then
scale-free conflict measures (analysis/near_miss.py). Outputs to
data/results/near_miss/:
    events.json       all events, ranked by severity
    REPORT.md         per-sequence summary + top events
    clips/            original-frame clips (+-1.5 s) of the top events, vehicles highlighted
    top_events.jpg    peak frames of the top events

Run tools/cache_detections.py first.

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/mine_near_misses.py [--top 20]
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.near_miss import CANDIDATE_THRESHOLDS, THRESHOLDS, build_tracks, find_events  # noqa: E402
from analysis.self_calibration import estimate_road_frame  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DETRAC_ROOT = PROJECT_ROOT / "data" / "UA-DETRAC" / "train"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "near_miss"
LOW_CONFIDENCE_VP_RESIDUAL = 8.0     # px
LOW_CONFIDENCE_MIN_LINES = 8

KIND_LABEL = {"rear_end": "rear-end TTC_ff", "tailgating": "tailgating THW",
              "hard_braking": "hard braking", "abrupt_lateral": "abrupt lateral"}
KIND_UNIT = {"rear_end": "s", "tailgating": "s", "hard_braking": "speed drop in 1 s",
             "abrupt_lateral": "m/s"}


def sequence_dir(sequence: str) -> Path:
    return next(DETRAC_ROOT.glob(f"detrac_*_{sequence}"))


def format_value(kind: str, value: float) -> str:
    if kind == "hard_braking":
        return f"{value * 100:.0f}% speed drop in 1 s"
    return f"{value:.2f} {KIND_UNIT[kind]}"


def render_clip(sequence: str, cache: dict, event: dict, path: Path, fps: float):
    """Original frames around the event peak with all vehicles drawn, involved ones highlighted."""
    images = sorted(sequence_dir(sequence).glob("image*.jpg"))
    half = int(1.5 * fps)
    lo, hi = max(0, event["start"] - half), min(len(images) - 1, event["end"] + half)
    involved = set(event["track_ids"])
    first = cv2.imread(str(images[lo]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    peak_frame = None
    for f in range(lo, hi + 1):
        frame = cv2.imread(str(images[f]))
        for tid, (x1, y1, x2, y2), name in cache["frames"][f]:
            hit = tid in involved
            color = (0, 0, 255) if hit else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if hit else 1)
            if hit:
                cv2.putText(frame, f"ID {tid}", (x1, max(12, y1 - 6)), 0, 0.5, color, 2)
        in_event = event["start"] <= f <= event["end"]
        label = f"{sequence}  {KIND_LABEL[event['kind']]}: {format_value(event['kind'], event['value'])}"
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 21), 0, 0.6, (0, 0, 255) if in_event else (255, 255, 255), 2)
        cv2.putText(frame, f"frame {f + 1}", (w - 130, 21), 0, 0.55, (255, 255, 255), 1)
        writer.write(frame)
        if f == event["peak"]:
            peak_frame = frame.copy()
    writer.release()
    return peak_frame


def main():
    parser = argparse.ArgumentParser(description="Mine near-miss moments from UA-DETRAC")
    parser.add_argument("--top", type=int, default=20, help="Events to render as clips")
    args = parser.parse_args()

    caches = sorted(CACHE_DIR.glob("detrac_MVI_*_detections.pkl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clips").mkdir(exist_ok=True)

    all_events, summary = [], []
    for path in caches:
        cache = pickle.loads(path.read_bytes())
        sequence = path.name[len("detrac_"):-len("_detections.pkl")]
        try:
            road = estimate_road_frame(cache["frames"], cache["transforms"], cache["size"])
        except ValueError as e:
            summary.append({"sequence": sequence, "status": f"skipped: {e}"})
            print(f"{sequence}: skipped ({e})")
            continue
        low_conf = (road.vp_residual_px > LOW_CONFIDENCE_VP_RESIDUAL
                    or road.n_tracks < LOW_CONFIDENCE_MIN_LINES)
        tracks = build_tracks(cache["frames"], cache["transforms"], cache["size"], road, cache["fps"])
        rejected = {}
        events = find_events(tracks, len(cache["frames"]), cache["fps"], rejected, CANDIDATE_THRESHOLDS)
        for e in events:
            all_events.append({"sequence": sequence, "kind": e.kind, "start": e.start, "end": e.end,
                               "peak": e.peak, "track_ids": list(e.track_ids), "value": e.value,
                               "severity": e.severity, "conflict": e.is_conflict, "details": e.details,
                               "low_confidence_geometry": low_conf})
        counts = Counter(e.kind for e in events if e.is_conflict)
        candidates = len(events)
        summary.append({"sequence": sequence, "status": "ok", "frames": len(cache["frames"]),
                        "vehicles": len(tracks), "vp": [round(v, 1) for v in road.vanishing_point],
                        "vp_residual_px": round(road.vp_residual_px, 1), "vp_lines": road.n_tracks,
                        "camera_height_m": round(road.height_m, 2), "low_confidence": low_conf,
                        "rejected_artifacts": sum(rejected.values()), "candidates": candidates,
                        **{k: counts.get(k, 0) for k in KIND_LABEL}})
        print(f"{sequence}: {len(tracks)} vehicles, conflicts {dict(counts)}, closest-call candidates {candidates}"
              f"{'  [low-confidence geometry]' if low_conf else ''}", flush=True)

    all_events.sort(key=lambda e: (e["low_confidence_geometry"], -e["severity"]))
    (OUT_DIR / "events.json").write_text(json.dumps({"thresholds": THRESHOLDS,
                                                     "candidate_thresholds": CANDIDATE_THRESHOLDS,
                                                     "summary": summary, "events": all_events}, indent=1))

    # Clips and peak frames of the top events
    tiles = []
    cache_by_seq = {}
    for rank, event in enumerate(all_events[:args.top], 1):
        seq = event["sequence"]
        if seq not in cache_by_seq:
            cache_by_seq = {seq: pickle.loads((CACHE_DIR / f"detrac_{seq}_detections.pkl").read_bytes())}
        clip = OUT_DIR / "clips" / f"{rank:02d}_{seq}_{event['kind']}_f{event['peak'] + 1}.mp4"
        peak = render_clip(seq, cache_by_seq[seq], event, clip, cache_by_seq[seq]["fps"])
        event["clip"] = str(clip.relative_to(PROJECT_ROOT))
        if peak is not None:
            tile = cv2.resize(peak, (480, 270))
            cv2.putText(tile, f"#{rank}", (8, 262), 0, 0.8, (0, 255, 255), 2)
            tiles.append(tile)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT_DIR / "top_events.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    (OUT_DIR / "events.json").write_text(json.dumps({"thresholds": THRESHOLDS,
                                                     "candidate_thresholds": CANDIDATE_THRESHOLDS,
                                                     "summary": summary, "events": all_events}, indent=1))

    # Report
    lines = ["# Near-miss mining report (UA-DETRAC)", "",
             "Generated by `tools/mine_near_misses.py`. Method and caveats: docs/TECHNICAL_NOTES.md "
             "(near-miss section). TTC_ff/THW are front-to-front (include one vehicle length), so "
             "they are conservative (true TTC is shorter). No absolute speeds (scale unknown).", "",
             "## Thresholds (conflict / closest-call candidate)", ""]
    lines += [f"- `{k}` = {v} / {CANDIDATE_THRESHOLDS[k]}" for k, v in THRESHOLDS.items()]
    n_conflicts = sum(e["conflict"] for e in all_events)
    lines += ["", f"Conflicts (beyond the conflict thresholds): **{n_conflicts}**. "
              f"Closest-call candidates: {len(all_events)}.", "",
              f"## Top {min(args.top, len(all_events))} (ranked by severity; >= 1.00 = conflict)", "",
              "| # | sequence | kind | value | severity | frames | vehicles | clip |",
              "|---|---|---|---|---|---|---|---|"]
    for rank, e in enumerate(all_events[:args.top], 1):
        lines.append(f"| {rank} | {e['sequence']} | {KIND_LABEL[e['kind']]} | {format_value(e['kind'], e['value'])} | "
                     f"{e['severity']:.2f}{' **conflict**' if e['conflict'] else ''} | "
                     f"{e['start'] + 1}-{e['end'] + 1} (peak {e['peak'] + 1}) | {', '.join(map(str, e['track_ids']))} | "
                     f"`{Path(e.get('clip', '')).name}` |")
    lines += ["", "## Per sequence", "",
              "| sequence | vehicles | vp residual (px) | camera h (m) | conflicts: rear-end | tailgating | hard braking "
              "| abrupt lateral | candidates | rejected artifacts | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if s["status"] != "ok":
            lines.append(f"| {s['sequence']} | | | | | | | | | | {s['status']} |")
            continue
        lines.append(f"| {s['sequence']} | {s['vehicles']} | {s['vp_residual_px']} | {s['camera_height_m']} | "
                     f"{s['rear_end']} | {s['tailgating']} | {s['hard_braking']} | {s['abrupt_lateral']} | "
                     f"{s['candidates']} | {s['rejected_artifacts']} | "
                     f"{'low-confidence geometry' if s['low_confidence'] else ''} |")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(all_events)} events; report: {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
