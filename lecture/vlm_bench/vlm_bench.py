"""Benchmark vision-language models served by Ollama on CCTV tasks (DGX Spark lab).

Each model answers five tasks over a hand-labeled image set (images/, labels.json): 32 CCTV
frames (UA-DETRAC), 18 crash photos and 8 look-alike non-crash photos (Wikimedia Commons):
  T1 scene_json     the project's own scene prompt (perception/vlm/prompts.py), JSON
  T2 accident       yes/no accident detection on all 58 images
  T3 count          number of vehicles in the near half of a CCTV frame (hand-labeled ranges)
  T4 attributes     bus / truck / people at the bus stop / two-way traffic, JSON
  T5 korean         a two-sentence description in Korean
Timing comes from Ollama's own counters (load, prompt eval = image + prompt, eval = answer).

    python3 -I vlm_bench.py qwen2.5vl:7b [more models...]      -> results/<model>.jsonl
    python3 -I vlm_bench.py --report                            -> results/REPORT.md

Standard library only; talks to OLLAMA_URL (default http://localhost:11434).
"""

import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
RESULTS = HERE / "results"
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

LABELS = json.load(open(HERE / "labels.json"))
CCTV = list(LABELS["cctv"])
PHOTOS = list(LABELS["photos"])
CRASH = [k for k in PHOTOS if LABELS["photos"][k]["accident"]]
LOOKALIKE = [k for k in PHOTOS if not LABELS["photos"][k]["accident"]]
# CCTV frames: no accident, no emergency vehicle (checked by eye)
ACCIDENT = {**{k: False for k in CCTV}, **{k: v["accident"] for k, v in LABELS["photos"].items()}}
EMERGENCY = {**{k: False for k in CCTV}, **{k: v["emergency"] for k, v in LABELS["photos"].items()}}
ATTRIBUTES = {k: {a: v[a] for a in ("bus", "truck", "people_waiting", "two_way")} for k, v in LABELS["cctv"].items()}
COUNT_RANGE = {k: tuple(v["count"]) for k, v in LABELS["cctv"].items()}
NUM_CTX = 8192   # the same context window for every model, so memory use compares fairly

SCENE_PROMPT = """You are analyzing one frame of a fixed roadside traffic CCTV camera.
Describe only what is visible. Answer with a single JSON object and nothing else:
{
  "summary": "<one sentence: the traffic situation>",
  "accident": <true if a collision, crashed, overturned or burning vehicle is visible, else false>,
  "accident_description": "<what and where in the image, or empty>",
  "stopped_vehicles_in_lanes": <number of vehicles standing still in travel lanes>,
  "people_on_road": <true if people are on the carriageway or shoulder, else false>,
  "smoke_or_fire": <true or false>,
  "emergency_vehicles": <true if police, ambulance, fire or tow vehicles are visible, else false>,
  "hazards": ["<short phrase>", ...],
  "confidence": <0.0 to 1.0>
}"""
ACCIDENT_PROMPT = ("Is there a traffic accident in this image (a collision, or a crashed, overturned or "
                   "badly damaged vehicle)? Answer with exactly one word: yes or no.")
COUNT_PROMPT = ("How many vehicles (cars, vans, buses, trucks) are visible in this image, including vehicles "
                "partly cut off at the edges? Answer with a single integer only.")
ATTR_PROMPT = """Look at this traffic camera image and answer with a single JSON object and nothing else:
{"bus": <true if a full-size bus or coach is visible (not a minivan)>,
 "truck": <true if a truck or lorry is visible>,
 "people_waiting": <true if people are standing or waiting at a bus stop or by the road>,
 "two_way": <true if traffic moves in both directions on this road>}"""
KOREAN_PROMPT = "이 CCTV 장면을 한국어로 두 문장으로 설명해줘."

TASKS = [  # (task, prompt, json mode, images)
    ("scene_json", SCENE_PROMPT, True, CCTV + PHOTOS),
    ("accident", ACCIDENT_PROMPT, False, CCTV + PHOTOS),
    ("count", COUNT_PROMPT, False, [k + "_near" for k in CCTV]),
    ("attributes", ATTR_PROMPT, True, CCTV),
    ("korean", KOREAN_PROMPT, False, LABELS["korean"]),
]


def capabilities(model):
    req = urllib.request.Request(URL + "/api/show", json.dumps({"model": model}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read()).get("capabilities", [])


THINKING = set()   # models that reason before answering: asked not to (think=false), else
                   # the 400-token budget goes to hidden reasoning and the answer comes back empty


def call(model, prompt, image, json_mode, keep_alive="15m"):
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "seed": 0, "num_predict": 400, "num_ctx": NUM_CTX}}
    if model in THINKING:
        body["think"] = False
    if image:
        body["images"] = [base64.b64encode((IMAGES / f"{image}.jpg").read_bytes()).decode()]
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(URL + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        error = None
    except Exception as e:   # noqa: BLE001 - recorded, the run continues
        out, error = {}, f"{type(e).__name__}: {e}"
    ns = 1e9
    return {"answer": out.get("response", ""), "error": error, "wall_s": time.time() - t0,
            "load_s": out.get("load_duration", 0) / ns,
            "prompt_tokens": out.get("prompt_eval_count", 0), "prompt_s": out.get("prompt_eval_duration", 0) / ns,
            "answer_tokens": out.get("eval_count", 0), "answer_s": out.get("eval_duration", 0) / ns}


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes")
    return None


def grade(task, image, answer):
    """-> dict of named checks, each True / False (None = could not parse)."""
    key = image.replace("_near", "")
    if task == "scene_json":
        d = parse_json(answer)
        if d is None:
            return {"json_ok": False, "accident_ok": None, "emergency_ok": None}
        checks = {"json_ok": True, "accident_ok": as_bool(d.get("accident")) == ACCIDENT[key]}
        if EMERGENCY[key] is not None:   # null = ambiguous by eye: the key is left out, not scored
            checks["emergency_ok"] = as_bool(d.get("emergency_vehicles")) == EMERGENCY[key]
        return checks
    if task == "accident":
        word = re.findall(r"[a-z]+", answer.lower())
        said = word[0] == "yes" if word and word[0] in ("yes", "no") else None
        return {"format_ok": said is not None, "accident_ok": None if said is None else said == ACCIDENT[key]}
    if task == "count":
        nums = re.findall(r"\d+", answer)
        if not nums:
            return {"format_ok": False, "in_range": None, "abs_err": None}
        n, (lo, hi) = int(nums[0]), COUNT_RANGE[key]
        return {"format_ok": True, "in_range": lo <= n <= hi, "abs_err": max(lo - n, n - hi, 0), "n": n}
    if task == "attributes":
        d = parse_json(answer)
        if d is None:
            return {"json_ok": False}
        checks = {"json_ok": True}
        for k, v in ATTRIBUTES[key].items():
            if v is not None:   # null = ambiguous by eye, not scored
                checks[k + "_ok"] = as_bool(d.get(k)) == v
        return checks
    if task == "korean":
        letters = [c for c in answer if c.isalpha()]
        hangul = sum("가" <= c <= "힣" for c in letters)
        return {"korean_ok": bool(letters) and hangul / len(letters) > 0.6}
    return {}


def run(model):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (model.replace(":", "_").replace("/", "_") + ".jsonl")
    print(f"== {model}", flush=True)
    if "thinking" in capabilities(model):
        THINKING.add(model)
        print("   thinking model: think=false", flush=True)
    warm = call(model, "Reply with OK.", None, False)   # load the model, time it apart from the tasks
    with open(out, "w") as f:
        f.write(json.dumps({"model": model, "task": "_load", **warm}) + "\n")
        print(f"   load {warm['load_s']:.1f}s  {warm['error'] or ''}", flush=True)
        ps = json.loads(urllib.request.urlopen(URL + "/api/ps").read())
        for m in ps.get("models", []):
            if m["name"] == model or m.get("model") == model:
                f.write(json.dumps({"model": model, "task": "_memory", "size_gb": m["size"] / 1e9,
                                    "vram_gb": m.get("size_vram", 0) / 1e9}) + "\n")
        for task, prompt, json_mode, images in TASKS:
            for image in images:
                r = call(model, prompt, image, json_mode)
                r.update(model=model, task=task, image=image, checks=grade(task, image, r["answer"]))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                short = r["error"] or r["answer"].replace("\n", " ")[:70]
                print(f"   {task:10} {image:16} {r['wall_s']:6.1f}s  {short}", flush=True)
    call(model, "", None, False, keep_alive=0)   # unload: free the memory for the next model
    print(f"   -> {out}", flush=True)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def report():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        recs = [json.loads(line) for line in open(path)]
        model = recs[0]["model"]
        load = next((r for r in recs if r["task"] == "_load"), {})
        mem = next((r for r in recs if r["task"] == "_memory"), {})
        tasks = [r for r in recs if not r["task"].startswith("_")]

        def score(task, check):
            vals = [r["checks"].get(check) for r in tasks if r["task"] == task]
            vals = [bool(v) for v in vals if v is not None] + [False] * sum(v is None for v in vals)
            return mean([1.0 if v else 0.0 for v in vals])

        timed = [r for r in tasks if not r["error"]]
        tok_s = mean([r["answer_tokens"] / r["answer_s"] for r in timed if r["answer_s"] > 0])
        scene = [r for r in timed if r["task"] == "scene_json"]
        acc = [r for r in tasks if r["task"] == "accident"]
        tp = sum(1 for r in acc if r["image"] in CRASH and r["checks"].get("accident_ok"))
        fp = sum(1 for r in acc if r["image"] in CCTV and r["checks"].get("accident_ok") is False)
        fp_look = sum(1 for r in acc if r["image"] in LOOKALIKE and r["checks"].get("accident_ok") is False)
        attr = [r for r in tasks if r["task"] == "attributes"]
        attr_checks = [v for r in attr for k, v in r["checks"].items() if k != "json_ok"]
        korean = [r for r in tasks if r["task"] == "korean"]
        rows.append({
            "model": model, "size_gb": mem.get("size_gb"), "load_s": load.get("load_s"),
            "errors": sum(1 for r in tasks if r["error"]),
            "scene_s": mean([r["wall_s"] for r in scene]), "tok_s": tok_s,
            "prompt_s": mean([r["prompt_s"] for r in timed]),
            "scene_json": score("scene_json", "json_ok"), "scene_acc": score("scene_json", "accident_ok"),
            "scene_emerg": score("scene_json", "emergency_ok"),
            "acc_tp": tp, "acc_fp": fp, "acc_fp_look": fp_look, "acc_ok": score("accident", "accident_ok"),
            "count_in": score("count", "in_range"),
            "count_err": mean([r["checks"].get("abs_err") for r in tasks if r["task"] == "count"]),
            "attr": mean([1.0 if v else 0.0 for v in attr_checks]) if attr_checks else None,
            "korean": score("korean", "korean_ok"),
            "korean_text": [r["answer"].strip().replace("\n", " ") for r in korean],
        })

    def f(v, fmt="{:.0%}"):
        return "–" if v is None else fmt.format(v)

    lines = ["# VLM benchmark on DGX Spark (GB10, 128 GB unified memory)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `vlm_bench.py --report`. Ollama, temperature 0, "
             f"one request at a time, num_ctx {NUM_CTX}, Isaac Sim not running. "
             f"{len(CCTV)} CCTV frames, {len(CRASH)} crash photos, {len(LOOKALIKE)} look-alike photos.", "",
             "## Speed and memory", "",
             "| Model | Memory (GB) | Load (s) | Scene JSON answer (s) | Image+prompt eval (s) | Answer tokens/s | Errors |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {f(r['size_gb'], '{:.1f}')} | {f(r['load_s'], '{:.1f}')} | "
                     f"{f(r['scene_s'], '{:.1f}')} | {f(r['prompt_s'], '{:.2f}')} | {f(r['tok_s'], '{:.1f}')} | {r['errors']} |")
    lines += ["", "## Task quality", "",
              f"| Model | T1 JSON valid | T1 accident right | T1 emergency right | T2 crashes found (of {len(CRASH)}) | "
              f"T2 false alarms: CCTV (of {len(CCTV)}) | T2 false alarms: look-alikes (of {len(LOOKALIKE)}) | "
              "T3 count in range | T3 mean miss | T4 attributes right | T5 Korean |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {f(r['scene_json'])} | {f(r['scene_acc'])} | {f(r['scene_emerg'])} | "
                     f"{r['acc_tp']} | {r['acc_fp']} | {r['acc_fp_look']} | {f(r['count_in'])} | {f(r['count_err'], '{:.2f}')} | "
                     f"{f(r['attr'])} | {f(r['korean'])} |")
    skipped = RESULTS / "skipped.txt"
    if skipped.exists():
        lines += ["", "## Not measured", ""] + [f"- {s}" for s in skipped.read_text().splitlines() if s.strip()]
    lines += ["", "## Korean answers (T5: " + ", ".join(LABELS["korean"]) + ")", ""]
    for r in rows:
        lines.append(f"- **{r['model']}**")
        for t in r["korean_text"]:
            lines.append(f"  - {t[:300]}")
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n")
    json.dump(rows, open(RESULTS / "summary.json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:] == ["--report"]:
        report()
    else:
        for m in sys.argv[1:]:
            run(m)
