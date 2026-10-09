"""Prompts for the vision-language model.

The model answers in JSON so the result can be used by code (perception/vlm/analyzer.py
parses it leniently). Keys that the model omits are left empty, never guessed.
"""

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

# Keys of SCENE_PROMPT's answer, with the type each is converted to
SCENE_FIELDS = {
    "summary": str,
    "accident": bool,
    "accident_description": str,
    "stopped_vehicles_in_lanes": int,
    "people_on_road": bool,
    "smoke_or_fire": bool,
    "emergency_vehicles": bool,
    "hazards": list,
    "confidence": float,
}

ACCIDENT_YESNO_PROMPT = """Look carefully at every vehicle in this roadside traffic camera image.
Is there a car crash, a collision, an overturned or damaged vehicle, debris on the road, or a burning vehicle?
Answer "yes" or "no" first, then say where in the image (left/right/top/bottom) and what you see."""

DESCRIBE_PROMPT = """Describe this roadside traffic camera image in detail: the road, the traffic,
and anything unusual (stopped vehicles, damaged vehicles, debris, people on the road, smoke, emergency vehicles)."""

# name -> (prompt, ask the server for JSON). Free-text answers starting with yes/no set `accident`.
PRESETS = {
    "scene_json": (SCENE_PROMPT, True),
    "accident_yesno": (ACCIDENT_YESNO_PROMPT, False),
    "describe": (DESCRIBE_PROMPT, False),
}
