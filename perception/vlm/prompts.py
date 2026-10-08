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
