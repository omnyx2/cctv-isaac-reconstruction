"""World model: image detections -> physically consistent, driven world state.

Pure numpy, no Isaac Sim dependency (shared by main.py and the evaluation tools).

camera              the world's camera: image <-> road (homography, pinhole model, frame
                    transform, covariance, distances); the one place image and road meet
scene_logic         dynamic logic in the image: static / dynamic edges per pixel, and the
                    calibration automaton that estimates the camera
visibility_logic    dynamic logic of the visual conditions (illumination, atmosphere,
                    precipitation): which calibration work the image allows; completeness
                    checked by verify(); Koschmieder restoration of the road in fog
dynamic_logic       dynamic logic on the road, per road user: Kalman, motion mode, heading,
                    flow, coasting, existence
scene_consistency   footprints, overlap test (SAT), overlap resolution
world_tracker       per-frame reconstruction: projection -> dynamics -> consistency
vehicle_dynamics    kinematic bicycle + tracking controller that drives each vehicle
pipeline            builds the whole world model from a config
"""
