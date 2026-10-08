"""World model: image detections -> physically consistent, driven world state.

Pure numpy, no Isaac Sim dependency (shared by main.py and the evaluation tools).

coordinate_mapper   image <-> ground plane (homography, frame transform, covariance)
dynamic_logic       per-object filtering: Kalman, motion mode, heading, flow, coasting
scene_consistency   footprints, overlap test (SAT), overlap resolution
world_tracker       per-frame reconstruction: projection -> dynamics -> consistency
vehicle_dynamics    kinematic bicycle + tracking controller that drives each vehicle
pipeline            builds the whole world model from a config
"""
