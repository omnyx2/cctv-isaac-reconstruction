"""Perception: video -> tracked road users in image space (no Isaac Sim dependency).

video_player       decode frames
yolo_tracker       YOLOv8 + ByteTrack detection/tracking
stabilizer         camera motion estimate against the calibration frame
perception_worker  runs the above in a separate process, one frame ahead
"""
