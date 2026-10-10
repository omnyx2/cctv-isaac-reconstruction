"""Camera motion compensation.

The ground-plane homography is calibrated on one reference frame. If the camera
shakes or drifts (wind, handheld footage), every pixel shifts and parked vehicles
appear to move. The stabilizer estimates, per frame, a similarity transform from
the current frame to the reference frame using background features, so detections
can be mapped back into the calibrated reference view.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraStabilizer:
    """Estimates current-frame -> reference-frame image transforms."""

    def __init__(self, max_features: int = 1500, scale: float = 0.5,
                 min_inliers: int = 30, ransac_threshold: float = 2.0, work_width: int = 480):
        """
        Args:
            max_features: ORB features per frame
            scale: Image downscale for feature matching (speed), at most; frames are
                matched at no more than `work_width` pixels wide (ORB on a 1920 px frame
                halved still took 45 ms, more than a 25 fps frame budget)
            min_inliers: Below this the previous transform is kept
            ransac_threshold: RANSAC reprojection threshold in full-res pixels
            work_width: Width (px) the frames are matched at, at most
        """
        self.max_scale = scale
        self.scale = scale
        self.work_width = work_width
        self.min_inliers = min_inliers
        self.ransac_threshold_full = ransac_threshold
        self.ransac_threshold = ransac_threshold * scale
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_keypoints = None
        self._ref_descriptors = None
        self._last = np.eye(3)
        self.static_mask: Optional[np.ndarray] = None   # full-res, nonzero where the scene is static

    def set_static_mask(self, mask: Optional[np.ndarray]):
        """Features only where the scene is static (e.g. the static edges of
        world/scene_logic.py, dilated): traffic is not matched even without vehicle boxes."""
        self.static_mask = None if mask is None else (np.asarray(mask) > 0).astype(np.uint8) * 255

    def _features(self, frame: np.ndarray, exclude_boxes: Optional[List[Tuple]] = None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale)
        mask = np.full(gray.shape, 255, np.uint8)
        if self.static_mask is not None:
            mask = cv2.resize(self.static_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        for x1, y1, x2, y2 in exclude_boxes or []:
            mask[int(y1 * self.scale):int(y2 * self.scale) + 1,
                 int(x1 * self.scale):int(x2 * self.scale) + 1] = 0
        return self._orb.detectAndCompute(gray, mask)

    def set_reference(self, frame: np.ndarray, exclude_boxes: Optional[List[Tuple]] = None):
        """Use this frame as the calibrated reference view."""
        self.scale = min(self.max_scale, self.work_width / frame.shape[1])
        self.ransac_threshold = self.ransac_threshold_full * self.scale
        self._ref_keypoints, self._ref_descriptors = self._features(frame, exclude_boxes)
        self._last = np.eye(3)

    @property
    def has_reference(self) -> bool:
        return self._ref_descriptors is not None

    def estimate(self, frame: np.ndarray, exclude_boxes: Optional[List[Tuple]] = None) -> np.ndarray:
        """
        3x3 transform mapping current-frame pixels to reference-frame pixels.

        Vehicle boxes should be excluded: moving traffic is not part of the scene
        the camera moves against. Falls back to the last good transform.
        """
        if not self.has_reference:
            self.set_reference(frame, exclude_boxes)
            return self._last

        keypoints, descriptors = self._features(frame, exclude_boxes)
        if descriptors is None or len(keypoints) < self.min_inliers:
            return self._last

        matches = self._matcher.match(descriptors, self._ref_descriptors)
        if len(matches) < self.min_inliers:
            return self._last

        src = np.float32([keypoints[m.queryIdx].pt for m in matches])
        dst = np.float32([self._ref_keypoints[m.trainIdx].pt for m in matches])
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold)
        if M is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return self._last

        # Back to full-resolution pixel coordinates
        S = np.diag([self.scale, self.scale, 1.0])
        self._last = np.linalg.inv(S) @ np.vstack([M, [0, 0, 1]]) @ S
        return self._last
