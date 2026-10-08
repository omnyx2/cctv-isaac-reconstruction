"""Estimate the CCTV camera pose from 4+ image <-> road-plane correspondences.

World frame: X = lateral (right), Y = forward along the road, Z = up, meters.

The focal length is solved from the homography assuming square pixels and a
centered principal point; the pose then comes from solvePnP. Prints a YAML
snippet for the `mapping` and `simulation.camera` config sections.

Usage:
    python tools/calibrate_camera.py --size 960 540 \
        --points "540,335:0,0" "692,335:3.5,0" "540,175:0,21" "620,175:3.5,21"
"""

import argparse

import cv2
import numpy as np


def parse_point(text):
    pixel, world = text.split(":")
    return [float(v) for v in pixel.split(",")], [float(v) for v in world.split(",")]


def solve_focal(G, cx, cy):
    """Focal length (px) that makes K^-1 G columns closest to orthonormal."""
    def residual(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        M = np.linalg.inv(K) @ G
        a, b = M[:, 0], M[:, 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return abs(a @ b) / (na * nb) + abs(na / nb - 1)

    focals = np.arange(200, 5000, 1.0)
    return float(focals[int(np.argmin([residual(f) for f in focals]))])


def main():
    parser = argparse.ArgumentParser(description="CCTV camera calibration from road points")
    parser.add_argument("--size", nargs=2, type=int, required=True, metavar=("W", "H"))
    parser.add_argument("--points", nargs="+", required=True, help='"px,py:wx,wy" pairs')
    parser.add_argument("--focal", type=float, default=None, help="Known focal length in px")
    args = parser.parse_args()

    pairs = [parse_point(p) for p in args.points]
    pixel = np.float32([p for p, _ in pairs])
    world = np.float32([w for _, w in pairs])
    width, height = args.size
    cx, cy = width / 2, height / 2

    H, _ = cv2.findHomography(pixel, world)
    focal = args.focal or solve_focal(np.linalg.inv(H), cx, cy)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    obj = np.float32([[x, y, 0] for x, y in world])
    _, rvec, tvec = cv2.solvePnP(obj, pixel, K, None, flags=cv2.SOLVEPNP_IPPE)
    R, _ = cv2.Rodrigues(rvec)
    position = (-R.T @ tvec).ravel()
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    pitch = np.degrees(np.arcsin(-forward[2]))
    yaw = np.degrees(np.arctan2(-forward[0], forward[1]))

    reproj = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
    error = float(np.abs(reproj - pixel).max())

    print(f"# focal={focal:.0f}px  hfov={np.degrees(2 * np.arctan(cx / focal)):.1f}deg  "
          f"height={position[2]:.2f}m  pitch_down={pitch:.1f}deg  reproj_err={error:.2f}px")
    print("mapping:")
    print("  calibration_points:")
    for p, w in pairs:
        print(f"    - [[{p[0]:g}, {p[1]:g}], [{w[0]:g}, {w[1]:g}]]")
    print("simulation:")
    print("  camera:")
    print(f"    position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
    print(f"    pitch_deg: {pitch:.2f}")
    print(f"    yaw_deg: {yaw:.2f}")
    print(f"    focal_px: {focal:.0f}")
    print(f"    image_size: [{width}, {height}]")


if __name__ == "__main__":
    main()
