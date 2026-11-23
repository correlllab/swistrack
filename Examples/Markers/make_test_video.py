#!/usr/bin/env python3
"""Renders markers.avi: four AprilTag 36h11 markers moving and rotating in front of a plain background.

Requires Python 3 with OpenCV (pip install opencv-python, or opencv-contrib-python for OpenCV < 4.7).
Usage: python3 make_test_video.py [output.avi]
"""
import math
import sys

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, FRAMES = 720, 480, 10, 80
CELL = 8    # pixels per marker cell: 8 cells (6 data + 2 border) = 64 px markers
MARGIN = 8  # white quiet zone around each marker

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def marker_image(marker_id):
    side = 8 * CELL
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    else:  # OpenCV < 4.7 (contrib)
        img = cv2.aruco.drawMarker(dictionary, marker_id, side)
    return cv2.copyMakeBorder(img, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=255)


def pose(marker_id, t):
    """Center (x, y) [px] and clockwise rotation [deg] of each marker at time t in [0, 1)."""
    a = 2 * math.pi * t
    if marker_id == 3:  # circle, facing along its path
        return 180 + 80 * math.cos(a), 280 + 80 * math.sin(a), math.degrees(a) + 90
    if marker_id == 7:  # circle in the other direction
        return 420 + 80 * math.cos(-a), 280 + 80 * math.sin(-a), -math.degrees(a)
    if marker_id == 12:  # back and forth along the top
        return 60 + 600 * (0.5 - 0.5 * math.cos(a)), 60, 20 * math.sin(2 * a)
    return 650, 420 - 220 * (0.5 - 0.5 * math.cos(a)), 360 * t  # id 25: up and down on the right, spinning


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "markers.avi"
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT), True)
    images = {i: marker_image(i) for i in (3, 7, 12, 25)}
    for frame in range(FRAMES):
        canvas = np.full((HEIGHT, WIDTH), 150, np.uint8)
        for marker_id, img in images.items():
            x, y, angle = pose(marker_id, frame / FRAMES)
            h, w = img.shape
            m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -angle, 1.0)
            m[0, 2] += x - (w - 1) / 2
            m[1, 2] += y - (h - 1) / 2
            cv2.warpAffine(img, m, (WIDTH, HEIGHT), canvas, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    writer.release()


if __name__ == "__main__":
    main()
