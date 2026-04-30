"""Drawing helpers for the simplified penalty pipeline."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COLORS = {
    "shooter": (0, 200, 0),
    "goalkeeper": (0, 0, 255),
    "ball": (0, 255, 255),
    "goal": (255, 255, 0),
    "text": (255, 255, 255),
    "pose": (255, 255, 255),
}


SKELETON_EDGES = [
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_bbox(frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    """Draw a labeled bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_role_detections(
    frame: np.ndarray,
    shooter: Any = None,
    goalkeeper: Any = None,
    ball: Any = None,
    goal: Any = None,
) -> None:
    """Draw detected shooter, goalkeeper, ball, and goal boxes."""
    if shooter is not None:
        draw_bbox(frame, shooter.bbox_xyxy, COLORS["shooter"], f"shooter {shooter.confidence:.2f}")
    if goalkeeper is not None:
        draw_bbox(frame, goalkeeper.bbox_xyxy, COLORS["goalkeeper"], f"goalkeeper {goalkeeper.confidence:.2f}")
    if ball is not None:
        draw_bbox(frame, ball.bbox_xyxy, COLORS["ball"], f"ball {ball.confidence:.2f}")
    if goal is not None:
        draw_bbox(frame, goal.bbox_xyxy, COLORS["goal"], f"goal {goal.confidence:.2f}")


def draw_pose(frame: np.ndarray, pose_result: Any, color: tuple[int, int, int]) -> None:
    """Draw a compact COCO skeleton from pose result raw keypoints."""
    if pose_result is None:
        return

    raw_xy = getattr(pose_result, "raw_xy", None)
    raw_conf = getattr(pose_result, "raw_conf", None)
    if raw_xy is None:
        return

    for start_idx, end_idx in SKELETON_EDGES:
        if start_idx >= len(raw_xy) or end_idx >= len(raw_xy):
            continue
        c1 = float(raw_conf[start_idx]) if raw_conf is not None else 1.0
        c2 = float(raw_conf[end_idx]) if raw_conf is not None else 1.0
        if c1 < 0.2 or c2 < 0.2:
            continue
        x1, y1 = raw_xy[start_idx]
        x2, y2 = raw_xy[end_idx]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    for idx in range(len(raw_xy)):
        conf = float(raw_conf[idx]) if raw_conf is not None else 1.0
        if conf < 0.2:
            continue
        x, y = raw_xy[idx]
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def draw_trajectory(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], max_points: int = 30) -> None:
    """Draw polyline trajectory from a list of points."""
    if not points:
        return
    path = points[-max_points:]
    for i in range(1, len(path)):
        cv2.line(frame, path[i - 1], path[i], color, 2)


def draw_metrics_text(frame: np.ndarray, metrics: Any, frame_idx: int) -> None:
    """Draw metrics panel and frame number."""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = COLORS["text"]

    if metrics is not None:
        if metrics.shooter_shoulder_angle is not None:
            text = f"Shooter Angle: {metrics.shooter_shoulder_angle:.1f} deg"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 30

        if metrics.goalkeeper_movement:
            text = f"GK Movement: {metrics.goalkeeper_movement}"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 30

        if metrics.goalkeeper_body_angle is not None:
            text = f"GK Body Angle: {metrics.goalkeeper_body_angle:.1f} deg"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 30

        if metrics.goalkeeper_reaction_time_ms is not None:
            text = f"GK Reaction: {metrics.goalkeeper_reaction_time_ms:.0f} ms"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)

    cv2.putText(frame, f"Frame: {frame_idx}", (10, frame.shape[0] - 20), font, font_scale, color, thickness)