"""Build the detector input frame used by ball and player models."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import ModelConfig


def make_goal_focus_mask(
    frame_shape: tuple[int, ...],
    goal,
    tight: bool = False,
) -> np.ndarray | None:
    """Return pixels the player/ball detectors are allowed to see."""
    if goal is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = goal.bbox_xyxy
    goal_w = max(1, x2 - x1)
    goal_h = max(1, y2 - y1)

    if tight:
        side_ratio = ModelConfig.MASK_SIDE_PADDING_RATIO
        top_ratio = ModelConfig.MASK_TOP_PADDING_RATIO
        field_ratio = ModelConfig.MASK_FIELD_BELOW_GOAL_RATIO
    else:
        side_ratio = ModelConfig.PLAYER_MASK_SIDE_PADDING_RATIO
        top_ratio = ModelConfig.PLAYER_MASK_TOP_PADDING_RATIO
        field_ratio = ModelConfig.PLAYER_MASK_FIELD_BELOW_GOAL_RATIO

    side_pad = int(goal_w * side_ratio)
    top_pad = int(goal_h * top_ratio)
    field_pad = int(goal_h * field_ratio)

    keep_x1 = max(0, x1 - side_pad)
    keep_x2 = min(w, x2 + side_pad)
    keep_y1 = max(0, y1 - top_pad)
    field_y = min(h, y2 + field_pad)

    keep = np.zeros((h, w), dtype=bool)
    keep[keep_y1:field_y, keep_x1:keep_x2] = True
    if tight:
        keep[field_y:, keep_x1:keep_x2] = True
    else:
        keep[field_y:, :] = True
    return keep


def apply_blur_outside_mask(frame: np.ndarray, keep_mask: np.ndarray | None) -> np.ndarray:
    if keep_mask is None:
        return frame.copy()
    kernel = ModelConfig.MASK_BLUR_KERNEL
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    masked = cv2.GaussianBlur(frame, (kernel, kernel), 0) if kernel > 1 else frame.copy()
    masked[keep_mask] = frame[keep_mask]
    return masked


def make_inference_frame(frame: np.ndarray, goal, tight: bool = False) -> np.ndarray:
    """Blur irrelevant areas before any detector sees the frame."""
    return apply_blur_outside_mask(frame, make_goal_focus_mask(frame.shape, goal, tight))
