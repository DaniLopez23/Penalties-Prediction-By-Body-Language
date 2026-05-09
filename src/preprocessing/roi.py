from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import Detection, GoalBox
from src.utils.geometry import clamp


class PlayAreaMasker:
    """Builds the useful penalty corridor from the goal to the striker."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def build_mask(self, frame_shape: tuple[int, int, int], goal: Optional[GoalBox]) -> Optional[np.ndarray]:
        if not self.config.roi.enabled or goal is None:
            return None

        height, width = frame_shape[:2]
        polygon = self._polygon(width, height, goal)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def apply_for_detection(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return frame
        fill = np.full_like(frame, self.config.roi.detection_fill_value)
        return np.where(mask[:, :, None] > 0, frame, fill)

    def apply_blur(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return frame.copy()
        kernel = max(3, self.config.roi.blur_kernel | 1)
        blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
        output = np.where(mask[:, :, None] > 0, frame, blurred)
        if self.config.roi.show_roi_border:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, self.config.roi.roi_border_color, 1, cv2.LINE_AA)
        return output

    def filter_detections(self, detections: list[Detection], mask: Optional[np.ndarray]) -> list[Detection]:
        if mask is None:
            return detections
        return [det for det in detections if self.contains_point(mask, det.center)]

    @staticmethod
    def contains_point(mask: Optional[np.ndarray], point: tuple[float, float]) -> bool:
        if mask is None:
            return True
        height, width = mask.shape[:2]
        x = int(round(point[0]))
        y = int(round(point[1]))
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return mask[y, x] > 0

    def _polygon(self, width: int, height: int, goal: GoalBox) -> np.ndarray:
        goal_w = max(1.0, goal.width)
        center_x = (goal.x1 + goal.x2) * 0.5
        top_expand = goal_w * self.config.roi.top_expand_goal_width
        bottom_half_w = goal_w * self.config.roi.bottom_width_goal_multiplier * 0.5
        bottom_y = int(clamp(height * self.config.roi.bottom_y_ratio, 0, height - 1))
        top_y = int(clamp(goal.y1, 0, height - 1))
        left_top = int(clamp(goal.x1 - top_expand, 0, width - 1))
        right_top = int(clamp(goal.x2 + top_expand, 0, width - 1))
        left_bottom = int(clamp(center_x - bottom_half_w, 0, width - 1))
        right_bottom = int(clamp(center_x + bottom_half_w, 0, width - 1))
        return np.array(
            [
                [left_top, top_y],
                [right_top, top_y],
                [right_bottom, bottom_y],
                [left_bottom, bottom_y],
            ],
            dtype=np.int32,
        )
