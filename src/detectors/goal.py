from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import GoalBox
from src.utils.geometry import clamp


class GoalDetector:
    """Detect and smooth the goal every frame so small camera motion is followed."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.tracked_goal: Optional[GoalBox] = None
        self.missed_frames = 0

    def detect(self, frame: np.ndarray) -> Optional[GoalBox]:
        self.frame_index += 1
        raw_goal = self._detect_from_mask(frame)
        if raw_goal is not None:
            if self._is_consistent_with_track(raw_goal, frame.shape):
                self.missed_frames = 0
                self.tracked_goal = self._smooth(raw_goal)
                return self.tracked_goal

        self.missed_frames += 1
        if self.tracked_goal is not None and self.missed_frames <= self.config.goal.keep_last_for_frames:
            return GoalBox(
                self.tracked_goal.x1,
                self.tracked_goal.y1,
                self.tracked_goal.x2,
                self.tracked_goal.y2,
                max(0.08, self.tracked_goal.confidence * 0.95),
                detected=False,
                tracked=True,
            )

        fallback = self._fallback_goal(frame)
        self.tracked_goal = fallback
        return fallback

    def _is_consistent_with_track(self, goal: GoalBox, frame_shape: tuple[int, int, int]) -> bool:
        if self.tracked_goal is None:
            return True
        frame_h, frame_w = frame_shape[:2]
        max_shift = max(1.0, ((frame_w**2 + frame_h**2) ** 0.5) * self.config.goal.max_tracking_shift_ratio)
        prev_cx, prev_cy = self.tracked_goal.center
        next_cx, next_cy = goal.center
        shift = ((next_cx - prev_cx) ** 2 + (next_cy - prev_cy) ** 2) ** 0.5
        width_change = abs(goal.width - self.tracked_goal.width) / max(1.0, self.tracked_goal.width)
        height_change = abs(goal.height - self.tracked_goal.height) / max(1.0, self.tracked_goal.height)
        size_ok = (
            width_change <= self.config.goal.max_tracking_size_change_ratio
            and height_change <= self.config.goal.max_tracking_size_change_ratio
        )
        if shift <= max_shift and size_ok:
            return True
        return self.missed_frames > self.config.goal.keep_last_for_frames

    def _smooth(self, goal: GoalBox) -> GoalBox:
        if self.tracked_goal is None or not self.tracked_goal.detected:
            return GoalBox(goal.x1, goal.y1, goal.x2, goal.y2, goal.confidence, detected=True, tracked=True)
        alpha = self.config.goal.tracking_smoothing
        return GoalBox(
            alpha * goal.x1 + (1.0 - alpha) * self.tracked_goal.x1,
            alpha * goal.y1 + (1.0 - alpha) * self.tracked_goal.y1,
            alpha * goal.x2 + (1.0 - alpha) * self.tracked_goal.x2,
            alpha * goal.y2 + (1.0 - alpha) * self.tracked_goal.y2,
            max(goal.confidence, self.tracked_goal.confidence * 0.95),
            detected=True,
            tracked=True,
        )

    def _fallback_goal(self, frame: np.ndarray) -> Optional[GoalBox]:
        if not self.config.goal.use_fallback_goal:
            return None
        height, width = frame.shape[:2]
        goal_cfg = self.config.goal
        return GoalBox(
            goal_cfg.fallback_x1_ratio * width,
            goal_cfg.fallback_y1_ratio * height,
            goal_cfg.fallback_x2_ratio * width,
            goal_cfg.fallback_y2_ratio * height,
            confidence=0.12,
            detected=False,
            tracked=False,
        )

    def _detect_from_mask(self, frame: np.ndarray) -> Optional[GoalBox]:
        frame_h, frame_w = frame.shape[:2]
        goal_cfg = self.config.goal
        y_roi1 = int(frame_h * goal_cfg.goal_roi_top_ratio)
        y_roi2 = int(frame_h * goal_cfg.goal_roi_bottom_ratio)
        x_roi1 = int(frame_w * goal_cfg.goal_roi_x_margin_ratio)
        x_roi2 = int(frame_w * (1.0 - goal_cfg.goal_roi_x_margin_ratio))
        roi = frame[y_roi1:y_roi2, x_roi1:x_roi2]
        if roi.size == 0:
            return None

        mask = self._white_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = self._best_goal_contour(contours, frame_w, frame_h, x_roi1, y_roi1)
        if best is None:
            return None

        x, y, w, h = cv2.boundingRect(best)
        x1, x2 = x_roi1 + x, x_roi1 + x + w
        y1, y2 = y_roi1 + y, y_roi1 + y + h

        roi_mask = mask[y : y + h, x : x + w]
        cols = np.where(roi_mask.sum(axis=0) > 0)[0]
        rows = np.where(roi_mask.sum(axis=1) > 0)[0]
        if len(cols) > 0 and len(rows) > 0:
            x1 = x_roi1 + x + int(cols[0])
            x2 = x_roi1 + x + int(cols[-1])
            y1 = y_roi1 + y + int(rows[0])
            y2 = y_roi1 + y + int(rows[-1])

        goal_w = x2 - x1
        goal_h = y2 - y1
        if goal_w <= 0 or goal_h <= 0:
            return None

        expected_h = int(goal_w / goal_cfg.goal_ratio)
        if expected_h < goal_h:
            y1 = max(0, y2 - expected_h)

        shrink_x_px = int((x2 - x1) * goal_cfg.goal_shrink_x)
        shrink_y_px = int((y2 - y1) * goal_cfg.goal_shrink_y)
        x1 = int(clamp(x1 + shrink_x_px, 0, frame_w - 1))
        x2 = int(clamp(x2 - shrink_x_px, 0, frame_w - 1))
        y1 = int(clamp(y1 + shrink_y_px, 0, frame_h - 1))
        y2 = int(clamp(y2 - shrink_y_px, 0, frame_h - 1))
        y2 = self._extend_bottom_to_goal_line(frame, x1, x2, y2)

        if not self._valid_goal_box(x1, y1, x2, y2, frame_w, frame_h):
            return None

        area = (x2 - x1) * (y2 - y1)
        confidence = min(1.0, area / max(1.0, frame_h * frame_w * 0.5))
        return GoalBox(x1, y1, x2, y2, float(confidence), detected=True, tracked=False)

    def _extend_bottom_to_goal_line(self, frame: np.ndarray, x1: int, x2: int, y2: int) -> int:
        goal_cfg = self.config.goal
        frame_h, frame_w = frame.shape[:2]
        search_bottom = int(clamp(y2 + frame_h * goal_cfg.goal_bottom_extend_ratio, y2, frame_h - 1))
        roi = frame[y2:search_bottom, x1:x2]
        if roi.size == 0:
            return y2
        mask = self._white_mask(roi)
        row_hits = np.where(mask.sum(axis=1) > max(20, (x2 - x1) * 255 * 0.08))[0]
        if len(row_hits) == 0:
            return y2
        return int(clamp(y2 + int(row_hits[-1]), 0, frame_h - 1))

    def _white_mask(self, roi: np.ndarray) -> np.ndarray:
        goal_cfg = self.config.goal
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, goal_cfg.white_value_min], dtype=np.uint8),
            np.array([180, goal_cfg.white_saturation_max, 255], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=goal_cfg.goal_erode_iterations)
        mask = cv2.dilate(mask, kernel, iterations=goal_cfg.goal_dilate_iterations)
        blur_size = max(3, goal_cfg.goal_median_blur | 1)
        return cv2.medianBlur(mask, blur_size)

    def _best_goal_contour(
        self,
        contours: list[np.ndarray],
        frame_w: int,
        frame_h: int,
        x_offset: int,
        y_offset: int,
    ) -> Optional[np.ndarray]:
        scored: list[tuple[float, np.ndarray]] = []
        min_area = frame_w * frame_h * self.config.goal.min_contour_area_ratio
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            x1, y1 = x_offset + x, y_offset + y
            x2, y2 = x_offset + x + w, y_offset + y + h
            if not self._valid_goal_box(x1, y1, x2, y2, frame_w, frame_h):
                continue
            cx = (x1 + x2) * 0.5
            center_score = 1.0 - clamp(abs(cx - frame_w * 0.5) / (frame_w * 0.5), 0.0, 1.0)
            ratio = w / max(1.0, h)
            ratio_score = 1.0 - clamp(abs(ratio - self.config.goal.goal_ratio) / self.config.goal.goal_ratio, 0.0, 1.0)
            score = area * (0.55 + 0.30 * center_score + 0.15 * ratio_score)
            scored.append((score, contour))
        if not scored:
            return None
        return max(scored, key=lambda item: item[0])[1]

    def _valid_goal_box(self, x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int) -> bool:
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        goal_cfg = self.config.goal
        if width < frame_w * goal_cfg.min_goal_width_ratio:
            return False
        if width > frame_w * goal_cfg.max_goal_width_ratio:
            return False
        if height < frame_h * goal_cfg.min_goal_height_ratio:
            return False
        return True
