from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GoalDetection:
    bbox_xyxy: tuple[int, int, int, int]
    left_post_x: int
    right_post_x: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        # Backward-compatible (x, y, w, h) format.
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def side(self) -> str:
        # Legacy API expected by older visualization code.
        return "center"


class GoalPostDetector:
    """Detect goal posts by vertical line analysis with frame-to-frame stabilization."""

    def __init__(
        self,
        roi_x_start_ratio: float = 0.2,
        roi_x_end_ratio: float = 0.8,
        roi_y_start_ratio: float = 0.05,
        roi_y_end_ratio: float = 0.65,
        min_post_height_ratio: float = 0.18,
        min_goal_width_ratio: float = 0.12,
        max_goal_width_ratio: float = 0.65,
        smooth_alpha: float = 0.25,
        max_missing_frames: int = 20,
    ) -> None:
        self.roi_x_start_ratio = roi_x_start_ratio
        self.roi_x_end_ratio = roi_x_end_ratio
        self.roi_y_start_ratio = roi_y_start_ratio
        self.roi_y_end_ratio = roi_y_end_ratio
        self.min_post_height_ratio = min_post_height_ratio
        self.min_goal_width_ratio = min_goal_width_ratio
        self.max_goal_width_ratio = max_goal_width_ratio
        self.smooth_alpha = smooth_alpha
        self.max_missing_frames = max_missing_frames

        self._smoothed_bbox: tuple[int, int, int, int] | None = None
        self._missing_frames = 0

    def detect(self, frame: np.ndarray) -> GoalDetection | None:
        height, width = frame.shape[:2]
        rx1 = int(width * self.roi_x_start_ratio)
        rx2 = int(width * self.roi_x_end_ratio)
        ry1 = int(height * self.roi_y_start_ratio)
        ry2 = int(height * self.roi_y_end_ratio)
        roi = frame[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return self._fallback_from_history()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, threshold1=60, threshold2=170)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=55,
            minLineLength=int(max(20, (ry2 - ry1) * self.min_post_height_ratio)),
            maxLineGap=12,
        )

        if lines is None:
            return self._fallback_from_history()

        vertical_segments: list[tuple[int, int, int, int, int]] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy <= 0:
                continue

            # Keep near-vertical segments and reject horizontal field lines.
            if dx > max(8, int(0.08 * dy)):
                continue

            seg_h = dy
            if seg_h < int((ry2 - ry1) * self.min_post_height_ratio):
                continue

            x_mean = (x1 + x2) // 2
            vertical_segments.append((x_mean, y1, y2, seg_h, dx))

        if len(vertical_segments) < 2:
            return self._fallback_from_history()

        vertical_segments.sort(key=lambda s: s[0])

        min_goal_w = int(width * self.min_goal_width_ratio)
        max_goal_w = int(width * self.max_goal_width_ratio)

        best_pair = None
        best_score = -1.0
        for i in range(len(vertical_segments)):
            for j in range(i + 1, len(vertical_segments)):
                left = vertical_segments[i]
                right = vertical_segments[j]
                pair_w = right[0] - left[0]
                if pair_w < min_goal_w or pair_w > max_goal_w:
                    continue

                pair_height = min(left[3], right[3])
                center_dist = abs(((left[0] + right[0]) / 2.0) - ((rx2 - rx1) / 2.0))
                center_bonus = 1.0 - min(1.0, center_dist / max(1.0, (rx2 - rx1) * 0.5))
                score = pair_height + 35.0 * center_bonus
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)

        if best_pair is None:
            return self._fallback_from_history()

        left, right = best_pair
        lx = rx1 + left[0]
        rx = rx1 + right[0]

        top_local = min(left[1], left[2], right[1], right[2])
        bottom_local = max(left[1], left[2], right[1], right[2])
        top_y = max(0, ry1 + top_local - int(0.05 * height))
        bottom_y = min(height - 1, ry1 + bottom_local + int(0.08 * height))

        target_bbox = (max(0, lx - 8), top_y, min(width - 1, rx + 8), bottom_y)
        smoothed_bbox = self._smooth_bbox(target_bbox)

        self._missing_frames = 0
        return GoalDetection(
            bbox_xyxy=smoothed_bbox,
            left_post_x=smoothed_bbox[0],
            right_post_x=smoothed_bbox[2],
            confidence=min(0.98, 0.7 + min(0.28, best_score / max(1.0, height * 0.75))),
        )

    def _smooth_bbox(self, target_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self._smoothed_bbox is None:
            self._smoothed_bbox = target_bbox
            return target_bbox

        ax = self.smooth_alpha
        sx1, sy1, sx2, sy2 = self._smoothed_bbox
        tx1, ty1, tx2, ty2 = target_bbox
        smoothed = (
            int((1.0 - ax) * sx1 + ax * tx1),
            int((1.0 - ax) * sy1 + ax * ty1),
            int((1.0 - ax) * sx2 + ax * tx2),
            int((1.0 - ax) * sy2 + ax * ty2),
        )
        self._smoothed_bbox = smoothed
        return smoothed

    def _fallback_from_history(self) -> GoalDetection | None:
        if self._smoothed_bbox is None:
            return None

        self._missing_frames += 1
        if self._missing_frames > self.max_missing_frames:
            self._smoothed_bbox = None
            return None

        x1, y1, x2, y2 = self._smoothed_bbox
        return GoalDetection(
            bbox_xyxy=self._smoothed_bbox,
            left_post_x=x1,
            right_post_x=x2,
            confidence=max(0.35, 0.75 - 0.02 * self._missing_frames),
        )


class GoalDetector(GoalPostDetector):
    """Backward-compatible alias for older pipeline modules."""

