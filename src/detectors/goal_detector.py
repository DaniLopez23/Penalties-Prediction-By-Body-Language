"""Goal detector and drawing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .inference_mask import make_inference_frame


@dataclass
class GoalDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]


class GoalDetector:
    """Detect goal posts using white HSV segmentation and geometry cleanup."""

    def __init__(self):
        self.lower_white = np.array([0, 0, 200], dtype=np.uint8)
        self.upper_white = np.array([180, 30, 255], dtype=np.uint8)
        self.goal_ratio = 3.0
        self.roi_top = 0.05
        self.roi_bottom = 0.55
        self.shrink_x = 0.02
        self.shrink_y = 0.03

    def detect(self, frame: np.ndarray) -> GoalDetection | None:
        frame_h, frame_w = frame.shape[:2]
        y_roi1 = int(frame_h * self.roi_top)
        y_roi2 = int(frame_h * self.roi_bottom)
        roi = frame[y_roi1:y_roi2, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x1 = x
        y1 = y + y_roi1
        x2 = x + w
        y2 = y + h + y_roi1

        roi_mask = mask[y : y + h, x : x + w]
        cols = np.where(roi_mask.sum(axis=0) > 0)[0]
        rows = np.where(roi_mask.sum(axis=1) > 0)[0]
        if len(cols) > 0 and len(rows) > 0:
            x1 = x + int(cols[0])
            x2 = x + int(cols[-1])
            y1 = y_roi1 + y + int(rows[0])
            y2 = y_roi1 + y + int(rows[-1])

        goal_w = x2 - x1
        goal_h = y2 - y1
        if goal_w <= 0 or goal_h <= 0:
            return None

        expected_h = int(goal_w / self.goal_ratio)
        if expected_h < goal_h:
            y1 = max(0, y2 - expected_h)

        shrink_x_px = int((x2 - x1) * self.shrink_x)
        shrink_y_px = int((y2 - y1) * self.shrink_y)
        x1 = max(0, x1 + shrink_x_px)
        x2 = min(frame_w - 1, x2 - shrink_x_px)
        y1 = max(0, y1 + shrink_y_px)
        y2 = min(frame_h - 1, y2 - shrink_y_px)

        area = (x2 - x1) * (y2 - y1)
        confidence = min(1.0, area / max(1.0, frame_h * frame_w * 0.5))
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return GoalDetection((int(x1), int(y1), int(x2), int(y2)), float(confidence), center)

    def split_goal_into_zones(self, goal: GoalDetection) -> list[dict]:
        x1, y1, x2, y2 = goal.bbox_xyxy
        cell_w = (x2 - x1) / 3
        zones = []
        names = ["izquierda", "centro", "derecha"]
        for col, name in enumerate(names):
            zones.append(
                {
                    "id": name,
                    "col": col,
                    "bbox": (
                        int(x1 + col * cell_w),
                        int(y1),
                        int(x1 + (col + 1) * cell_w),
                        int(y2),
                    ),
                }
            )
        return zones

    def get_ball_zone(
        self,
        goal: GoalDetection | None,
        ball_center: tuple[float, float] | None,
    ) -> str | None:
        if goal is None or ball_center is None:
            return None
        x1, y1, x2, y2 = goal.bbox_xyxy
        bx, by = ball_center
        if not (x1 <= bx <= x2 and y1 <= by <= y2):
            return None
        col = min(2, max(0, int((bx - x1) / max(1, (x2 - x1) / 3))))
        return ["izquierda", "centro", "derecha"][col]

    def annotate_zones(
        self,
        frame: np.ndarray,
        goal: GoalDetection | None,
        ball_zone: str | None = None,
    ) -> np.ndarray:
        if goal is None:
            return frame
        for zone in self.split_goal_into_zones(goal):
            x1, y1, x2, y2 = zone["bbox"]
            color = (255, 0, 0) if ball_zone != zone["id"] else (255, 80, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                str(zone["id"]),
                (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def mask_outside_goal_area(self, frame: np.ndarray, goal: GoalDetection | None, **_) -> np.ndarray:
        return make_inference_frame(frame, goal, tight=True)

    def mask_for_player_detection(self, frame: np.ndarray, goal: GoalDetection | None, **_) -> np.ndarray:
        return make_inference_frame(frame, goal, tight=False)
