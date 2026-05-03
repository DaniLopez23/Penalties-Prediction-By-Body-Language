"""Goal detection using HSV color segmentation with bbox refinement and 3x3 zones."""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class GoalDetection:
    """Detection result for a goal."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]


class GoalDetector:
    """Detect goal using white HSV segmentation and goal geometry refinement."""

    def __init__(self):
        self.lower_white = np.array([0, 0, 200], dtype=np.uint8)
        self.upper_white = np.array([180, 30, 255], dtype=np.uint8)

        # Portería real: 7.32 / 2.44 = 3
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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        x1 = x
        y1 = y + y_roi1
        x2 = x + w
        y2 = y + h + y_roi1

        roi_mask = mask[y:y + h, x:x + w]

        cols = np.where(roi_mask.sum(axis=0) > 0)[0]
        rows = np.where(roi_mask.sum(axis=1) > 0)[0]

        if len(cols) > 0 and len(rows) > 0:
            x1 = x + cols[0]
            x2 = x + cols[-1]
            y1 = y_roi1 + y + rows[0]
            y2 = y_roi1 + y + rows[-1]

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
        frame_area = frame_h * frame_w

        confidence = min(1.0, area / (frame_area * 0.5))

        center = (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

        return GoalDetection(
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(confidence),
            center=center,
        )

    def split_goal_into_zones(self, goal: GoalDetection) -> list[dict]:
        """Divide la portería en una cuadrícula 3x3.

        Zonas:
            1 | 2 | 3
            4 | 5 | 6
            7 | 8 | 9
        """
        x1, y1, x2, y2 = goal.bbox_xyxy

        cell_w = (x2 - x1) / 3
        cell_h = (y2 - y1) / 3

        zones = []

        for row in range(3):
            for col in range(3):
                zx1 = int(x1 + col * cell_w)
                zy1 = int(y1 + row * cell_h)
                zx2 = int(x1 + (col + 1) * cell_w)
                zy2 = int(y1 + (row + 1) * cell_h)

                zone_id = row * 3 + col + 1

                zones.append(
                    {
                        "id": zone_id,
                        "row": row,
                        "col": col,
                        "bbox": (zx1, zy1, zx2, zy2),
                    }
                )

        return zones

    def get_ball_zone(
        self,
        goal: GoalDetection | None,
        ball_center: tuple[float, float] | None,
    ) -> int | None:
        """Devuelve la zona 1-9 en la que está el balón."""
        if goal is None or ball_center is None:
            return None

        x1, y1, x2, y2 = goal.bbox_xyxy
        bx, by = ball_center

        if not (x1 <= bx <= x2 and y1 <= by <= y2):
            return None

        cell_w = (x2 - x1) / 3
        cell_h = (y2 - y1) / 3

        col = int((bx - x1) / cell_w)
        row = int((by - y1) / cell_h)

        col = min(max(col, 0), 2)
        row = min(max(row, 0), 2)

        return row * 3 + col + 1

    def annotate(self, frame: np.ndarray, detection: GoalDetection | None) -> np.ndarray:
        """Dibuja la portería detectada."""
        annotated = frame.copy()

        if detection is None:
            return annotated

        x1, y1, x2, y2 = detection.bbox_xyxy

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"goal {detection.confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def annotate_zones(
        self,
        frame: np.ndarray,
        goal: GoalDetection | None,
        ball_zone: int | None = None,
    ) -> np.ndarray:
        """Dibuja la cuadrícula 3x3 sobre la portería."""
        annotated = frame.copy()

        if goal is None:
            return annotated

        zones = self.split_goal_into_zones(goal)

        for zone in zones:
            x1, y1, x2, y2 = zone["bbox"]
            zone_id = zone["id"]

            color = (0, 255, 255)

            if ball_zone == zone_id:
                color = (0, 0, 255)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                annotated,
                str(zone_id),
                (x1 + 8, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def mask_outside_goal_area(
        self,
        frame: np.ndarray,
        goal: GoalDetection | None,
        side_padding: int = 24,
        top_padding: int = 18,
        mask_color: int = 0,
    ) -> np.ndarray:
        """Mask (darken to black) everything except the goal corridor and the area below it.
        
        Much faster than GaussianBlur: uses binary masking instead.
        The preserved window keeps the goal mouth, the goalkeeper area, and the
        lower field intact while darkening the top clutter and the sides that can
        confuse the detectors.
        
        Args:
            frame: Input frame as BGR numpy array.
            goal: GoalDetection object (if None, returns original frame).
            side_padding: Horizontal padding around goal (pixels).
            top_padding: Vertical padding above goal (pixels).
            mask_color: Color to apply outside zone (0=black, 255=white).
            
        Returns:
            Frame with masked areas or original frame if goal is None.
        """
        if goal is None:
            return frame.copy()

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, _ = goal.bbox_xyxy

        keep_x1 = max(0, x1 - side_padding)
        keep_x2 = min(frame_w, x2 + side_padding)
        keep_y1 = max(0, y1 - top_padding)

        masked = frame.copy()
        
        # Mask top area (above goal + padding)
        if keep_y1 > 0:
            masked[:keep_y1, :] = mask_color
        
        # Mask left side (before goal - padding)
        if keep_x1 > 0:
            masked[keep_y1:, :keep_x1] = mask_color
        
        # Mask right side (after goal + padding)
        if keep_x2 < frame_w:
            masked[keep_y1:, keep_x2:] = mask_color
        
        return masked