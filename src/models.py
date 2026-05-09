from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


COCO_POSE_EDGES = (
    (5, 7),
    (6, 8),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
)


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    track_id: Optional[int] = None
    source: str = "yolo"

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


@dataclass
class PoseDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray


@dataclass
class PoseMetrics:
    body_lean_deg: Optional[float] = None
    shoulder_angle_deg: Optional[float] = None
    hip_angle_deg: Optional[float] = None
    left_arm_trunk_angle_deg: Optional[float] = None
    right_arm_trunk_angle_deg: Optional[float] = None


@dataclass
class GoalBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    detected: bool = True
    tracked: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def zone_bounds(self) -> dict[str, tuple[int, int, int, int]]:
        third = self.width / 3.0
        y1, y2 = int(round(self.y1)), int(round(self.y2))
        return {
            "left": (int(round(self.x1)), y1, int(round(self.x1 + third)), y2),
            "center": (
                int(round(self.x1 + third)),
                y1,
                int(round(self.x1 + 2.0 * third)),
                y2,
            ),
            "right": (int(round(self.x1 + 2.0 * third)), y1, int(round(self.x2)), y2),
        }

    def contains(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        x, y = point
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )


@dataclass
class BallState:
    center: tuple[float, float]
    radius: int
    confidence: float
    source: str
    observed: bool
    track_id: Optional[int] = None


@dataclass
class PenaltyAnalysisState:
    shot_state: str = "pre-shot"
    ball_zone: Optional[str] = None
    goalkeeper_direction: str = "unknown"
    striker_left_arm_trunk_angle_deg: Optional[float] = None
    striker_right_arm_trunk_angle_deg: Optional[float] = None
    striker_body_lean_deg: Optional[float] = None
    goalkeeper_lean_deg: Optional[float] = None


@dataclass
class FrameAnalysisRecord:
    frame_index: int
    time_sec: float
    shot_state: str
    ball_zone: Optional[str] = None
    goalkeeper_direction: str = "unknown"
    striker_shoulder_angle_deg: Optional[float] = None
    striker_body_lean_deg: Optional[float] = None
    goalkeeper_lean_deg: Optional[float] = None
