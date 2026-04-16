from dataclasses import dataclass
from math import hypot

import numpy as np
from ultralytics import YOLO

from .ball_detection import BallDetection
from .goal_detection import GoalDetection


KEYPOINT_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 3),
    (3, 1),
    (6, 4),
    (4, 2),
]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class PlayerDetection:
    role: str
    bbox: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    keypoints: list[Keypoint]


@dataclass
class PlayersDetections:
    goalkeeper: PlayerDetection | None
    launcher: PlayerDetection | None


class PlayersDetector:
    def __init__(self, model_name: str = "yolov8n-pose.pt", confidence: float = 0.25) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame, goal_detection: GoalDetection | None, ball_detection: BallDetection | None) -> PlayersDetections:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes if results else None
        keypoints = results[0].keypoints if results else None

        persons: list[PlayerDetection] = []

        if boxes is not None:
            for index, box in enumerate(boxes):
                class_id = int(box.cls.item())
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf.item())
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                person_keypoints = self._extract_keypoints(keypoints, index)

                persons.append(
                    PlayerDetection(
                        role="persona",
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=confidence,
                        center=(center_x, center_y),
                        keypoints=person_keypoints,
                    )
                )

        launcher = self._select_launcher(persons, ball_detection)
        goalkeeper = self._select_goalkeeper(persons, goal_detection, launcher)

        return PlayersDetections(goalkeeper=goalkeeper, launcher=launcher)

    def _extract_keypoints(self, keypoints, person_index: int) -> list[Keypoint]:
        if keypoints is None:
            return []

        if not hasattr(keypoints, "xy"):
            return []

        xy = keypoints.xy[person_index]
        conf = getattr(keypoints, "conf", None)
        conf_values = conf[person_index] if conf is not None else None

        extracted: list[Keypoint] = []
        for keypoint_index, point in enumerate(xy):
            confidence = float(conf_values[keypoint_index].item()) if conf_values is not None else 0.0
            extracted.append(Keypoint(x=float(point[0].item()), y=float(point[1].item()), confidence=confidence))

        return extracted

    def _select_launcher(self, persons: list[PlayerDetection], ball_detection: BallDetection | None) -> PlayerDetection | None:
        if not persons or ball_detection is None:
            return None

        ball_x, ball_y = ball_detection.center
        return min(persons, key=lambda person: hypot(person.center[0] - ball_x, person.center[1] - ball_y))

    def _select_goalkeeper(
        self,
        persons: list[PlayerDetection],
        goal_detection: GoalDetection | None,
        launcher: PlayerDetection | None,
    ) -> PlayerDetection | None:
        if not persons or goal_detection is None:
            return None

        candidates = [person for person in persons if launcher is None or person != launcher]
        if not candidates:
            return None

        if goal_detection.side == "left":
            return min(candidates, key=lambda person: person.center[0])

        return max(candidates, key=lambda person: person.center[0])
