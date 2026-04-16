from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None


@dataclass
class Detection:
    role: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class GoalZones:
    goal_bbox_xyxy: tuple[int, int, int, int]
    zones: dict[str, tuple[int, int, int, int]]


@dataclass
class FrameDetections:
    shooter: Detection | None
    goalkeeper: Detection | None
    ball: Detection | None
    goal: Detection | None
    goal_zones: GoalZones | None
    persons: list[Detection]


class PenaltyDetector:
    """YOLO inference for player roles and ball, with robust small-ball recovery."""

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        person_model_path: str | None = None,
        ball_model_path: str | None = None,
        goal_model_path: str | None = None,
        confidence: float = 0.2,
        person_confidence: float = 0.25,
        ball_confidence: float = 0.12,
        iou: float = 0.5,
        imgsz: int = 1280,
        ball_area_ratio_max: float = 0.015,
        ball_search_window_ratio: float = 0.28,
        use_goal_roi: bool = True,
        use_goal_hough_validation: bool = True,
    ) -> None:
        # goal_model_path/use_goal_* are kept for backward compatibility.
        # Goal localization is handled by the dedicated GoalPostDetector in goal_detection.py.
        _ = goal_model_path
        _ = use_goal_roi
        _ = use_goal_hough_validation

        self.person_model = YOLO(person_model_path or model_path)
        self.ball_model = YOLO(ball_model_path or model_path)
        self.confidence = confidence
        self.person_confidence = person_confidence
        self.ball_confidence = ball_confidence
        self.iou = iou
        self.imgsz = imgsz
        self.ball_area_ratio_max = ball_area_ratio_max
        self.ball_search_window_ratio = ball_search_window_ratio

        self.device = "0" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.use_half = self.device != "cpu"

        self._person_names = {int(k): str(v).lower() for k, v in self.person_model.names.items()}
        self._ball_names = {int(k): str(v).lower() for k, v in self.ball_model.names.items()}
        self._ball_class_ids = [cls_id for cls_id, name in self._ball_names.items() if self._is_ball_label(name)]

    def detect(
        self,
        frame,
        goal_bbox_xyxy: tuple[int, int, int, int] | None = None,
        last_ball_center: tuple[float, float] | None = None,
        last_goalkeeper_center: tuple[float, float] | None = None,
    ) -> FrameDetections:
        person_results = self.person_model.predict(
            frame,
            conf=self.person_confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
            classes=[0],
        )

        ball_classes = self._ball_class_ids if self._ball_class_ids else None
        ball_results = self.ball_model.predict(
            frame,
            conf=self.ball_confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
            classes=ball_classes,
        )

        person_boxes = person_results[0].boxes if person_results else None
        ball_boxes = ball_results[0].boxes if ball_results else None
        persons: list[Detection] = []
        explicit_shooters: list[Detection] = []
        explicit_goalkeepers: list[Detection] = []
        ball_candidates: list[Detection] = []

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        if person_boxes is not None:
            for box in person_boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = self._person_names.get(cls_id, "")
                det = Detection(role=label, bbox_xyxy=(x1, y1, x2, y2), confidence=conf)

                if self._is_person_label(label):
                    persons.append(det)
                elif self._is_shooter_label(label):
                    explicit_shooters.append(Detection("shooter", det.bbox_xyxy, conf))
                elif self._is_goalkeeper_label(label):
                    explicit_goalkeepers.append(Detection("goalkeeper", det.bbox_xyxy, conf))

        if ball_boxes is not None:
            for box in ball_boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = self._ball_names.get(cls_id, "")
                det = Detection(role=label, bbox_xyxy=(x1, y1, x2, y2), confidence=conf)
                if self._is_ball_label(label) and self._is_ball_candidate(det, frame_area):
                    ball_candidates.append(Detection("ball", det.bbox_xyxy, conf))

        if not ball_candidates and last_ball_center is not None:
            local_ball = self._detect_ball_in_local_window(frame, last_ball_center, frame_area)
            if local_ball is not None:
                ball_candidates.append(local_ball)

        goal = Detection(role="goal", bbox_xyxy=goal_bbox_xyxy, confidence=1.0) if goal_bbox_xyxy is not None else None

        ball = self._select_ball(ball_candidates, last_ball_center)

        shooter = self._choose_highest_conf(explicit_shooters)
        goalkeeper = self._choose_highest_conf(explicit_goalkeepers)

        if shooter is None:
            shooter = self._select_shooter(persons, ball)

        if goalkeeper is None:
            goalkeeper = self._select_goalkeeper(
                persons,
                goal,
                shooter,
                frame_width,
                frame_height,
                last_goalkeeper_center,
            )

        goal_zones = self._build_goal_zones(goal)
        return FrameDetections(
            shooter=shooter,
            goalkeeper=goalkeeper,
            ball=ball,
            goal=goal,
            goal_zones=goal_zones,
            persons=persons,
        )

    @staticmethod
    def _choose_highest_conf(detections: Iterable[Detection]) -> Detection | None:
        detections = list(detections)
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    @staticmethod
    def _is_person_label(label: str) -> bool:
        return label in {"person", "player", "athlete"}

    @staticmethod
    def _is_shooter_label(label: str) -> bool:
        return label in {"shooter", "kicker", "launcher"}

    @staticmethod
    def _is_goalkeeper_label(label: str) -> bool:
        return label in {"goalkeeper", "keeper", "portero"}

    @staticmethod
    def _is_ball_label(label: str) -> bool:
        return label in {"sports ball", "ball", "football", "soccer ball"}

    def _is_ball_candidate(self, detection: Detection, frame_area: float) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = w / float(h)
        area_ratio = detection.area / max(1.0, frame_area)
        return 0.4 <= aspect_ratio <= 2.5 and area_ratio <= self.ball_area_ratio_max

    def _detect_ball_in_local_window(
        self,
        frame,
        last_ball_center: tuple[float, float],
        frame_area: float,
    ) -> Detection | None:
        frame_h, frame_w = frame.shape[:2]
        cx, cy = int(last_ball_center[0]), int(last_ball_center[1])
        half_size = int(min(frame_w, frame_h) * self.ball_search_window_ratio * 0.5)

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(frame_w, cx + half_size)
        y2 = min(frame_h, cy + half_size)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame[y1:y2, x1:x2]
        ball_classes = self._ball_class_ids if self._ball_class_ids else None
        local_results = self.ball_model.predict(
            crop,
            conf=max(0.05, self.ball_confidence * 0.75),
            iou=self.iou,
            imgsz=max(640, self.imgsz // 2),
            device=self.device,
            half=self.use_half,
            verbose=False,
            classes=ball_classes,
        )

        boxes = local_results[0].boxes if local_results else None
        if boxes is None:
            return None

        candidates: list[Detection] = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            lx1, ly1, lx2, ly2 = [int(v) for v in box.xyxy[0].tolist()]
            label = self._ball_names.get(cls_id, "")
            if not self._is_ball_label(label):
                continue

            det = Detection(
                role="ball",
                bbox_xyxy=(x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2),
                confidence=conf,
            )
            if self._is_ball_candidate(det, frame_area):
                candidates.append(det)

        return self._choose_highest_conf(candidates)

    def _select_ball(self, candidates: list[Detection], last_ball_center: tuple[float, float] | None) -> Detection | None:
        if not candidates:
            return None

        if last_ball_center is None:
            return max(candidates, key=lambda d: d.confidence)

        lx, ly = last_ball_center

        def score(det: Detection) -> float:
            dx = det.center[0] - lx
            dy = det.center[1] - ly
            distance = (dx * dx + dy * dy) ** 0.5
            return det.confidence - 0.002 * distance

        return max(candidates, key=score)

    @staticmethod
    def _select_shooter(persons: list[Detection], ball: Detection | None) -> Detection | None:
        if not persons:
            return None

        if ball is not None:
            bx, by = ball.center
            return min(persons, key=lambda p: (p.center[0] - bx) ** 2 + (p.center[1] - by) ** 2)

        # Rear camera assumption: shooter is often closest to camera (largest y center).
        return max(persons, key=lambda p: p.center[1])

    @staticmethod
    def _select_goalkeeper(
        persons: list[Detection],
        goal: Detection | None,
        shooter: Detection | None,
        frame_width: int,
        frame_height: int,
        last_goalkeeper_center: tuple[float, float] | None,
    ) -> Detection | None:
        if not persons:
            return None

        candidates = [p for p in persons if shooter is None or p.bbox_xyxy != shooter.bbox_xyxy]
        if not candidates:
            return None

        # Remove spectators by enforcing a goalkeeper prior: around center-top and near goal ROI.
        constrained: list[Detection] = []
        if goal is not None:
            gx1, gy1, gx2, gy2 = goal.bbox_xyxy
            goal_w = max(1, gx2 - gx1)
            expanded_x1 = gx1 - int(0.2 * goal_w)
            expanded_x2 = gx2 + int(0.2 * goal_w)
            max_y = gy2 + int(0.15 * frame_height)

            for person in candidates:
                cx, cy = person.center
                if expanded_x1 <= cx <= expanded_x2 and cy <= max_y:
                    constrained.append(person)
        else:
            min_x = int(frame_width * 0.2)
            max_x = int(frame_width * 0.8)
            max_y = int(frame_height * 0.62)
            for person in candidates:
                cx, cy = person.center
                if min_x <= cx <= max_x and cy <= max_y:
                    constrained.append(person)

        if constrained:
            candidates = constrained

        if last_goalkeeper_center is not None:
            gx_prev, gy_prev = last_goalkeeper_center
            return min(candidates, key=lambda p: (p.center[0] - gx_prev) ** 2 + (p.center[1] - gy_prev) ** 2)

        if goal is not None:
            gx, gy = goal.center
            return min(candidates, key=lambda p: (p.center[0] - gx) ** 2 + (p.center[1] - gy) ** 2)

        # Fallback: goalkeeper typically appears farther from camera (smaller y center).
        return min(candidates, key=lambda p: p.center[1])

    @staticmethod
    def _build_goal_zones(goal: Detection | None) -> GoalZones | None:
        if goal is None:
            return None

        x1, y1, x2, y2 = goal.bbox_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        third = width // 3
        half = height // 2

        zones = {
            "left_top": (x1, y1, x1 + third, y1 + half),
            "center_top": (x1 + third, y1, x1 + 2 * third, y1 + half),
            "right_top": (x1 + 2 * third, y1, x2, y1 + half),
            "left_bottom": (x1, y1 + half, x1 + third, y2),
            "center_bottom": (x1 + third, y1 + half, x1 + 2 * third, y2),
            "right_bottom": (x1 + 2 * third, y1 + half, x2, y2),
        }
        return GoalZones(goal_bbox_xyxy=goal.bbox_xyxy, zones=zones)
