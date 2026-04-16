from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from .tracking import TrackedFrame, TrackedObject


COCO_KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
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


@dataclass
class PoseResult:
    role: str
    keypoints: dict[str, tuple[float, float, float]]
    raw_xy: np.ndarray
    raw_conf: np.ndarray


class PoseEstimator:
    def __init__(self, model_path: str = "yolov8s-pose.pt", confidence: float = 0.25, iou: float = 0.5, imgsz: int = 960) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def estimate(self, frame: np.ndarray, tracked: TrackedFrame) -> dict[str, PoseResult]:
        tracked_subjects = {
            "shooter": tracked.shooter,
            "goalkeeper": tracked.goalkeeper,
        }

        if all(subject is None for subject in tracked_subjects.values()):
            return {}

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            classes=[0],
        )
        if not results:
            return {}

        output = results[0]
        if output.boxes is None or output.keypoints is None:
            return {}

        boxes_xyxy = output.boxes.xyxy.cpu().numpy()
        keypoints_xy = output.keypoints.xy.cpu().numpy()
        keypoints_conf = output.keypoints.conf.cpu().numpy() if output.keypoints.conf is not None else None

        role_to_pose: dict[str, PoseResult] = {}
        for role, subject in tracked_subjects.items():
            if subject is None:
                continue

            match_index = self._best_iou_match(subject, boxes_xyxy)
            if match_index is None:
                continue

            raw_xy = keypoints_xy[match_index]
            raw_conf = keypoints_conf[match_index] if keypoints_conf is not None else np.ones(raw_xy.shape[0], dtype=np.float32)

            selected = {}
            for key_name, idx in COCO_KEYPOINTS.items():
                x, y = raw_xy[idx]
                c = float(raw_conf[idx])
                selected[key_name] = (float(x), float(y), c)

            role_to_pose[role] = PoseResult(
                role=role,
                keypoints=selected,
                raw_xy=raw_xy,
                raw_conf=raw_conf,
            )

        return role_to_pose

    @staticmethod
    def _best_iou_match(subject: TrackedObject, candidate_boxes: np.ndarray) -> int | None:
        if candidate_boxes.size == 0:
            return None

        sx1, sy1, sx2, sy2 = subject.bbox_xyxy
        subject_box = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

        best_iou = 0.0
        best_idx: int | None = None
        for i, cbox in enumerate(candidate_boxes):
            iou = PoseEstimator._iou(subject_box, cbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx if best_iou > 0.1 else None

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0


def draw_pose(frame: np.ndarray, pose_result: PoseResult, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    for start_idx, end_idx in SKELETON_EDGES:
        if start_idx >= len(pose_result.raw_xy) or end_idx >= len(pose_result.raw_xy):
            continue

        c1 = float(pose_result.raw_conf[start_idx]) if pose_result.raw_conf is not None else 1.0
        c2 = float(pose_result.raw_conf[end_idx]) if pose_result.raw_conf is not None else 1.0
        if c1 < 0.2 or c2 < 0.2:
            continue

        x1, y1 = pose_result.raw_xy[start_idx]
        x2, y2 = pose_result.raw_xy[end_idx]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    for key_name in COCO_KEYPOINTS:
        x, y, conf = pose_result.keypoints[key_name]
        if conf < 0.2:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 215, 255), -1)
