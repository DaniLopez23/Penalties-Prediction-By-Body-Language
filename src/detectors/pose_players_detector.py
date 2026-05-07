"""YOLO pose detector for players.

This module is only responsible for expensive model inference. Role
assignment and role ghosts live elsewhere.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ..models import ModelConfig
from ..pose.pose_estimator import COCO_KEYPOINTS, Keypoint, PoseEstimation
from .players_detector import PlayerDetection


class PosePlayersDetector:
    """Detect people, keypoints and tracker IDs in one YOLO pose call."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        imgsz: int | None = None,
        tracker: str | None = None,
    ):
        self.model_path = model_path or ModelConfig.get_players_model_path()
        self.confidence = (
            confidence if confidence is not None else ModelConfig.PLAYERS_CONFIDENCE
        )
        self.imgsz = imgsz or ModelConfig.PLAYERS_IMGSZ
        self.tracker = tracker or ModelConfig.PLAYERS_TRACKER
        self.model = YOLO(self.model_path)

        self.central_x_min = ModelConfig.PLAYERS_CENTRAL_X_MIN
        self.central_x_max = ModelConfig.PLAYERS_CENTRAL_X_MAX
        self.top_ignore_ratio = ModelConfig.PLAYERS_TOP_IGNORE_RATIO
        self.min_area_ratio = ModelConfig.PLAYERS_MIN_AREA_RATIO

    def track(self, frame: np.ndarray) -> list[PlayerDetection]:
        track_kwargs = {
            "imgsz": self.imgsz,
            "persist": True,
            "tracker": self.tracker,
            "conf": self.confidence,
            "classes": [ModelConfig.PERSON_CLASS_ID],
            "verbose": False,
        }
        if ModelConfig.PLAYERS_DEVICE is not None:
            track_kwargs["device"] = ModelConfig.PLAYERS_DEVICE
        if ModelConfig.PLAYERS_HALF:
            track_kwargs["half"] = True

        results = self.model.track(
            frame,
            **track_kwargs,
        )

        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        keypoints = results[0].keypoints
        keypoints_xy = keypoints.xy.cpu().numpy() if keypoints is not None else None
        keypoints_conf = (
            keypoints.conf.cpu().numpy()
            if keypoints is not None and keypoints.conf is not None
            else None
        )

        detections: list[PlayerDetection] = []
        shape = frame.shape
        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            bbox = (int(x1), int(y1), int(x2), int(y2))
            if not self._size_ok(bbox, shape):
                continue
            if not self._position_ok(bbox, center, shape):
                continue

            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            detections.append(
                PlayerDetection(
                    bbox_xyxy=bbox,
                    confidence=conf,
                    center=center,
                    track_id=track_id,
                    pose=self._build_pose(keypoints_xy, keypoints_conf, i),
                )
            )

        return detections

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        """Backward-compatible alias for the tracked pose pass."""
        return self.track(frame)

    def _size_ok(self, bbox: tuple[int, int, int, int], shape: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area >= self.min_area_ratio * float(shape[0] * shape[1])

    def _position_ok(self, bbox: tuple[int, int, int, int], center: tuple, shape: tuple) -> bool:
        _, _, _, y2 = bbox
        cx, cy = center
        h, w = shape[:2]
        if not (self.central_x_min * w <= cx <= self.central_x_max * w):
            return False
        if cy < self.top_ignore_ratio * h:
            return False
        if y2 < self.top_ignore_ratio * h:
            return False
        return True

    @staticmethod
    def _build_pose(
        keypoints_xy: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
        index: int,
    ) -> PoseEstimation | None:
        if keypoints_xy is None or index >= len(keypoints_xy):
            return None

        raw_xy = keypoints_xy[index].copy()
        if keypoints_conf is not None and index < len(keypoints_conf):
            raw_conf = keypoints_conf[index].copy()
        else:
            raw_conf = np.ones(len(raw_xy), dtype=np.float32)

        keypoints: dict[str, Keypoint] = {}
        for key_name, key_idx in COCO_KEYPOINTS.items():
            if key_idx >= len(raw_xy):
                continue
            x, y = raw_xy[key_idx]
            conf = float(raw_conf[key_idx]) if key_idx < len(raw_conf) else 1.0
            keypoints[key_name] = Keypoint(float(x), float(y), conf)

        return PoseEstimation(keypoints=keypoints, raw_xy=raw_xy, raw_conf=raw_conf)
