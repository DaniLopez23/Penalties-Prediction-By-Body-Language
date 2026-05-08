"""YOLO11 pose + BoT-SORT player tracker."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ..models import ModelConfig
from ..pose.pose_estimator import COCO_KEYPOINTS, Keypoint, PoseEstimation
from .players_detector import PlayerDetection


class PosePlayersDetector:
    """One established model call for player boxes, track IDs and pose."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        imgsz: int | None = None,
        tracker: str | None = None,
    ):
        self.model_path = model_path or ModelConfig.get_players_model_path()
        self.confidence = confidence if confidence is not None else ModelConfig.PLAYERS_CONFIDENCE
        self.imgsz = imgsz or ModelConfig.PLAYERS_IMGSZ
        self.tracker = tracker or ModelConfig.PLAYERS_TRACKER
        self.model = YOLO(self.model_path)

    def track(self, frame: np.ndarray) -> list[PlayerDetection]:
        kwargs = {
            "imgsz": self.imgsz,
            "persist": True,
            "tracker": self.tracker,
            "conf": self.confidence,
            "classes": [ModelConfig.PERSON_CLASS_ID],
            "verbose": False,
        }
        if ModelConfig.PLAYERS_DEVICE is not None:
            kwargs["device"] = ModelConfig.PLAYERS_DEVICE
        if ModelConfig.PLAYERS_HALF:
            kwargs["half"] = True

        results = self.model.track(frame, **kwargs)
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

        players: list[PlayerDetection] = []
        frame_area = float(frame.shape[0] * frame.shape[1])
        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy().astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if area < ModelConfig.PLAYERS_MIN_AREA_RATIO * frame_area:
                continue

            track_id = int(boxes.id[index].item()) if boxes.id is not None else None
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            players.append(
                PlayerDetection(
                    bbox_xyxy=bbox,
                    confidence=float(boxes.conf[index].item()),
                    center=center,
                    track_id=track_id,
                    pose=self._build_pose(keypoints_xy, keypoints_conf, index),
                )
            )
        return players

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        return self.track(frame)

    @staticmethod
    def _build_pose(
        keypoints_xy: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
        index: int,
    ) -> PoseEstimation | None:
        if keypoints_xy is None or index >= len(keypoints_xy):
            return None

        raw_xy = keypoints_xy[index].copy()
        raw_conf = (
            keypoints_conf[index].copy()
            if keypoints_conf is not None and index < len(keypoints_conf)
            else np.ones(len(raw_xy), dtype=np.float32)
        )

        named = {}
        for name, keypoint_index in COCO_KEYPOINTS.items():
            if keypoint_index >= len(raw_xy):
                continue
            x, y = raw_xy[keypoint_index]
            conf = float(raw_conf[keypoint_index])
            named[name] = Keypoint(float(x), float(y), conf)

        return PoseEstimation(keypoints=named, raw_xy=raw_xy, raw_conf=raw_conf)
