from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from src.config import PipelineConfig
from src.models import Detection, PoseDetection


class YOLODetector:
    def __init__(self, config: PipelineConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.models.detector_model)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model_cfg = self.config.models
        kwargs = {
            "persist": True,
            "tracker": model_cfg.tracker,
            "conf": model_cfg.detector_confidence,
            "iou": model_cfg.iou_threshold,
            "imgsz": model_cfg.image_size,
            "classes": [model_cfg.coco_person_class_id, model_cfg.coco_ball_class_id],
            "verbose": False,
        }
        if model_cfg.device:
            kwargs["device"] = model_cfg.device

        results = self.model.track(frame, **kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        if boxes.id is not None:
            ids: Iterable[Optional[int]] = boxes.id.detach().cpu().numpy().astype(int).tolist()
        else:
            ids = [None] * len(xyxy)

        detections: list[Detection] = []
        for box, confidence, class_id, track_id in zip(xyxy, confidences, classes, ids):
            if class_id == model_cfg.coco_person_class_id and confidence < model_cfg.person_confidence:
                continue
            if class_id == model_cfg.coco_ball_class_id and confidence < model_cfg.ball_confidence:
                continue
            detections.append(
                Detection(
                    xyxy=tuple(float(v) for v in box),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    track_id=track_id,
                )
            )
        return detections


class YOLOPoseEstimator:
    def __init__(self, config: PipelineConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.models.pose_model)
        self.last_poses: list[PoseDetection] = []

    def detect(self, frame: np.ndarray, frame_index: int) -> list[PoseDetection]:
        stride = max(1, self.config.models.pose_stride)
        if frame_index % stride != 0:
            return self.last_poses

        kwargs = {
            "conf": self.config.models.pose_confidence,
            "imgsz": self.config.models.image_size,
            "verbose": False,
        }
        if self.config.models.device:
            kwargs["device"] = self.config.models.device

        results = self.model.predict(frame, **kwargs)
        if not results:
            self.last_poses = []
            return []

        boxes = results[0].boxes
        keypoints = results[0].keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            self.last_poses = []
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        keypoint_data = keypoints.data.detach().cpu().numpy()
        poses = [
            PoseDetection(
                xyxy=tuple(float(v) for v in box),
                confidence=float(confidence),
                keypoints=points.astype(np.float32, copy=False),
            )
            for box, confidence, points in zip(xyxy, confidences, keypoint_data)
        ]
        self.last_poses = poses
        return poses

