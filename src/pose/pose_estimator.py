"""Pose estimation using YOLO pose model."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from ultralytics import YOLO
from ..models import ModelConfig


# COCO pose keypoint indices (YOLOv8 pose output has 17 keypoints)
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


@dataclass
class Keypoint:
    """Single keypoint with coordinates and confidence."""
    x: float
    y: float
    confidence: float


@dataclass
class PoseEstimation:
    """Complete pose estimation result."""
    keypoints: Dict[str, Keypoint]
    raw_xy: np.ndarray  # All 17 COCO keypoints coordinates
    raw_conf: np.ndarray  # Confidence for each keypoint


class PoseEstimator:
    """Estimate human pose using YOLO pose model."""
    
    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ):
        """Initialize pose estimator.
        
        Args:
            model_path: Path to YOLO pose model weights. If None, uses ModelConfig.POSE_MODEL.
            confidence: Confidence threshold. If None, uses ModelConfig.POSE_CONFIDENCE.
        """
        if model_path is None:
            model_path = ModelConfig.get_pose_model_path()
        if confidence is None:
            confidence = ModelConfig.POSE_CONFIDENCE
            
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence
    
    def estimate(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> PoseEstimation | None:
        """Estimate pose for person in bounding box.
        
        Args:
            frame: Input frame as BGR numpy array.
            bbox_xyxy: Bounding box coordinates (x1, y1, x2, y2).
            
        Returns:
            PoseEstimation object or None if pose cannot be estimated.
        """
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.35)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_w, x2 + pad_x)
        crop_y2 = min(frame_h, y2 + pad_y)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            return None

        results = self.model.predict(
            crop,
            conf=self.confidence,
            verbose=False
        )
        
        if not results or results[0].keypoints is None:
            return None
        
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()
        keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
        
        if len(keypoints_xy) == 0:
            return None
        
        crop_bbox = np.array([[x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]], dtype=np.float32)
        match_index = self._best_iou_match(tuple(crop_bbox[0].astype(int)), results[0].boxes.xyxy.cpu().numpy())
        if match_index is None:
            return None

        raw_xy = keypoints_xy[match_index].copy()
        raw_xy[:, 0] += crop_x1
        raw_xy[:, 1] += crop_y1
        raw_conf = keypoints_conf[match_index] if keypoints_conf is not None else np.ones(17)
        
        # Extract named keypoints from COCO indices
        keypoints = {}
        for key_name, idx in COCO_KEYPOINTS.items():
            if idx < len(raw_xy):
                x, y = raw_xy[idx]
                conf = float(raw_conf[idx])
                keypoints[key_name] = Keypoint(x=float(x), y=float(y), confidence=conf)
        
        return PoseEstimation(
            keypoints=keypoints,
            raw_xy=raw_xy,
            raw_conf=raw_conf
        )

    @staticmethod
    def _best_iou_match(subject_box: tuple[int, int, int, int], candidate_boxes: np.ndarray) -> int | None:
        """Match pose result to a player bbox using IoU."""
        if candidate_boxes.size == 0:
            return None

        sx1, sy1, sx2, sy2 = subject_box
        subject = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

        best_iou = 0.0
        best_idx: int | None = None
        for i, candidate in enumerate(candidate_boxes):
            iou = PoseEstimator._iou(subject, candidate)
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
