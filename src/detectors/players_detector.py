"""Players detection using YOLO."""

import numpy as np
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO
from ..models import ModelConfig


@dataclass
class PlayerDetection:
    """Detection result for a player."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None  # Ultralytics ByteTrack ID
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class PlayersDetector:
    """Detect players in soccer/penalty scenes using YOLO."""
    
    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ):
        """Initialize players detector.
        
        Args:
            model_path: Path to YOLO model weights. If None, uses ModelConfig.PLAYERS_MODEL.
            confidence: Confidence threshold. If None, uses ModelConfig.PLAYERS_CONFIDENCE.
        """
        if model_path is None:
            model_path = ModelConfig.get_players_model_path()
        if confidence is None:
            confidence = ModelConfig.PLAYERS_CONFIDENCE
            
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence
        # Fixed-camera penalty setup: keep only plausible field players.
        self.central_x_min = 0.15
        self.central_x_max = 0.85
        self.top_ignore_ratio = 0.14
        self.min_area_ratio = 0.001
    
    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detect all players in frame using ByteTrack.
        
        Args:
            frame: Input frame as BGR numpy array.
            
        Returns:
            List of PlayerDetection objects (empty list if no players detected).
        """
        # Use track() with ByteTrack for native tracking
        results = self.model.track(
            frame,
            imgsz=1280,
            persist=True,
            tracker='bytetrack.yaml',
            conf=self.confidence,
            classes=[0],  # COCO class for person
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return []
        
        detections = []
        boxes = results[0].boxes
        frame_shape = frame.shape
        
        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if not self._size_ok((x1, y1, x2, y2), frame_shape):
                continue

            if not self._position_ok((x1, y1, x2, y2), center, frame_shape):
                continue
            
            # Extract track ID if available (None if tracking not active yet)
            track_id: int | None = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
            
            detections.append(PlayerDetection(
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                center=center,
                track_id=track_id
            ))
        
        return detections

    def _size_ok(self, bbox_xyxy: tuple[int, int, int, int], shape: tuple[int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        det_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = float(shape[0] * shape[1])
        return det_area >= (self.min_area_ratio * frame_area)

    def _position_ok(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        center: tuple[float, float],
        shape: tuple[int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        cx, cy = center
        h, w = shape[:2]

        # Keep central field area and ignore detections high in the stands.
        if not (self.central_x_min * w <= cx <= self.central_x_max * w):
            return False

        if cy < self.top_ignore_ratio * h:
            return False

        if y2 < self.top_ignore_ratio * h:
            return False

        return True
