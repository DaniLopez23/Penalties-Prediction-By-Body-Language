"""Players detection using YOLO."""

import numpy as np
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO


@dataclass
class PlayerDetection:
    """Detection result for a player."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class PlayersDetector:
    """Detect players in soccer/penalty scenes using YOLO."""
    
    def __init__(self, model_path: str = "yolov8s.pt", confidence: float = 0.25):
        """Initialize players detector.
        
        Args:
            model_path: Path to YOLO model weights.
            confidence: Confidence threshold for detections (default 0.25 - permissive).
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detect all players in frame.
        
        Args:
            frame: Input frame as BGR numpy array.
            
        Returns:
            List of PlayerDetection objects (empty list if no players detected).
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[0],  # COCO class for person
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return []
        
        detections = []
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            detections.append(PlayerDetection(
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                center=center
            ))
        
        return detections
