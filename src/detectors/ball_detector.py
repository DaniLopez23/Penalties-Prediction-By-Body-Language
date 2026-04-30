"""Ball detection using YOLO."""

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class BallDetection:
    """Detection result for a ball."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class BallDetector:
    """Detect ball in soccer/penalty scenes using YOLO."""
    
    def __init__(self, model_path: str = "yolov8s.pt", confidence: float = 0.2):
        """Initialize ball detector.
        
        Args:
            model_path: Path to YOLO model weights.
            confidence: Confidence threshold for detections (default 0.2 - permissive).
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect(self, frame: np.ndarray) -> BallDetection | None:
        """Detect ball in frame.
        
        Args:
            frame: Input frame as BGR numpy array.
            
        Returns:
            BallDetection object or None if no ball detected.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[32],  # COCO class for sports ball
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return None
        
        # Get detection with highest confidence
        boxes = results[0].boxes
        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
        
        coords = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        conf = float(boxes.conf[best_idx])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        return BallDetection(
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf,
            center=center
        )
