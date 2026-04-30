"""Goal detection using HSV color segmentation."""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class GoalDetection:
    """Detection result for a goal."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]


class GoalDetector:
    """Detect goal posts/net in soccer scenes using HSV color segmentation.
    
    Simple approach: detects white color regions (typical soccer goal appearance).
    """
    
    def __init__(self):
        """Initialize goal detector with HSV color ranges for white."""
        # HSV range for white color (goalkeeper uniforms may also match)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
    
    def detect(self, frame: np.ndarray) -> GoalDetection | None:
        """Detect goal using white color segmentation.
        
        Args:
            frame: Input frame as BGR numpy array.
            
        Returns:
            GoalDetection object or None if no goal detected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Take largest contour (assuming it's the goal)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Confidence based on area ratio
        area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        confidence = min(1.0, area / (frame_area * 0.5))
        
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        return GoalDetection(
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=confidence,
            center=center
        )
