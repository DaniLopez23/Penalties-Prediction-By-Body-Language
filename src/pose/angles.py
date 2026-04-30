"""Calculate body angles from pose keypoints."""

import numpy as np
from math import atan2, degrees
from typing import Optional
from .pose_estimator import PoseEstimation


class AngleCalculator:
    """Calculate body angles from pose keypoints."""
    
    @staticmethod
    def shoulder_angle(pose: PoseEstimation) -> Optional[float]:
        """Calculate angle between shoulders (rotation in horizontal plane).
        
        Args:
            pose: PoseEstimation object.
            
        Returns:
            Angle in degrees or None if keypoints not available/confident.
        """
        left = pose.keypoints.get("left_shoulder")
        right = pose.keypoints.get("right_shoulder")
        
        if left is None or right is None:
            return None
        
        if left.confidence < 0.2 or right.confidence < 0.2:
            return None
        
        angle = degrees(atan2(right.y - left.y, right.x - left.x))
        return angle
    
    @staticmethod
    def torso_angle(pose: PoseEstimation) -> Optional[float]:
        """Calculate angle from shoulder center to hip center (torso inclination).
        
        Args:
            pose: PoseEstimation object.
            
        Returns:
            Angle in degrees or None if keypoints not available/confident.
        """
        ls = pose.keypoints.get("left_shoulder")
        rs = pose.keypoints.get("right_shoulder")
        lh = pose.keypoints.get("left_hip")
        rh = pose.keypoints.get("right_hip")
        
        if any(p is None for p in [ls, rs, lh, rh]):
            return None
        
        if any(p.confidence < 0.2 for p in [ls, rs, lh, rh]):
            return None
        
        shoulder_center = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
        hip_center = ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
        
        angle = degrees(atan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0]))
        return angle
    
    @staticmethod
    def center_of_mass(pose: PoseEstimation) -> Optional[tuple[int, int]]:
        """Calculate center of mass from main joints (shoulders + hips).
        
        Args:
            pose: PoseEstimation object.
            
        Returns:
            (x, y) coordinates or None if keypoints not available/confident.
        """
        keys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        pts = [pose.keypoints.get(k) for k in keys]
        
        if any(p is None for p in pts):
            return None
        
        if any(p.confidence < 0.2 for p in pts):
            return None
        
        cx = int(sum(p.x for p in pts) / 4.0)
        cy = int(sum(p.y for p in pts) / 4.0)
        return (cx, cy)
