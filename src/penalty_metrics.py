"""Penalty kick metrics calculation and tracking."""

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List
from .pose.pose_estimator import PoseEstimation
from .pose.angles import AngleCalculator


@dataclass
class PenaltyMetrics:
    """Metrics computed for penalty kick analysis."""
    shooter_shoulder_angle: Optional[float] = None
    shooter_body_angle: Optional[float] = None
    ball_trajectory: List[tuple[int, int]] = field(default_factory=list)
    shot_direction: Optional[str] = None
    goalkeeper_movement: Optional[str] = None
    goalkeeper_shoulder_angle: Optional[float] = None
    goalkeeper_body_angle: Optional[float] = None
    goalkeeper_reaction_time_ms: Optional[float] = None


class MetricsCalculator:
    """Calculate penalty kick metrics from detections and poses."""
    
    def __init__(self, history_size: int = 90):
        """Initialize metrics calculator.
        
        Args:
            history_size: Maximum number of frames to keep in trajectory history.
        """
        self.ball_trajectory: deque = deque(maxlen=history_size)
        self.goalkeeper_trajectory: deque = deque(maxlen=history_size)
        self.frame_index = 0
        self.shot_start_frame: Optional[int] = None
        self.goalkeeper_move_start_frame: Optional[int] = None
    
    def update(
        self,
        frame_idx: int,
        ball_center: Optional[tuple[float, float]],
        goalkeeper_center: Optional[tuple[float, float]],
        shooter_pose: Optional[PoseEstimation] = None,
        goalkeeper_pose: Optional[PoseEstimation] = None,
        fps: float = 30.0
    ) -> PenaltyMetrics:
        """Update metrics for current frame.
        
        Args:
            frame_idx: Current frame index.
            ball_center: Ball center coordinates or None.
            goalkeeper_center: Goalkeeper center coordinates or None.
            shooter_pose: Shooter pose estimation or None.
            goalkeeper_pose: Goalkeeper pose estimation or None.
            fps: Frames per second for time calculations.
            
        Returns:
            PenaltyMetrics object with computed metrics.
        """
        self.frame_index = frame_idx
        
        # Update trajectories
        if ball_center:
            self.ball_trajectory.append((int(ball_center[0]), int(ball_center[1])))
        
        if goalkeeper_center:
            self.goalkeeper_trajectory.append((int(goalkeeper_center[0]), int(goalkeeper_center[1])))
        
        # Detect shot start
        self._detect_shot_start()
        
        # Detect goalkeeper movement
        self._detect_goalkeeper_movement()
        
        # Calculate metrics
        metrics = PenaltyMetrics()
        
        if shooter_pose:
            metrics.shooter_shoulder_angle = AngleCalculator.shoulder_angle(shooter_pose)
            metrics.shooter_body_angle = AngleCalculator.torso_angle(shooter_pose)
        
        metrics.ball_trajectory = list(self.ball_trajectory)
        metrics.goalkeeper_movement = self._compute_movement_direction(self.goalkeeper_trajectory)
        
        if goalkeeper_pose:
            metrics.goalkeeper_shoulder_angle = AngleCalculator.shoulder_angle(goalkeeper_pose)
            metrics.goalkeeper_body_angle = AngleCalculator.torso_angle(goalkeeper_pose)

        metrics.shot_direction = self._compute_movement_direction(self.ball_trajectory)
        
        if self.shot_start_frame is not None and self.goalkeeper_move_start_frame is not None:
            reaction_frames = max(0, self.goalkeeper_move_start_frame - self.shot_start_frame)
            metrics.goalkeeper_reaction_time_ms = (reaction_frames / fps) * 1000
        
        return metrics
    
    def _detect_shot_start(self):
        """Detect when shot begins (ball movement threshold exceeded).
        
        Updates shot_start_frame when ball velocity exceeds threshold.
        """
        if self.shot_start_frame is not None or len(self.ball_trajectory) < 4:
            return
        
        p0 = self.ball_trajectory[-4]
        p1 = self.ball_trajectory[-1]
        distance = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        
        if distance > 18.0:
            self.shot_start_frame = self.frame_index
    
    def _detect_goalkeeper_movement(self):
        """Detect when goalkeeper starts moving (movement threshold exceeded).
        
        Updates goalkeeper_move_start_frame when goalkeeper displacement exceeds threshold.
        """
        if self.goalkeeper_move_start_frame is not None or len(self.goalkeeper_trajectory) < 4:
            return
        
        p0 = self.goalkeeper_trajectory[-4]
        p1 = self.goalkeeper_trajectory[-1]
        distance = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        
        if distance > 12.0:
            self.goalkeeper_move_start_frame = self.frame_index
    
    @staticmethod
    def _compute_movement_direction(trajectory: deque) -> Optional[str]:
        """Compute movement direction from trajectory.
        
        Args:
            trajectory: Deque of (x, y) coordinates.
            
        Returns:
            Direction string: "left", "right", "up", "down", "stable", or None.
        """
        if len(trajectory) < 2:
            return None
        
        first_x, first_y = trajectory[0]
        last_x, last_y = trajectory[-1]
        dx, dy = last_x - first_x, last_y - first_y
        
        if abs(dx) < 5 and abs(dy) < 5:
            return "stable"
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

