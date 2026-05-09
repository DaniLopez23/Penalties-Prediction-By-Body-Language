from __future__ import annotations

from collections import deque
from typing import Optional

from src.models import BallState, GoalBox, PenaltyAnalysisState, PoseMetrics
from src.utils.geometry import center_distance


class PenaltyAnalyzer:
    def __init__(
        self,
        shot_speed_threshold_px: float = 14.0,
        shot_confirm_frames: int = 2,
    ) -> None:
        self.shot_speed_threshold_px = shot_speed_threshold_px
        self.shot_confirm_frames = shot_confirm_frames
        self.ball_positions: deque[tuple[float, float]] = deque(maxlen=8)
        self.fast_frames = 0
        self.shot_detected = False
        self.ball_zone: Optional[str] = None

    def update(
        self,
        goal: Optional[GoalBox],
        ball: Optional[BallState],
        pose_metrics: dict[str, PoseMetrics],
    ) -> PenaltyAnalysisState:
        if ball is not None and ball.observed:
            self._update_ball_state(ball.center, goal)

        striker_metrics = pose_metrics.get("striker")
        goalkeeper_metrics = pose_metrics.get("goalkeeper")
        goalkeeper_lean = goalkeeper_metrics.body_lean_deg if goalkeeper_metrics else None

        return PenaltyAnalysisState(
            shot_state="shot" if self.shot_detected else "pre-shot",
            ball_zone=self.ball_zone,
            goalkeeper_direction=self._goalkeeper_direction(goalkeeper_lean),
            striker_left_arm_trunk_angle_deg=(
                striker_metrics.left_arm_trunk_angle_deg if striker_metrics else None
            ),
            striker_right_arm_trunk_angle_deg=(
                striker_metrics.right_arm_trunk_angle_deg if striker_metrics else None
            ),
            striker_body_lean_deg=striker_metrics.body_lean_deg if striker_metrics else None,
            goalkeeper_lean_deg=goalkeeper_lean,
        )

    def _update_ball_state(self, center: tuple[float, float], goal: Optional[GoalBox]) -> None:
        if self.ball_positions:
            speed = center_distance(center, self.ball_positions[-1])
            if speed >= self.shot_speed_threshold_px:
                self.fast_frames += 1
            else:
                self.fast_frames = max(0, self.fast_frames - 1)
            if self.fast_frames >= self.shot_confirm_frames:
                self.shot_detected = True

        self.ball_positions.append(center)

        if self.shot_detected and goal is not None and goal.contains(center, margin=max(6.0, goal.width * 0.03)):
            self.ball_zone = self._goal_zone(goal, center)

    @staticmethod
    def _goal_zone(goal: GoalBox, center: tuple[float, float]) -> str:
        x, _ = center
        third = goal.width / 3.0
        if x < goal.x1 + third:
            return "left"
        if x < goal.x1 + 2.0 * third:
            return "center"
        return "right"

    @staticmethod
    def _goalkeeper_direction(lean_deg: Optional[float]) -> str:
        if lean_deg is None:
            return "unknown"
        if lean_deg <= -8.0:
            return "left"
        if lean_deg >= 8.0:
            return "right"
        return "center"

