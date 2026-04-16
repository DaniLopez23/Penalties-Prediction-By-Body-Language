from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees

from .pose import PoseResult
from .tracking import TrackedFrame


@dataclass
class AnalyticsSnapshot:
    shoulder_angle_deg: float | None
    ball_trajectory: list[tuple[int, int]]
    shot_direction_zone: str | None
    goalkeeper_movement: str | None
    goalkeeper_body_angle_deg: float | None
    goalkeeper_dive_direction: str | None
    goalkeeper_reaction_time_frames: int | None
    goalkeeper_center_of_mass_path: list[tuple[int, int]]


class PenaltyAnalytics:
    def __init__(self, history_size: int = 90) -> None:
        self.ball_path: deque[tuple[int, int]] = deque(maxlen=history_size)
        self.goalkeeper_path: deque[tuple[int, int]] = deque(maxlen=history_size)
        self.goalkeeper_com_path: deque[tuple[int, int]] = deque(maxlen=history_size)
        self.frame_index = 0
        self.shot_start_frame: int | None = None
        self.goalkeeper_move_start_frame: int | None = None

    def update(self, tracked: TrackedFrame, poses: dict[str, PoseResult]) -> AnalyticsSnapshot:
        self.frame_index += 1

        if tracked.ball is not None:
            self.ball_path.append((int(tracked.ball.center[0]), int(tracked.ball.center[1])))

        if tracked.goalkeeper is not None:
            self.goalkeeper_path.append((int(tracked.goalkeeper.center[0]), int(tracked.goalkeeper.center[1])))

        goalkeeper_pose = poses.get("goalkeeper")
        shooter_pose = poses.get("shooter")

        goalkeeper_com = self._compute_center_of_mass(goalkeeper_pose)
        if goalkeeper_com is not None:
            self.goalkeeper_com_path.append(goalkeeper_com)

        self._update_shot_start()
        self._update_goalkeeper_reaction()

        shoulder_angle = self._compute_shoulder_angle(shooter_pose)
        shot_zone = self._compute_shot_zone(tracked)
        keeper_movement = self._compute_goalkeeper_movement(self.goalkeeper_path)
        goalkeeper_dive = self._compute_goalkeeper_movement(self.goalkeeper_com_path)
        body_angle = self._compute_torso_angle(goalkeeper_pose)
        reaction_frames = None
        if self.shot_start_frame is not None and self.goalkeeper_move_start_frame is not None:
            reaction_frames = max(0, self.goalkeeper_move_start_frame - self.shot_start_frame)

        return AnalyticsSnapshot(
            shoulder_angle_deg=shoulder_angle,
            ball_trajectory=tracked.ball_trajectory_smooth if tracked.ball_trajectory_smooth else list(self.ball_path),
            shot_direction_zone=shot_zone,
            goalkeeper_movement=keeper_movement,
            goalkeeper_body_angle_deg=body_angle,
            goalkeeper_dive_direction=goalkeeper_dive,
            goalkeeper_reaction_time_frames=reaction_frames,
            goalkeeper_center_of_mass_path=list(self.goalkeeper_com_path),
        )

    @staticmethod
    def _compute_shoulder_angle(pose: PoseResult | None) -> float | None:
        if pose is None:
            return None

        left = pose.keypoints.get("left_shoulder")
        right = pose.keypoints.get("right_shoulder")
        if left is None or right is None:
            return None

        lx, ly, lc = left
        rx, ry, rc = right
        if lc < 0.2 or rc < 0.2:
            return None

        angle = degrees(atan2(ry - ly, rx - lx))
        return angle

    @staticmethod
    def _compute_shot_zone(tracked: TrackedFrame) -> str | None:
        if tracked.goal_zones is None or tracked.ball is None:
            return None

        bx, by = int(tracked.ball.center[0]), int(tracked.ball.center[1])
        for zone_name, (x1, y1, x2, y2) in tracked.goal_zones.items():
            if x1 <= bx <= x2 and y1 <= by <= y2:
                return zone_name

        return "outside_goal"

    @staticmethod
    def _compute_goalkeeper_movement(path: deque[tuple[int, int]]) -> str | None:
        if len(path) < 2:
            return None

        first_x, first_y = path[0]
        last_x, last_y = path[-1]
        dx, dy = last_x - first_x, last_y - first_y

        if abs(dx) < 5 and abs(dy) < 5:
            return "stable"

        horizontal = "right" if dx > 0 else "left"
        vertical = "down" if dy > 0 else "up"

        if abs(dx) >= abs(dy):
            return horizontal

        return vertical

    @staticmethod
    def _compute_torso_angle(pose: PoseResult | None) -> float | None:
        if pose is None:
            return None

        ls = pose.keypoints.get("left_shoulder")
        rs = pose.keypoints.get("right_shoulder")
        lh = pose.keypoints.get("left_hip")
        rh = pose.keypoints.get("right_hip")
        if not ls or not rs or not lh or not rh:
            return None

        points = [ls, rs, lh, rh]
        if any(p[2] < 0.2 for p in points):
            return None

        shoulder_center = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        hip_center = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        return degrees(atan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0]))

    @staticmethod
    def _compute_center_of_mass(pose: PoseResult | None) -> tuple[int, int] | None:
        if pose is None:
            return None

        keys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        pts = [pose.keypoints.get(k) for k in keys]
        if any(p is None or p[2] < 0.2 for p in pts):
            return None

        cx = int(sum(p[0] for p in pts if p is not None) / 4.0)
        cy = int(sum(p[1] for p in pts if p is not None) / 4.0)
        return (cx, cy)

    def _update_shot_start(self) -> None:
        if self.shot_start_frame is not None or len(self.ball_path) < 4:
            return

        p0 = self.ball_path[-4]
        p1 = self.ball_path[-1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if (dx * dx + dy * dy) ** 0.5 > 18.0:
            self.shot_start_frame = self.frame_index

    def _update_goalkeeper_reaction(self) -> None:
        if self.goalkeeper_move_start_frame is not None or len(self.goalkeeper_path) < 4:
            return

        p0 = self.goalkeeper_path[-4]
        p1 = self.goalkeeper_path[-1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if (dx * dx + dy * dy) ** 0.5 > 12.0:
            self.goalkeeper_move_start_frame = self.frame_index
