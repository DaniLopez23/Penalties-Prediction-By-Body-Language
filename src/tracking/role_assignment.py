"""Simple role assignment for behind-the-shooter penalty videos."""

from __future__ import annotations

from dataclasses import dataclass

from ..detectors.players_detector import PlayerDetection
from ..models import ModelConfig


@dataclass
class RoleAssignment:
    shooter: PlayerDetection | None
    goalkeeper: PlayerDetection | None
    roles_frozen: bool = False


class RoleAssigner:
    """Assign goalkeeper near the goal and shooter lower in the image."""

    def __init__(self, ghost_tracker=None):
        _ = ghost_tracker
        self.last_shooter_id: int | None = None
        self.last_goalkeeper_id: int | None = None
        self._locked_shooter: PlayerDetection | None = None
        self._locked_goalkeeper: PlayerDetection | None = None
        self._frozen_shooter: PlayerDetection | None = None
        self._frozen_goalkeeper: PlayerDetection | None = None
        self._roles_frozen = False

    @property
    def roles_frozen(self) -> bool:
        return self._roles_frozen

    def assign(
        self,
        players: list[PlayerDetection],
        goal,
        stable_goal_bbox: tuple[int, int, int, int] | None = None,
        shot_detected: bool = False,
        frame_shape: tuple[int, ...] | None = None,
    ) -> RoleAssignment:
        _ = stable_goal_bbox
        if shot_detected:
            self.lock_current_roles()

        if self._roles_frozen:
            return self._assign_locked(players)

        if not players:
            return RoleAssignment(None, None, self._roles_frozen)

        shooter = self._by_track(players, self.last_shooter_id)
        if shooter is not None and not self._is_valid_shooter(shooter, frame_shape):
            shooter = None
        goalkeeper = self._by_track(players, self.last_goalkeeper_id)
        if shooter is not None and goalkeeper is not None:
            if self._same_or_overlapping_role(shooter, goalkeeper):
                goalkeeper = None

        remaining = [p for p in players if p is not shooter and p is not goalkeeper]
        if goalkeeper is None:
            goalkeeper = self._choose_goalkeeper(remaining, goal)
        remaining = [p for p in players if p is not shooter and p is not goalkeeper]
        if shooter is None:
            shooter = self._choose_shooter(remaining, goalkeeper, frame_shape)

        if shooter is not None and goalkeeper is not None:
            if self._same_or_overlapping_role(shooter, goalkeeper):
                goalkeeper = None

        if shooter is not None and shooter.track_id is not None:
            self.last_shooter_id = shooter.track_id
        if goalkeeper is not None and goalkeeper.track_id is not None:
            self.last_goalkeeper_id = goalkeeper.track_id
        self._locked_shooter = shooter or self._locked_shooter
        self._locked_goalkeeper = goalkeeper or self._locked_goalkeeper

        return RoleAssignment(shooter, goalkeeper, self._roles_frozen)

    def lock_current_roles(self) -> None:
        if self._roles_frozen:
            return
        if self.last_shooter_id is None and self._locked_shooter is None:
            return
        if self.last_goalkeeper_id is None and self._locked_goalkeeper is None:
            return
        if (
            self._locked_shooter is not None
            and self._locked_goalkeeper is not None
            and self._same_or_overlapping_role(self._locked_shooter, self._locked_goalkeeper)
        ):
            self._locked_goalkeeper = None
            self.last_goalkeeper_id = None
            return
        self._frozen_shooter = self._locked_shooter
        self._frozen_goalkeeper = self._locked_goalkeeper
        self._roles_frozen = True

    def _assign_locked(self, players: list[PlayerDetection]) -> RoleAssignment:
        previous_goalkeeper = self._locked_goalkeeper
        shooter = self._safe_locked_update(
            players,
            self.last_shooter_id,
            self._locked_shooter,
            self._locked_goalkeeper,
            allow_horizontal_recovery=False,
        )
        goalkeeper = self._safe_locked_update(
            players,
            self.last_goalkeeper_id,
            self._locked_goalkeeper,
            shooter or self._locked_shooter,
            allow_horizontal_recovery=True,
        )

        self._locked_shooter = shooter or self._locked_shooter or self._frozen_shooter
        self._locked_goalkeeper = (
            goalkeeper or self._locked_goalkeeper or self._frozen_goalkeeper
        )
        if (
            self._locked_shooter is not None
            and self._locked_goalkeeper is not None
            and self._same_or_overlapping_role(self._locked_shooter, self._locked_goalkeeper)
        ):
            if previous_goalkeeper is not None and not self._same_or_overlapping_role(
                self._locked_shooter, previous_goalkeeper
            ):
                self._locked_goalkeeper = previous_goalkeeper
            else:
                self._locked_goalkeeper = None
        return RoleAssignment(self._locked_shooter, self._locked_goalkeeper, True)

    def _safe_locked_update(
        self,
        players: list[PlayerDetection],
        track_id: int | None,
        current: PlayerDetection | None,
        other_role: PlayerDetection | None,
        allow_horizontal_recovery: bool,
    ) -> PlayerDetection | None:
        if current is None:
            return None
        candidate = self._by_track(players, track_id)
        if candidate is None and allow_horizontal_recovery:
            candidate = self._find_horizontal_recovery(players, current, other_role)
        if candidate is None:
            return current

        if self._distance(candidate.center, current.center) > ModelConfig.ROLE_LOCK_MAX_UPDATE_DIST:
            return current
        if other_role is not None:
            if self._distance(candidate.center, other_role.center) < self._distance(
                candidate.center, current.center
            ):
                return current
            if self._iou(candidate.bbox_xyxy, other_role.bbox_xyxy) > ModelConfig.ROLE_LOCK_MAX_IOU:
                return current
        return candidate

    def _find_horizontal_recovery(
        self,
        players: list[PlayerDetection],
        current: PlayerDetection,
        other_role: PlayerDetection | None,
    ) -> PlayerDetection | None:
        candidates = []
        for player in players:
            if other_role is not None and self._same_or_overlapping_role(player, other_role):
                continue
            if abs(player.center[1] - current.center[1]) > ModelConfig.ROLE_LOCK_MAX_VERTICAL_DRIFT:
                continue
            if abs(player.center[0] - current.center[0]) > ModelConfig.ROLE_LOCK_MAX_HORIZONTAL_RECOVERY_DIST:
                continue
            candidates.append(player)
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._distance(p.center, current.center))

    @staticmethod
    def _by_track(players: list[PlayerDetection], track_id: int | None) -> PlayerDetection | None:
        if track_id is None:
            return None
        return next((p for p in players if p.track_id == track_id), None)

    @staticmethod
    def _choose_goalkeeper(players: list[PlayerDetection], goal) -> PlayerDetection | None:
        if not players:
            return None
        if goal is None:
            return min(players, key=lambda p: p.center[1])

        gx1, gy1, gx2, gy2 = goal.bbox_xyxy
        goal_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
        goal_diag = ((gx2 - gx1) ** 2 + (gy2 - gy1) ** 2) ** 0.5

        near_goal = [
            p
            for p in players
            if RoleAssigner._distance(p.center, goal_center)
            <= goal_diag * ModelConfig.GOALKEEPER_MAX_GOAL_DIST_RATIO
        ]
        candidates = near_goal or players
        return min(candidates, key=lambda p: RoleAssigner._distance(p.center, goal_center))

    @staticmethod
    def _choose_shooter(
        players: list[PlayerDetection],
        goalkeeper: PlayerDetection | None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> PlayerDetection | None:
        candidates = [p for p in players if p is not goalkeeper]
        candidates = [p for p in candidates if RoleAssigner._is_valid_shooter(p, frame_shape)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.center[1])

    @staticmethod
    def _is_valid_shooter(
        player: PlayerDetection,
        frame_shape: tuple[int, ...] | None = None,
    ) -> bool:
        if frame_shape is None:
            return True

        frame_h, frame_w = frame_shape[:2]
        cx, cy = player.center
        return (
            ModelConfig.SHOOTER_CENTER_X_MIN * frame_w
            <= cx
            <= ModelConfig.SHOOTER_CENTER_X_MAX * frame_w
            and cy >= ModelConfig.SHOOTER_BOTTOM_Y_MIN * frame_h
        )

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    @classmethod
    def _same_or_overlapping_role(
        cls,
        a: PlayerDetection,
        b: PlayerDetection,
    ) -> bool:
        if a is b:
            return True
        if a.track_id is not None and b.track_id is not None and a.track_id == b.track_id:
            return True
        if cls._distance(a.center, b.center) < ModelConfig.ROLE_MIN_ROLE_CENTER_DIST:
            return True
        return cls._iou(a.bbox_xyxy, b.bbox_xyxy) > ModelConfig.ROLE_LOCK_MAX_IOU

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0
