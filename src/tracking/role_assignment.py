"""Role assignment for shooter and goalkeeper."""

from __future__ import annotations

from dataclasses import dataclass

from ..detectors.players_detector import PlayerDetection, PlayerGhostTracker
from ..models import ModelConfig


@dataclass
class RoleAssignment:
    shooter: PlayerDetection | None
    goalkeeper: PlayerDetection | None
    roles_frozen: bool


class RoleAssigner:
    """Assign player detections to penalty-specific roles."""

    def __init__(self, ghost_tracker: PlayerGhostTracker):
        self.ghost_tracker = ghost_tracker
        self._last_role_map: dict[int, str] = {}
        self._stable_role_map: dict[int, str] = {}
        self._candidate_role_key: tuple[tuple[int, str], ...] | None = None
        self._candidate_role_frames = 0
        self._frozen_role_map: dict[int, str] = {}
        self._roles_frozen = False

    @property
    def roles_frozen(self) -> bool:
        return self._roles_frozen

    def assign(
        self,
        players: list[PlayerDetection],
        goal,
        stable_goal_bbox: tuple[int, int, int, int] | None,
        shot_detected: bool,
    ) -> RoleAssignment:
        players = self._filter_play_zone(players, stable_goal_bbox)
        current_track_ids = {p.track_id: p for p in players if p.track_id is not None}

        if shot_detected and not self._roles_frozen:
            freeze_source = self._stable_role_map or self._last_role_map
            if freeze_source:
                self._frozen_role_map = dict(freeze_source)
                self._roles_frozen = True

        if not players:
            return RoleAssignment(
                shooter=self.ghost_tracker.update("shooter", None, shot_detected),
                goalkeeper=self.ghost_tracker.update("goalkeeper", None, shot_detected),
                roles_frozen=self._roles_frozen,
            )

        shooter = None
        goalkeeper = None
        role_map = self._frozen_role_map if self._roles_frozen else self._last_role_map

        for track_id, role in role_map.items():
            player = current_track_ids.get(track_id)
            if player is None:
                continue
            if role == "shooter":
                shooter = player
            elif role == "goalkeeper":
                goalkeeper = player

        if self._roles_frozen:
            shooter, goalkeeper = self._recover_frozen_roles(
                players, shooter, goalkeeper, goal
            )
        else:
            shooter, goalkeeper = self._assign_spatial_roles(players, shooter, goalkeeper, goal)
            self._update_live_role_map(shooter, goalkeeper)

        return RoleAssignment(
            shooter=self.ghost_tracker.update("shooter", shooter, shot_detected),
            goalkeeper=self.ghost_tracker.update("goalkeeper", goalkeeper, shot_detected),
            roles_frozen=self._roles_frozen,
        )

    def _assign_spatial_roles(
        self,
        players: list[PlayerDetection],
        shooter: PlayerDetection | None,
        goalkeeper: PlayerDetection | None,
        goal,
    ) -> tuple[PlayerDetection | None, PlayerDetection | None]:
        available = [p for p in players if p is not shooter and p is not goalkeeper]
        goal_center = self._goal_center(goal)

        if goalkeeper is None and available:
            if goal_center is not None:
                goalkeeper = min(
                    available,
                    key=lambda p: self._distance(p.center, goal_center)
                    + max(0.0, p.center[1] - goal_center[1]) * 0.6,
                )
            else:
                goalkeeper = min(available, key=lambda p: p.center[1])
            available = [p for p in available if p is not goalkeeper]

        if shooter is None and available:
            shooter = max(available, key=lambda p: p.center[1])

        return shooter, goalkeeper

    def _recover_frozen_roles(
        self,
        players: list[PlayerDetection],
        shooter: PlayerDetection | None,
        goalkeeper: PlayerDetection | None,
        goal,
    ) -> tuple[PlayerDetection | None, PlayerDetection | None]:
        available = [p for p in players if p is not shooter and p is not goalkeeper]

        if goalkeeper is None and available:
            target = self.ghost_tracker.last_center("goalkeeper") or self._goal_center(goal)
            if target is not None:
                candidate = min(available, key=lambda p: self._distance(p.center, target))
                if self._distance(candidate.center, target) < ModelConfig.ROLE_GK_REACQUIRE_MAX_DIST:
                    goalkeeper = candidate
                    if candidate.track_id is not None:
                        self._frozen_role_map[candidate.track_id] = "goalkeeper"
                    available = [p for p in available if p is not candidate]

        if shooter is None and available:
            target = self.ghost_tracker.last_center("shooter")
            if target is not None:
                candidate = min(available, key=lambda p: self._distance(p.center, target))
                if self._distance(candidate.center, target) < ModelConfig.ROLE_SHOOTER_REACQUIRE_MAX_DIST:
                    shooter = candidate
                    if candidate.track_id is not None:
                        self._frozen_role_map[candidate.track_id] = "shooter"

        return shooter, goalkeeper

    def _update_live_role_map(
        self,
        shooter: PlayerDetection | None,
        goalkeeper: PlayerDetection | None,
    ) -> None:
        current_map: dict[int, str] = {}
        if shooter is not None and shooter.track_id is not None:
            current_map[shooter.track_id] = "shooter"
        if goalkeeper is not None and goalkeeper.track_id is not None:
            current_map[goalkeeper.track_id] = "goalkeeper"

        self._last_role_map = current_map
        self._update_stable_role_map(current_map)

    def _update_stable_role_map(self, current_map: dict[int, str]) -> None:
        if len(set(current_map.values())) < 2:
            self._candidate_role_key = None
            self._candidate_role_frames = 0
            return

        key = tuple(sorted(current_map.items()))
        if key == self._candidate_role_key:
            self._candidate_role_frames += 1
        else:
            self._candidate_role_key = key
            self._candidate_role_frames = 1

        if self._candidate_role_frames >= ModelConfig.ROLE_STABLE_FRAMES_BEFORE_FREEZE:
            self._stable_role_map = dict(current_map)

    @staticmethod
    def _filter_play_zone(
        players: list[PlayerDetection],
        stable_goal_bbox: tuple[int, int, int, int] | None,
    ) -> list[PlayerDetection]:
        if stable_goal_bbox is None or not players:
            return players

        gx1, gy1, gx2, gy2 = stable_goal_bbox
        crossbar_margin = ModelConfig.ROLE_CROSSBAR_MARGIN
        lateral_margin = ModelConfig.ROLE_LATERAL_MARGIN

        def in_play_zone(player: PlayerDetection) -> bool:
            if player.center[1] > gy2:
                return True
            if player.bbox_xyxy[1] >= gy1 + crossbar_margin:
                return True
            return gx1 - lateral_margin <= player.center[0] <= gx2 + lateral_margin

        return [p for p in players if in_play_zone(p)]

    @staticmethod
    def _goal_center(goal) -> tuple[float, float] | None:
        if goal is None:
            return None
        x1, y1, x2, y2 = goal.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
