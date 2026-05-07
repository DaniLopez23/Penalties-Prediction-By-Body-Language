"""Player detection data structures and role ghost tracking."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ModelConfig
from ..pose.pose_estimator import PoseEstimation


@dataclass
class PlayerDetection:
    """Detection/tracking result for one player."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None
    pose: PoseEstimation | None = None
    predicted: bool = False

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class PlayerGhostTracker:
    """Keep role boxes alive through short detector/tracker gaps."""

    GK_MAX_GHOST_FRAMES = ModelConfig.GK_MAX_GHOST_FRAMES
    GK_MAX_GHOST_FRAMES_POST_SHOT = ModelConfig.GK_MAX_GHOST_FRAMES_POST_SHOT
    SH_MAX_GHOST_FRAMES = ModelConfig.SHOOTER_MAX_GHOST_FRAMES
    GHOST_CONF_DECAY = ModelConfig.GHOST_CONF_DECAY
    GHOST_CONF_DECAY_POST_SHOT = ModelConfig.GHOST_CONF_DECAY_POST_SHOT

    def __init__(self):
        self._ghost: dict[str, dict] = {
            "goalkeeper": {"det": None, "missed": 0, "velocity": (0.0, 0.0)},
            "shooter": {"det": None, "missed": 0, "velocity": (0.0, 0.0)},
        }

    def update(
        self,
        role: str,
        confirmed: PlayerDetection | None,
        post_shot: bool = False,
    ) -> PlayerDetection | None:
        if role == "goalkeeper" and post_shot:
            max_miss = self.GK_MAX_GHOST_FRAMES_POST_SHOT
            decay = self.GHOST_CONF_DECAY_POST_SHOT
        elif role == "goalkeeper":
            max_miss = self.GK_MAX_GHOST_FRAMES
            decay = self.GHOST_CONF_DECAY
        else:
            max_miss = self.SH_MAX_GHOST_FRAMES
            decay = self.GHOST_CONF_DECAY

        state = self._ghost[role]

        if confirmed is not None:
            self._update_velocity(state, confirmed)
            state["det"] = confirmed
            state["missed"] = 0
            return confirmed

        state["missed"] += 1
        if state["det"] is None or state["missed"] > max_miss:
            return None

        return self._make_ghost(state, decay)

    def reset(self, role: str) -> None:
        self._ghost[role] = {"det": None, "missed": 0, "velocity": (0.0, 0.0)}

    def last_center(self, role: str) -> tuple[float, float] | None:
        det = self._ghost.get(role, {}).get("det")
        return det.center if det is not None else None

    def missed(self, role: str) -> int:
        return int(self._ghost.get(role, {}).get("missed", 0))

    @staticmethod
    def _update_velocity(state: dict, confirmed: PlayerDetection) -> None:
        prev = state.get("det")
        if prev is None:
            return
        vx = confirmed.center[0] - prev.center[0]
        vy = confirmed.center[1] - prev.center[1]
        old_vx, old_vy = state.get("velocity", (0.0, 0.0))
        state["velocity"] = (0.55 * old_vx + 0.45 * vx, 0.55 * old_vy + 0.45 * vy)

    @staticmethod
    def _make_ghost(state: dict, decay: float) -> PlayerDetection:
        det = state["det"]
        missed = state["missed"]
        ghost_conf = max(0.05, det.confidence * (decay**missed))

        vx, vy = state.get("velocity", (0.0, 0.0))
        motion_decay = 0.92 ** max(0, missed - 1)
        shift_x = vx * missed * motion_decay
        shift_y = vy * missed * motion_decay

        x1, y1, x2, y2 = det.bbox_xyxy
        return PlayerDetection(
            bbox_xyxy=(
                int(x1 + shift_x),
                int(y1 + shift_y),
                int(x2 + shift_x),
                int(y2 + shift_y),
            ),
            confidence=ghost_conf,
            center=(det.center[0] + shift_x, det.center[1] + shift_y),
            track_id=det.track_id,
            pose=det.pose,
            predicted=True,
        )
