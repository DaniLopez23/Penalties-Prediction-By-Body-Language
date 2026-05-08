"""Shared player detection dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from ..pose.pose_estimator import PoseEstimation


@dataclass
class PlayerDetection:
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
    """Compatibility shim kept for old imports.

    The simplified pipeline does not maintain a custom ghost tracker. It reuses
    the last BoT-SORT role boxes on non-refresh frames.
    """

    def update(self, role: str, confirmed: PlayerDetection | None, post_shot: bool = False):
        _ = role, post_shot
        return confirmed
