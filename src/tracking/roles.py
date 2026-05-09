from __future__ import annotations

import math
from typing import Optional

from src.config import PipelineConfig
from src.models import Detection, GoalBox
from src.utils.geometry import center_distance, clamp


class PlayerRoleAssigner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.locked_ids: dict[str, int] = {}
        self.previous: dict[str, Detection] = {}
        self.missing_counts: dict[str, int] = {"striker": 999, "goalkeeper": 999}

    def assign(
        self,
        persons: list[Detection],
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
        ball_center: Optional[tuple[float, float]] = None,
    ) -> dict[str, Detection]:
        assignments: dict[str, Detection] = {}
        used_indexes: set[int] = set()

        if self.config.roles.lock_track_ids:
            by_id = {det.track_id: (idx, det) for idx, det in enumerate(persons) if det.track_id is not None}
            for role, track_id in self.locked_ids.items():
                if track_id in by_id:
                    idx, det = by_id[track_id]
                    if role != "goalkeeper" or self._locked_goalkeeper_is_plausible(det, frame_shape, goal):
                        assignments[role] = det
                        used_indexes.add(idx)

        missing_roles = [
            role
            for role in ("goalkeeper", "striker")
            if role not in assignments
            and self.missing_counts.get(role, 999) >= self.config.roles.lost_reassign_frames
        ]
        if set(missing_roles) == {"goalkeeper", "striker"}:
            global_assignments = self._best_global_assignment(persons, used_indexes, frame_shape, goal, ball_center)
            for role, idx, det, score in global_assignments:
                if score < self.config.roles.min_assignment_score:
                    continue
                assignments[role] = det
                used_indexes.add(idx)
                if det.track_id is not None and self.config.roles.lock_track_ids:
                    self.locked_ids[role] = det.track_id
            missing_roles = [
                role
                for role in ("goalkeeper", "striker")
                if role not in assignments
                and self.missing_counts.get(role, 999) >= self.config.roles.lost_reassign_frames
            ]

        for role in missing_roles:
            candidate = self._best_candidate_for_role(role, persons, used_indexes, frame_shape, goal, ball_center)
            if candidate is None:
                continue
            idx, det, score = candidate
            if score < self.config.roles.min_assignment_score:
                continue
            assignments[role] = det
            used_indexes.add(idx)
            if det.track_id is not None and self.config.roles.lock_track_ids:
                self.locked_ids[role] = det.track_id

        for role in ("goalkeeper", "striker"):
            if role in assignments:
                self.previous[role] = assignments[role]
                self.missing_counts[role] = 0
            else:
                self.missing_counts[role] = self.missing_counts.get(role, 0) + 1

        return assignments

    def _locked_goalkeeper_is_plausible(
        self,
        det: Detection,
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
    ) -> bool:
        if goal is None:
            return True
        if self.missing_counts.get("goalkeeper", 0) >= self.config.roles.goalkeeper_reassign_after_frames:
            return False
        margin = max(goal.width, goal.height) * self.config.roles.goalkeeper_lock_goal_margin_ratio
        x, y = det.center
        return (
            goal.x1 - margin <= x <= goal.x2 + margin
            and goal.y1 - margin <= y <= goal.y2 + margin
        )

    def _best_global_assignment(
        self,
        persons: list[Detection],
        used_indexes: set[int],
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
        ball_center: Optional[tuple[float, float]],
    ) -> list[tuple[str, int, Detection, float]]:
        available = [(idx, det) for idx, det in enumerate(persons) if idx not in used_indexes]
        if not available:
            return []

        role_scores = {
            idx: {
                "goalkeeper": self._goalkeeper_score(det, frame_shape, goal),
                "striker": self._striker_score(det, frame_shape, goal, ball_center),
            }
            for idx, det in available
        }

        if len(available) == 1:
            idx, det = available[0]
            role = max(role_scores[idx], key=role_scores[idx].get)
            return [(role, idx, det, role_scores[idx][role])]

        best_pair: Optional[tuple[float, tuple[int, Detection], tuple[int, Detection]]] = None
        for goalkeeper_candidate in available:
            for striker_candidate in available:
                if goalkeeper_candidate[0] == striker_candidate[0]:
                    continue
                score = (
                    role_scores[goalkeeper_candidate[0]]["goalkeeper"]
                    + role_scores[striker_candidate[0]]["striker"]
                )
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, goalkeeper_candidate, striker_candidate)

        if best_pair is None:
            return []
        _, goalkeeper_candidate, striker_candidate = best_pair
        goalkeeper_idx, goalkeeper = goalkeeper_candidate
        striker_idx, striker = striker_candidate
        return [
            ("goalkeeper", goalkeeper_idx, goalkeeper, role_scores[goalkeeper_idx]["goalkeeper"]),
            ("striker", striker_idx, striker, role_scores[striker_idx]["striker"]),
        ]

    def _best_candidate_for_role(
        self,
        role: str,
        persons: list[Detection],
        used_indexes: set[int],
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
        ball_center: Optional[tuple[float, float]],
    ) -> Optional[tuple[int, Detection, float]]:
        scores = []
        for idx, det in enumerate(persons):
            if idx in used_indexes:
                continue
            score = self._goalkeeper_score(det, frame_shape, goal) if role == "goalkeeper" else self._striker_score(det, frame_shape, goal, ball_center)
            if role in self.previous:
                previous_distance = center_distance(det.center, self.previous[role].center)
                diag = math.hypot(frame_shape[1], frame_shape[0])
                score += 0.25 * (1.0 - clamp(previous_distance / max(1.0, diag * 0.25), 0.0, 1.0))
            scores.append((idx, det, score))
        if not scores:
            return None
        return max(scores, key=lambda item: item[2])

    def _striker_score(
        self,
        det: Detection,
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
        ball_center: Optional[tuple[float, float]] = None,
    ) -> float:
        height, width = frame_shape[:2]
        cx, cy = det.center
        lower_score = clamp(cy / height, 0.0, 1.0)
        center_score = 1.0 - clamp(abs(cx - width * 0.5) / (width * 0.5), 0.0, 1.0)
        size_score = clamp(det.area / max(1.0, width * height * 0.12), 0.0, 1.0)
        ball_score = 0.0
        if ball_center is not None:
            feet_center = ((det.x1 + det.x2) * 0.5, det.y2)
            distance = center_distance(feet_center, ball_center)
            max_distance = max(1.0, math.hypot(width, height) * self.config.roles.striker_ball_max_distance_ratio)
            ball_score = 1.0 - clamp(distance / max_distance, 0.0, 1.0)
        goal_penalty = 0.0
        if goal is not None:
            goal_margin = max(goal.width, goal.height) * self.config.roles.goalkeeper_goal_margin_ratio
            if goal.contains(det.center, goal_margin):
                goal_penalty = 1.2
        role_cfg = self.config.roles
        return (
            role_cfg.striker_bottom_weight * lower_score
            + role_cfg.striker_center_weight * center_score
            + role_cfg.striker_size_weight * size_score
            + role_cfg.striker_ball_weight * ball_score
            - goal_penalty
        ) / (
            role_cfg.striker_bottom_weight
            + role_cfg.striker_center_weight
            + role_cfg.striker_size_weight
            + role_cfg.striker_ball_weight
        )

    def _goalkeeper_score(
        self,
        det: Detection,
        frame_shape: tuple[int, int, int],
        goal: Optional[GoalBox],
    ) -> float:
        height, width = frame_shape[:2]
        cx, cy = det.center
        upper_score = 1.0 - clamp(cy / height, 0.0, 1.0)
        image_center_score = 1.0 - clamp(abs(cx - width * 0.5) / (width * 0.5), 0.0, 1.0)
        if goal is None:
            return 0.55 * upper_score + 0.45 * image_center_score

        goal_margin = max(goal.width, goal.height) * self.config.roles.goalkeeper_goal_margin_ratio
        goal_hit = 1.0 if goal.contains(det.center, goal_margin) else 0.0
        gx, gy = goal.center
        goal_center_score = 1.0 - clamp(abs(cx - gx) / max(1.0, goal.width * 0.75), 0.0, 1.0)
        goal_depth_score = 1.0 - clamp(abs(cy - gy) / max(1.0, goal.height * 1.2), 0.0, 1.0)
        role_cfg = self.config.roles
        return (
            role_cfg.goalkeeper_goal_weight * goal_hit
            + role_cfg.goalkeeper_center_weight * goal_center_score
            + role_cfg.goalkeeper_depth_weight * goal_depth_score
            + 0.45 * upper_score
        ) / (
            role_cfg.goalkeeper_goal_weight
            + role_cfg.goalkeeper_center_weight
            + role_cfg.goalkeeper_depth_weight
            + 0.45
        )
