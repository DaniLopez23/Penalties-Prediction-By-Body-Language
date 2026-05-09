from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.models import Detection, PoseDetection, PoseMetrics
from src.utils.geometry import box_iou, center_distance, clamp


class PoseAnalyzer:
    def __init__(self, keypoint_confidence: float) -> None:
        self.keypoint_confidence = keypoint_confidence

    def assign_poses(
        self,
        poses: list[PoseDetection],
        role_assignments: dict[str, Detection],
    ) -> dict[str, PoseDetection]:
        assigned: dict[str, PoseDetection] = {}
        used_pose_indexes: set[int] = set()
        for role, player in role_assignments.items():
            best_idx = None
            best_score = 0.0
            for idx, pose in enumerate(poses):
                if idx in used_pose_indexes:
                    continue
                pose_center = ((pose.xyxy[0] + pose.xyxy[2]) * 0.5, (pose.xyxy[1] + pose.xyxy[3]) * 0.5)
                iou = box_iou(player.xyxy, pose.xyxy)
                distance = center_distance(player.center, pose_center)
                max_side = max(player.width, player.height, 1.0)
                score = iou + 0.35 * (1.0 - clamp(distance / max_side, 0.0, 1.0))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score > 0.18:
                assigned[role] = poses[best_idx]
                used_pose_indexes.add(best_idx)
        return assigned

    def metrics(self, pose: PoseDetection) -> PoseMetrics:
        kp = pose.keypoints
        shoulder_angle = self._segment_angle(kp, 5, 6)
        hip_angle = self._segment_angle(kp, 11, 12)
        body_lean = self._body_lean(kp)
        return PoseMetrics(
            body_lean_deg=body_lean,
            shoulder_angle_deg=shoulder_angle,
            hip_angle_deg=hip_angle,
            left_arm_trunk_angle_deg=self._arm_trunk_angle(kp, shoulder_idx=5, elbow_idx=7, hip_idx=11),
            right_arm_trunk_angle_deg=self._arm_trunk_angle(kp, shoulder_idx=6, elbow_idx=8, hip_idx=12),
        )

    def _valid_point(self, keypoints: np.ndarray, idx: int) -> bool:
        return idx < len(keypoints) and keypoints[idx][2] >= self.keypoint_confidence

    def _segment_angle(self, keypoints: np.ndarray, a: int, b: int) -> Optional[float]:
        if not self._valid_point(keypoints, a) or not self._valid_point(keypoints, b):
            return None
        p1, p2 = keypoints[a], keypoints[b]
        return math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))

    def _body_lean(self, keypoints: np.ndarray) -> Optional[float]:
        needed = (5, 6, 11, 12)
        if not all(self._valid_point(keypoints, idx) for idx in needed):
            return None
        shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) * 0.5
        hip_mid = (keypoints[11][:2] + keypoints[12][:2]) * 0.5
        dx = float(shoulder_mid[0] - hip_mid[0])
        dy = float(hip_mid[1] - shoulder_mid[1])
        if abs(dy) < 1e-6:
            return None
        return math.degrees(math.atan2(dx, dy))

    def _arm_trunk_angle(
        self,
        keypoints: np.ndarray,
        shoulder_idx: int,
        elbow_idx: int,
        hip_idx: int,
    ) -> Optional[float]:
        if not all(self._valid_point(keypoints, idx) for idx in (shoulder_idx, elbow_idx, hip_idx)):
            return None
        shoulder = keypoints[shoulder_idx][:2]
        elbow = keypoints[elbow_idx][:2]
        hip = keypoints[hip_idx][:2]
        arm = elbow - shoulder
        trunk = hip - shoulder
        arm_norm = float(np.linalg.norm(arm))
        trunk_norm = float(np.linalg.norm(trunk))
        if arm_norm < 1e-6 or trunk_norm < 1e-6:
            return None
        cosine = float(np.dot(arm, trunk) / (arm_norm * trunk_norm))
        cosine = clamp(cosine, -1.0, 1.0)
        return math.degrees(math.acos(cosine))

