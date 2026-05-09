from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import COCO_POSE_EDGES, BallState, Detection, GoalBox, PenaltyAnalysisState, PoseDetection, PoseMetrics
from src.utils.geometry import clamp


class FrameAnnotator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        goal: Optional[GoalBox],
        role_assignments: dict[str, Detection],
        pose_assignments: dict[str, PoseDetection],
        pose_metrics: dict[str, PoseMetrics],
        ball: Optional[BallState],
        ball_trail: Iterable[tuple[int, int]],
        frame_index: int,
        analysis_state: Optional[PenaltyAnalysisState] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        if goal is not None:
            self._draw_goal(annotated, goal)
        for role, det in role_assignments.items():
            self._draw_player(annotated, role, det)
        for role, pose in pose_assignments.items():
            color = self._role_color(role)
            self._draw_pose(annotated, pose, color)
        self._draw_ball_trail(annotated, list(ball_trail))
        if ball is not None:
            self._draw_ball(annotated, ball)
        self._draw_frame_label(annotated, frame_index)
        if analysis_state is not None:
            self._draw_analysis_panel(annotated, analysis_state)
        return annotated

    def _draw_goal(self, frame: np.ndarray, goal: GoalBox) -> None:
        draw_cfg = self.config.draw
        overlay = frame.copy()
        zone_labels = []
        for idx, (name, bounds) in enumerate(goal.zone_bounds().items()):
            x1, y1, x2, y2 = bounds
            zone_color = tuple(int(clamp(channel + idx * 18, 0, 255)) for channel in draw_cfg.goal_zone_color)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_color, thickness=-1)
            zone_labels.append((name, (x1 + 6, y1 + 20), zone_color))
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)
        for name, origin, zone_color in zone_labels:
            self._label(frame, name, origin, zone_color)

        x1, y1, x2, y2 = map(int, (round(goal.x1), round(goal.y1), round(goal.x2), round(goal.y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_cfg.goal_color, draw_cfg.line_thickness)
        third = (x2 - x1) / 3.0
        for k in (1, 2):
            x = int(round(x1 + k * third))
            cv2.line(frame, (x, y1), (x, y2), draw_cfg.goal_color, draw_cfg.line_thickness)
        if goal.tracked and goal.detected:
            source = "tracked"
        else:
            source = "detected" if goal.detected else "estimated"
        self._label(frame, f"goal {source} {goal.confidence:.2f}", (x1, max(18, y1 - 8)), draw_cfg.goal_color)

    def _draw_player(self, frame: np.ndarray, role: str, det: Detection) -> None:
        color = self._role_color(role)
        x1, y1, x2, y2 = map(int, (round(det.x1), round(det.y1), round(det.x2), round(det.y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.draw.line_thickness)
        track = f" id:{det.track_id}" if det.track_id is not None else ""
        label = f"{role}{track} {det.confidence:.2f}"
        self._label(frame, label, (x1, max(18, y1 - 8)), color)

    def _draw_pose(self, frame: np.ndarray, pose: PoseDetection, color: tuple[int, int, int]) -> None:
        keypoints = pose.keypoints
        threshold = self.config.models.pose_keypoint_confidence
        for a, b in COCO_POSE_EDGES:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            if keypoints[a][2] < threshold or keypoints[b][2] < threshold:
                continue
            p1 = tuple(int(round(v)) for v in keypoints[a][:2])
            p2 = tuple(int(round(v)) for v in keypoints[b][:2])
            cv2.line(frame, p1, p2, color, 2)
        for idx, point in enumerate(keypoints):
            if point[2] >= threshold and idx in (5, 6, 7, 8, 11, 12, 13, 14):
                cv2.circle(frame, (int(round(point[0])), int(round(point[1]))), 3, color, thickness=-1)

    def _draw_analysis_panel(self, frame: np.ndarray, analysis: PenaltyAnalysisState) -> None:
        lines = [
            f"shot: {analysis.shot_state}",
            f"ball zone: {analysis.ball_zone or '-'}",
            f"GK direction: {analysis.goalkeeper_direction}",
            f"GK lean: {self._fmt_angle(analysis.goalkeeper_lean_deg)}",
            f"ST L arm/trunk: {self._fmt_angle(analysis.striker_left_arm_trunk_angle_deg)}",
            f"ST R arm/trunk: {self._fmt_angle(analysis.striker_right_arm_trunk_angle_deg)}",
            f"ST lean: {self._fmt_angle(analysis.striker_body_lean_deg)}",
        ]
        x, y = 12, 48
        line_h = 19
        width = 245
        height = line_h * len(lines) + 10
        cv2.rectangle(frame, (x - 4, y - 15), (x + width, y + height - 15), self.config.draw.label_bg, thickness=-1)
        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + idx * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.draw.font_scale,
                self.config.draw.text_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    @staticmethod
    def _fmt_angle(value: Optional[float]) -> str:
        return "-" if value is None else f"{value:+.1f}deg"

    def _draw_ball_trail(self, frame: np.ndarray, trail: list[tuple[int, int]]) -> None:
        if len(trail) < 2:
            return
        for idx in range(1, len(trail)):
            alpha = idx / max(1, len(trail) - 1)
            thickness = max(1, int(round(1 + 3 * alpha)))
            color = tuple(int(channel * alpha) for channel in self.config.draw.ball_color)
            cv2.line(frame, trail[idx - 1], trail[idx], color, thickness)

    def _draw_ball(self, frame: np.ndarray, ball: BallState) -> None:
        color = self.config.draw.ball_color if ball.observed else self.config.draw.predicted_ball_color
        center = tuple(int(round(v)) for v in ball.center)
        cv2.circle(frame, center, ball.radius, color, thickness=2)
        cv2.circle(frame, center, 2, color, thickness=-1)
        suffix = f" id:{ball.track_id}" if ball.track_id is not None else ""
        self._label(frame, f"ball {ball.source}{suffix}", (center[0] + ball.radius + 4, center[1]), color)

    def _draw_frame_label(self, frame: np.ndarray, frame_index: int) -> None:
        self._label(frame, f"frame {frame_index}", (12, 24), (210, 210, 210))

    def _label(self, frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        draw_cfg = self.config.draw
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = origin
        (text_w, text_h), baseline = cv2.getTextSize(text, font, draw_cfg.font_scale, 1)
        x = int(clamp(x, 0, max(0, frame.shape[1] - text_w - 6)))
        y = int(clamp(y, text_h + 4, max(text_h + 4, frame.shape[0] - 4)))
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - baseline - 2),
            (x + text_w + 4, y + baseline + 2),
            draw_cfg.label_bg,
            thickness=-1,
        )
        cv2.putText(frame, text, (x, y), font, draw_cfg.font_scale, color, thickness=1, lineType=cv2.LINE_AA)

    def _role_color(self, role: str) -> tuple[int, int, int]:
        if role == "striker":
            return self.config.draw.striker_color
        if role == "goalkeeper":
            return self.config.draw.goalkeeper_color
        return self.config.draw.text_color
