"""Penalty kick analysis pipeline.

The pipeline is organized around the entities we need through the whole kick:
goal, ball, shooter, goalkeeper and their poses. Expensive model calls are kept
to one tracked pose pass for people and one specialized ball pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detectors.ball_detector import BallDetector
from .detectors.goal_detector import GoalDetector
from .detectors.players_detector import PlayerDetection, PlayerGhostTracker
from .detectors.pose_players_detector import PosePlayersDetector
from .models import ModelConfig
from .penalty_metrics import MetricsCalculator, PenaltyMetrics
from .tracking.role_assignment import RoleAssigner
from .video_io import VideoReader, VideoWriter
from utils.drawings import draw_pose, draw_trajectory


COLORS = {
    "shooter": (0, 200, 0),
    "goalkeeper": (0, 0, 255),
    "goalkeeper_ghost": (0, 200, 255),
    "ball": (0, 255, 255),
    "goal": (255, 255, 0),
    "text": (255, 255, 255),
}


class PenaltyPipeline:
    """Main orchestrator for detection, tracking, roles, pose metrics and video output."""

    def __init__(
        self,
        ball_model: Optional[str] = None,
        players_model: Optional[str] = None,
        ball_confidence: Optional[float] = None,
        players_confidence: Optional[float] = None,
        process_every_n_frames: Optional[int] = None,
        goal_detect_every_n_frames: Optional[int] = None,
    ):
        ball_model = ball_model or ModelConfig.get_ball_model_path()
        players_model = players_model or ModelConfig.get_players_model_path()
        ball_confidence = (
            ball_confidence if ball_confidence is not None else ModelConfig.BALL_CONFIDENCE
        )
        players_confidence = (
            players_confidence
            if players_confidence is not None
            else ModelConfig.PLAYERS_CONFIDENCE
        )

        self.ball_detector = BallDetector(ball_model, ball_confidence)
        self.pose_players_detector = PosePlayersDetector(players_model, players_confidence)
        self.goal_detector = GoalDetector()
        self.metrics_calculator = MetricsCalculator()
        self.player_ghost_tracker = PlayerGhostTracker()
        self.role_assigner = RoleAssigner(self.player_ghost_tracker)

        # The pipeline runs every frame. Player tracking is delegated to
        # Ultralytics BoT-SORT through PosePlayersDetector.detect().
        self.process_every_n_frames = max(
            1,
            process_every_n_frames or ModelConfig.PROCESS_EVERY_N_FRAMES,
        )
        self.goal_detect_every_n_frames = max(
            1,
            goal_detect_every_n_frames or ModelConfig.GOAL_DETECT_EVERY_N_FRAMES,
        )

        self.last_ball = None
        self.last_goal = None
        self.last_shooter: PlayerDetection | None = None
        self.last_goalkeeper: PlayerDetection | None = None
        self.last_metrics: PenaltyMetrics | None = None

        self._stable_goal_bbox: tuple[int, int, int, int] | None = None
        self._goal_detection_count = 0
        self._GOAL_STABLE_AFTER = 4

        self._shot_detected = False
        self._shot_frame_idx: int | None = None
        self._ball_trajectory: list[tuple[int, int]] = []
        self._last_ball_detection_frame = -10_000
        self._last_players_track_frame = -10_000

    def process_video(
        self,
        input_video: str | Path,
        output_video: Optional[str | Path] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
    ) -> Path:
        video_reader = VideoReader(input_video)

        if output_video is None:
            output_video = Path("data/cv_output") / f"annotated_{Path(input_video).stem}.mp4"

        video_writer = VideoWriter(
            output_video,
            fps=video_reader.fps,
            width=video_reader.width,
            height=video_reader.height,
        )

        frame_count = 0

        try:
            for frame_idx, frame in video_reader:
                self._process_frame(frame, frame_idx, video_reader.fps)

                annotated = self._draw_annotations(frame, frame_idx)
                video_writer.write(annotated)

                if show_preview:
                    cv2.imshow("Penalty Analysis", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            video_reader.release()
            video_writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames")
        print(f"Output saved to: {output_video}")
        return Path(output_video)

    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> None:
        goal = self._detect_goal_if_needed(frame, frame_idx)
        effective_goal = goal or self._make_goal_from_stable() or self.last_goal

        goal_bbox = self._stable_goal_bbox
        if goal_bbox is None and effective_goal is not None:
            goal_bbox = effective_goal.bbox_xyxy
        self.ball_detector.set_goal_bbox(goal_bbox)

        ball = self._update_ball_state(frame, frame_idx)
        self._update_shot_state(ball, frame_idx)
        self._infer_shot_from_ball_loss(ball, frame_idx)

        shooter, goalkeeper = self._track_players(frame, frame_idx, effective_goal)

        self.last_ball = ball
        self.last_goal = effective_goal
        self.last_shooter = shooter
        self.last_goalkeeper = goalkeeper
        self.last_metrics = self.metrics_calculator.update(
            frame_idx=frame_idx,
            ball_center=(
                ball.center
                if ball is not None and not getattr(ball, "predicted", False)
                else None
            ),
            goalkeeper_center=goalkeeper.center if goalkeeper is not None else None,
            shooter_pose=shooter.pose if shooter is not None else None,
            goalkeeper_pose=goalkeeper.pose if goalkeeper is not None else None,
            fps=fps,
        )

    def _update_ball_state(self, frame: np.ndarray, frame_idx: int):
        mode = self._ball_update_mode(frame_idx)
        if mode in {"detect", "reacquire"}:
            self._last_ball_detection_frame = frame_idx
            return self.ball_detector.detect(frame)

        tracked = self.ball_detector.track_without_detection()
        if tracked is not None:
            return tracked

        self._last_ball_detection_frame = frame_idx
        return self.ball_detector.detect(frame)

    def _ball_update_mode(self, frame_idx: int) -> str:
        if self.last_ball is None or self.ball_detector.last_detection is None:
            return "detect"
        if self.ball_detector.missed_frames > 0:
            cadence = max(1, ModelConfig.BALL_REACQUIRE_EVERY_N_FRAMES)
            return (
                "reacquire"
                if frame_idx - self._last_ball_detection_frame >= cadence
                else "predict"
            )
        if self.last_ball.confidence < 0.10:
            return "detect"

        if self._shot_detected:
            cadence = max(1, ModelConfig.BALL_DETECT_EVERY_N_FRAMES_POST_SHOT)
            if self._shot_frame_idx is not None:
                force_window = ModelConfig.BALL_FORCE_DETECT_FRAMES_AFTER_SHOT
                if frame_idx - self._shot_frame_idx <= force_window:
                    cadence = 1
        else:
            cadence = max(1, ModelConfig.BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT)

        if frame_idx - self._last_ball_detection_frame >= cadence:
            return "detect"
        return "predict"

    def _track_players(
        self,
        frame: np.ndarray,
        frame_idx: int,
        effective_goal,
    ) -> tuple[PlayerDetection | None, PlayerDetection | None]:
        cadence = max(1, ModelConfig.PLAYERS_TRACK_EVERY_N_FRAMES)
        should_track = (
            self.last_shooter is None
            or self.last_goalkeeper is None
            or frame_idx - self._last_players_track_frame >= cadence
        )
        if not should_track:
            return self.last_shooter, self.last_goalkeeper

        players_frame = self._make_players_frame(frame, effective_goal)
        tracked_players = self.pose_players_detector.track(players_frame)
        self._last_players_track_frame = frame_idx
        roles = self.role_assigner.assign(
            tracked_players, effective_goal, self._stable_goal_bbox, self._shot_detected
        )
        return roles.shooter, roles.goalkeeper

    def _make_players_frame(self, frame: np.ndarray, goal) -> np.ndarray:
        return self.goal_detector.mask_for_player_detection(
            frame,
            goal,
            side_padding=ModelConfig.PLAYERS_MASK_SIDE_PADDING,
            top_padding=ModelConfig.PLAYERS_MASK_TOP_PADDING,
            keep_field_below_goal=ModelConfig.PLAYERS_MASK_KEEP_FIELD_BELOW_GOAL,
            blur_kernel=ModelConfig.MASK_BLUR_KERNEL,
        )

    def _detect_goal_if_needed(self, frame: np.ndarray, frame_idx: int):
        should_refresh = (
            self._stable_goal_bbox is None
            or frame_idx % self.goal_detect_every_n_frames == 0
        )
        if not should_refresh:
            return None

        goal = self.goal_detector.detect(frame)
        self._update_stable_goal(goal)
        return goal

    def _update_stable_goal(self, goal) -> None:
        if goal is None:
            return

        self._goal_detection_count += 1
        if self._goal_detection_count < self._GOAL_STABLE_AFTER:
            return

        if self._stable_goal_bbox is None:
            self._stable_goal_bbox = goal.bbox_xyxy
            return

        alpha = 0.2
        gx1, gy1, gx2, gy2 = goal.bbox_xyxy
        sx1, sy1, sx2, sy2 = self._stable_goal_bbox
        self._stable_goal_bbox = (
            int((1 - alpha) * sx1 + alpha * gx1),
            int((1 - alpha) * sy1 + alpha * gy1),
            int((1 - alpha) * sx2 + alpha * gx2),
            int((1 - alpha) * sy2 + alpha * gy2),
        )

    def _make_goal_from_stable(self):
        if self._stable_goal_bbox is None:
            return None

        class StableGoal:
            def __init__(self, bbox):
                self.bbox_xyxy = bbox
                self.confidence = 1.0
                self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        return StableGoal(self._stable_goal_bbox)

    def _update_shot_state(self, ball, frame_idx: int) -> None:
        if ball is None:
            return
        if getattr(ball, "predicted", False):
            return

        self._ball_trajectory.append((int(ball.center[0]), int(ball.center[1])))
        if self._shot_detected or len(self._ball_trajectory) < 4:
            return

        p0 = self._ball_trajectory[-4]
        p1 = self._ball_trajectory[-1]
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        if dist > 18.0:
            self._shot_detected = True
            self._shot_frame_idx = frame_idx
            self.ball_detector.set_shot_detected(True)

    def _infer_shot_from_ball_loss(self, ball, frame_idx: int) -> None:
        if self._shot_detected:
            return
        if ball is not None and not getattr(ball, "predicted", False):
            return
        if self.ball_detector.last_detection is None:
            return
        if self.ball_detector.missed_frames < 3:
            return

        self._shot_detected = True
        self._shot_frame_idx = frame_idx
        self.ball_detector.set_shot_detected(True)

    def _draw_annotations(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        annotated = frame.copy()

        if self.last_goal is not None:
            annotated = self.goal_detector.mask_outside_goal_area(
                annotated,
                self.last_goal,
                side_padding=ModelConfig.PLAYERS_MASK_SIDE_PADDING,
                top_padding=ModelConfig.PLAYERS_MASK_TOP_PADDING,
                blur_kernel=ModelConfig.MASK_BLUR_KERNEL,
            )
            ball_zone = None
            if self.last_ball is not None:
                ball_zone = self.goal_detector.get_ball_zone(
                    self.last_goal, self.last_ball.center
                )
            annotated = self.goal_detector.annotate_zones(
                annotated, self.last_goal, ball_zone
            )
            x1, y1, x2, y2 = self.last_goal.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["goal"], 2)

        self._draw_player(annotated, self.last_shooter, "shooter")
        self._draw_player(annotated, self.last_goalkeeper, "goalkeeper")

        if self.last_ball is not None:
            x1, y1, x2, y2 = self.last_ball.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["ball"], 2)
            cv2.putText(
                annotated,
                f"ball {self.last_ball.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLORS["ball"],
                2,
                cv2.LINE_AA,
            )

        draw_trajectory(annotated, self._ball_trajectory, COLORS["ball"], max_points=30)
        self._draw_hud(annotated, frame_idx)
        return annotated

    def _draw_player(
        self,
        frame: np.ndarray,
        player: PlayerDetection | None,
        role: str,
    ) -> None:
        if player is None:
            return

        is_ghost = player.confidence < 0.20
        color = COLORS["goalkeeper_ghost"] if is_ghost and role == "goalkeeper" else COLORS[role]
        label = f"{role} {player.confidence:.2f}"
        if is_ghost:
            label = f"{role} ghost {player.confidence:.2f}"

        x1, y1, x2, y2 = player.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if is_ghost else 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if player.track_id is not None:
            cv2.putText(
                frame,
                f"id {player.track_id}",
                (x1, min(frame.shape[0] - 6, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        draw_pose(frame, player.pose, color)

    def _draw_hud(self, frame: np.ndarray, frame_idx: int) -> None:
        y = 26
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.role_assigner.roles_frozen:
            state = "roles locked"
        elif self._shot_detected:
            state = "shot detected"
        else:
            state = "pre-shot"

        lines = [f"Frame: {frame_idx} | {state}"]
        metrics = self.last_metrics
        if metrics is not None:
            if metrics.shooter_shoulder_angle is not None:
                lines.append(f"Shooter shoulders: {metrics.shooter_shoulder_angle:.1f}")
            if metrics.shooter_body_angle is not None:
                lines.append(f"Shooter body: {metrics.shooter_body_angle:.1f}")
            if metrics.goalkeeper_shoulder_angle is not None:
                lines.append(f"GK shoulders: {metrics.goalkeeper_shoulder_angle:.1f}")
            if metrics.goalkeeper_body_angle is not None:
                lines.append(f"GK body: {metrics.goalkeeper_body_angle:.1f}")
            if metrics.goalkeeper_reaction_time_ms is not None:
                lines.append(f"GK reaction: {metrics.goalkeeper_reaction_time_ms:.0f} ms")

        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                font,
                0.55,
                COLORS["text"],
                2,
                cv2.LINE_AA,
            )
            y += 24
