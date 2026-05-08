"""Simple penalty video pipeline.

The pipeline owns cadence. Detectors only do their model call when asked:
goal refresh, player pose+BoT-SORT refresh, ball refresh/reacquisition. Every
other frame reuses the last known state so the output video still has 300
annotated frames without running every model 300 times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detectors.ball_detector import BallDetection, BallDetector
from .detectors.goal_detector import GoalDetection, GoalDetector
from .detectors.inference_mask import make_inference_frame
from .detectors.players_detector import PlayerDetection
from .detectors.pose_players_detector import PosePlayersDetector
from .models import ModelConfig
from .tracking.role_assignment import RoleAssigner
from .video_io import VideoReader, VideoWriter
from utils.drawings import draw_pose, draw_trajectory


COLORS = {
    "shooter": (0, 200, 0),
    "goalkeeper": (0, 0, 255),
    "ball": (0, 255, 255),
    "goal": (255, 0, 0),
    "text": (255, 255, 255),
}


class PenaltyPipeline:
    """Controller for detection cadence, tracking reuse and annotations."""

    def __init__(
        self,
        ball_model: Optional[str] = None,
        players_model: Optional[str] = None,
        ball_confidence: Optional[float] = None,
        players_confidence: Optional[float] = None,
        process_every_n_frames: Optional[int] = None,
        goal_detect_every_n_frames: Optional[int] = None,
    ):
        self.ball_detector = BallDetector(
            ball_model or ModelConfig.get_ball_model_path(),
            ball_confidence if ball_confidence is not None else ModelConfig.BALL_ACCEPT_CONFIDENCE,
        )
        self.players_detector = PosePlayersDetector(
            players_model or ModelConfig.get_players_model_path(),
            players_confidence
            if players_confidence is not None
            else ModelConfig.PLAYERS_CONFIDENCE,
        )
        self.goal_detector = GoalDetector()
        self.role_assigner = RoleAssigner()

        self.process_every_n_frames = max(
            1, process_every_n_frames or ModelConfig.PROCESS_EVERY_N_FRAMES
        )
        self.goal_detect_every_n_frames = max(
            1, goal_detect_every_n_frames or ModelConfig.GOAL_DETECT_EVERY_N_FRAMES
        )

        self.last_goal: GoalDetection | None = None
        self.last_players: list[PlayerDetection] = []
        self.last_shooter: PlayerDetection | None = None
        self.last_goalkeeper: PlayerDetection | None = None
        self.last_ball: BallDetection | None = None

        self._last_goal_frame = -10_000
        self._last_players_frame = -10_000
        self._last_ball_frame = -10_000
        self._shot_detected = False
        self._ball_trajectory: list[tuple[int, int]] = []
        self._real_ball_trajectory: list[tuple[int, int]] = []
        self._force_ball_detect_until = -1
        self._shot_frame_idx: int | None = None

    def process_video(
        self,
        input_video: str | Path,
        output_video: Optional[str | Path] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
    ) -> Path:
        reader = VideoReader(input_video)
        if output_video is None:
            output_video = Path("data/cv_output") / f"annotated_{Path(input_video).stem}.mp4"

        writer = VideoWriter(output_video, reader.fps, reader.width, reader.height)
        processed = 0
        try:
            for frame_idx, frame in reader:
                self._process_frame(frame, frame_idx)
                annotated = self._draw_annotations(frame, frame_idx)
                writer.write(annotated)

                if show_preview:
                    cv2.imshow("Penalty Analysis", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            reader.release()
            writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"Processed {processed} frames")
        print(f"Output saved to: {output_video}")
        return Path(output_video)

    def _process_frame(self, frame, frame_idx: int) -> None:
        goal = self._refresh_goal(frame, frame_idx)
        player_analysis_frame = make_inference_frame(frame, goal, tight=False)
        ball_analysis_frame = make_inference_frame(frame, goal, tight=True)
        if goal is not None:
            self.ball_detector.set_goal_bbox(goal.bbox_xyxy)

        players = self._refresh_players(player_analysis_frame, frame_idx, goal)
        if players is not None:
            self.last_players = players
            roles = self.role_assigner.assign(
                players,
                goal,
                None,
                self._shot_detected,
                frame_shape=frame.shape,
            )
            self.last_shooter = roles.shooter
            self.last_goalkeeper = roles.goalkeeper
            self.ball_detector.set_shooter_bbox(
                self.last_shooter.bbox_xyxy if self.last_shooter is not None else None
            )

        self.ball_detector.set_shot_detected(self._shot_detected)
        self.ball_detector.set_shooter_bbox(
            self.last_shooter.bbox_xyxy if self.last_shooter is not None else None
        )

        ball = self._refresh_ball(ball_analysis_frame, frame_idx)
        if ball is not None:
            self.last_ball = ball
            if not ball.predicted:
                self._ball_trajectory.append((int(ball.center[0]), int(ball.center[1])))
                shot_started = self._update_shot_state(ball, frame_idx)
                if shot_started:
                    self.role_assigner.lock_current_roles()
                    if ModelConfig.BALL_FORCE_DETECT_FRAMES_AFTER_SHOT > 0:
                        self._force_ball_detect_until = (
                            frame_idx + ModelConfig.BALL_FORCE_DETECT_FRAMES_AFTER_SHOT
                        )
        else:
            self.last_ball = None

    def _refresh_goal(self, frame, frame_idx: int) -> GoalDetection | None:
        if self.last_goal is None or self._due(frame_idx, self._last_goal_frame, self.goal_detect_every_n_frames):
            detected = self.goal_detector.detect(frame)
            self._last_goal_frame = frame_idx
            if detected is not None:
                self.last_goal = detected
                self.ball_detector.set_goal_bbox(detected.bbox_xyxy)
        return self.last_goal

    def _refresh_players(
        self,
        analysis_frame,
        frame_idx: int,
        goal: GoalDetection | None,
    ) -> list[PlayerDetection] | None:
        missing_roles = self.last_shooter is None or self.last_goalkeeper is None
        reacquire_cadence = max(1, ModelConfig.PLAYERS_TRACK_EVERY_N_FRAMES // 2)
        should_reacquire = missing_roles and self._due(
            frame_idx, self._last_players_frame, reacquire_cadence
        )
        should_refresh = self._due(
            frame_idx, self._last_players_frame, ModelConfig.PLAYERS_TRACK_EVERY_N_FRAMES
        )
        if should_reacquire or should_refresh:
            self._last_players_frame = frame_idx
            return self.players_detector.track(analysis_frame)
        return None

    def _refresh_ball(self, analysis_frame, frame_idx: int) -> BallDetection | None:
        cadence = (
            ModelConfig.BALL_DETECT_EVERY_N_FRAMES_POST_SHOT
            if self._shot_detected
            else ModelConfig.BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT
        )
        needs_reacquire = self.ball_detector.missed_frames > 0
        should_reacquire = needs_reacquire and self._due(
            frame_idx,
            self._last_ball_frame,
            ModelConfig.BALL_REACQUIRE_EVERY_N_FRAMES,
            urgent=self.ball_detector.missed_frames > 2,
        )
        detect_now = (
            self.ball_detector.last_detection is None
            or frame_idx <= self._force_ball_detect_until
            or should_reacquire
            or self._due(frame_idx, self._last_ball_frame, cadence)
        )
        if detect_now:
            self._last_ball_frame = frame_idx
            return self.ball_detector.detect(analysis_frame)
        held = self.ball_detector.hold_last()
        if held is None or held.predicted:
            self.ball_detector.increment_miss()
        return held

    def _update_shot_state(self, ball: BallDetection, frame_idx: int) -> bool:
        self._real_ball_trajectory.append((int(ball.center[0]), int(ball.center[1])))
        if self._shot_detected or len(self._real_ball_trajectory) < 4:
            return False

        x0, y0 = self._real_ball_trajectory[-4]
        x1, y1 = self._real_ball_trajectory[-1]
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if dist > ModelConfig.SHOT_VELOCITY_THRESHOLD:
            self._shot_detected = True
            self._shot_frame_idx = frame_idx
            self.ball_detector.set_shot_detected(True)
            return True
        return False

    def _due(self, frame_idx: int, last_frame: int, cadence: int, urgent: bool = False) -> bool:
        cadence_ok = frame_idx - last_frame >= max(1, cadence)
        if urgent:
            return cadence_ok
        return frame_idx % self.process_every_n_frames == 0 and cadence_ok

    def _draw_annotations(self, frame, frame_idx: int):
        annotated = (
            self.goal_detector.mask_outside_goal_area(frame, self.last_goal)
            if self.last_goal is not None
            else frame.copy()
        )

        if self.last_goal is not None:
            x1, y1, x2, y2 = self.last_goal.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["goal"], 4)
            ball_zone = (
                self.goal_detector.get_ball_zone(self.last_goal, self.last_ball.center)
                if self.last_ball is not None
                else None
            )
            self.goal_detector.annotate_zones(annotated, self.last_goal, ball_zone)

        self._draw_player(annotated, self.last_shooter, "shooter")
        self._draw_player(annotated, self.last_goalkeeper, "goalkeeper")

        if self.last_ball is not None:
            self._draw_box(
                annotated,
                self.last_ball.bbox_xyxy,
                COLORS["ball"],
                f"ball {self.last_ball.confidence:.2f}",
            )

        draw_trajectory(annotated, self._ball_trajectory, COLORS["ball"], max_points=30)
        self._draw_hud(annotated, frame_idx)
        return annotated

    def _draw_player(self, frame, player: PlayerDetection | None, role: str) -> None:
        if player is None:
            return
        label = f"{role} {player.confidence:.2f}"
        if player.track_id is not None:
            label += f" id:{player.track_id}"
        self._draw_box(frame, player.bbox_xyxy, COLORS[role], label)
        draw_pose(frame, player.pose, COLORS[role])

    @staticmethod
    def _draw_box(frame, bbox: tuple[int, int, int, int], color, label: str) -> None:
        h, w = frame.shape[:2]
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError, OverflowError):
            return
        if not all(np.isfinite([x1, y1, x2, y2])):
            return
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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

    def _draw_hud(self, frame, frame_idx: int) -> None:
        state = "shot" if self._shot_detected else "pre-shot"
        y = 26
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top-line summary
        cv2.putText(
            frame,
            f"Frame {frame_idx} | {state}",
            (10, y),
            font,
            0.55,
            COLORS["text"],
            2,
            cv2.LINE_AA,
        )
        y += 22

        # Ball diagnostics
        try:
            bd = self.last_ball
            if bd is not None:
                conf = bd.confidence
                pred = getattr(bd, "predicted", False)
                cx, cy = int(bd.center[0]), int(bd.center[1])
                cv2.putText(frame, f"ball: conf={conf:.2f} pred={pred} @{cx},{cy}", (10, y), font, 0.5, COLORS["ball"], 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "ball: none", (10, y), font, 0.5, COLORS["ball"], 1, cv2.LINE_AA)
        except Exception:
            cv2.putText(frame, "ball: err", (10, y), font, 0.5, COLORS["ball"], 1, cv2.LINE_AA)
        y += 18

        # Detector internal state
        try:
            missed = getattr(self.ball_detector, "missed_frames", "-")
            vel = getattr(self.ball_detector, "velocity", None) or (0.0, 0.0)
            lv = getattr(self.ball_detector, "last_real_detection", None)
            lv_c = f"{int(lv.center[0])},{int(lv.center[1])}" if lv is not None else "-"
            cv2.putText(frame, f"missed={missed} vel={vel[0]:.1f},{vel[1]:.1f} last_real={lv_c}", (10, y), font, 0.45, COLORS["ball"], 1, cv2.LINE_AA)
        except Exception:
            cv2.putText(frame, "missed/vel: err", (10, y), font, 0.45, COLORS["ball"], 1, cv2.LINE_AA)
        y += 18

        phase = self._ball_phase()
        reject = getattr(self.ball_detector, "last_reject_reason", None) or "-"
        cv2.putText(
            frame,
            f"ball_phase={phase} reject={reject}",
            (10, y),
            font,
            0.45,
            COLORS["ball"],
            1,
            cv2.LINE_AA,
        )
        y += 18

        cv2.putText(
            frame,
            f"ball model:{ModelConfig.BALL_MODEL} tracker:{ModelConfig.BALL_TRACKER}",
            (10, y),
            font,
            0.42,
            COLORS["text"],
            1,
            cv2.LINE_AA,
        )

    def _ball_phase(self) -> str:
        if self.last_ball is None:
            return "lost"
        return "post_shot" if self._shot_detected else "pre_shot"
