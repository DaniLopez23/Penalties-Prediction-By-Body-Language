"""Main penalty kick analysis pipeline — minimal version (ball + players only)."""
 
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
 
from .video_io import VideoReader, VideoWriter
from .detectors.ball_detector import BallDetector
from .detectors.players_detector import PlayersDetector, PlayerDetection
from .detectors.goal_detector import GoalDetector
from .models import ModelConfig
 
 
COLORS = {
    "shooter":    (0, 200, 0),      # green
    "goalkeeper": (0, 0, 255),      # blue
    "ball":       (0, 255, 255),    # cyan
    "goal":       (255, 255, 0),    # yellow
    "text":       (255, 255, 255),  # white
}
 
 
class PenaltyPipeline:
    """
    Minimal pipeline: ball detector + players detector only.
    No pose, no metrics, no drawings module.
    Use this to validate tracking quality before adding further modules.
    """
 
    def __init__(
        self,
        ball_model: Optional[str] = None,
        players_model: Optional[str] = None,
        ball_confidence: Optional[float] = None,
        players_confidence: Optional[float] = None,
        process_every_n_frames: Optional[int] = None,
    ):
        ball_model        = ball_model        or ModelConfig.get_ball_model_path()
        players_model     = players_model     or ModelConfig.get_players_model_path()
        ball_confidence   = ball_confidence   if ball_confidence   is not None else ModelConfig.BALL_CONFIDENCE
        players_confidence= players_confidence if players_confidence is not None else ModelConfig.PLAYERS_CONFIDENCE
        process_every_n_frames = process_every_n_frames or ModelConfig.PROCESS_EVERY_N_FRAMES
 
        self.ball_detector    = BallDetector(ball_model, ball_confidence)
        self.players_detector = PlayersDetector(players_model, players_confidence)
        self.goal_detector    = GoalDetector()
        self.process_every_n_frames = process_every_n_frames
 
        self.last_ball       = None
        self.last_goal       = None
        self.last_shooter    = None
        self.last_goalkeeper = None
 
        self._last_role_map:  dict[int, str]                       = {}
        self._stable_goal_bbox: Optional[tuple[int, int, int, int]] = None
        self._goal_detection_count: int                             = 0
        self._GOAL_STABLE_AFTER: int                                = 4
 
        # Shot state — passed to ball detector to tighten re-acquisition post-kick
        self._shot_detected: bool = False
        self._ball_trajectory: list[tuple[int, int]] = []
 
        # Role lock — once the shot is detected, roles are frozen so the
        # shooter entering the goal area cannot flip the GK/shooter labels
        self._frozen_role_map: dict[int, str] = {}   # track_id -> role, locked at kick
        self._roles_frozen: bool = False
 
    # ── Main loop ─────────────────────────────────────────────────────────────
 
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
 
        for frame_idx, frame in video_reader:
            should_process = (frame_idx % self.process_every_n_frames) == 0
 
            if should_process:
                # ── Goal (stable reference, not drawn as focus) ──────────────
                goal = self.goal_detector.detect(frame)
                self._update_stable_goal(goal)
                self.ball_detector.set_goal_bbox(self._stable_goal_bbox)
 
                effective_goal = goal if goal is not None else self._make_goal_from_stable()
                analysis_frame = self.goal_detector.mask_outside_goal_area(frame, effective_goal)
 
                # ── Ball ─────────────────────────────────────────────────────
                ball = self.ball_detector.detect(analysis_frame)
 
                # Detect shot and notify ball detector
                if ball is not None:
                    self._ball_trajectory.append((int(ball.center[0]), int(ball.center[1])))
                    if not self._shot_detected and len(self._ball_trajectory) >= 4:
                        p0 = self._ball_trajectory[-4]
                        p1 = self._ball_trajectory[-1]
                        dist = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) ** 0.5
                        if dist > 18.0:
                            self._shot_detected = True
                            self.ball_detector.set_shot_detected(True)
 
                # ── Players ───────────────────────────────────────────────────
                players = self.players_detector.detect(analysis_frame)
                shooter, goalkeeper = self._identify_roles_by_track_id(
                    players, effective_goal, self._stable_goal_bbox
                )
 
                self.last_ball       = ball
                self.last_goal       = effective_goal
                self.last_shooter    = shooter
                self.last_goalkeeper = goalkeeper
 
            annotated = self._draw_annotations(frame, frame_idx)
            video_writer.write(annotated)
 
            if show_preview:
                cv2.imshow("Penalty Analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
 
        video_reader.release()
        video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
 
        print(f"Processed {frame_count} frames")
        print(f"Output saved to: {output_video}")
        return Path(output_video)
 
    # ── Goal stabilization ────────────────────────────────────────────────────
 
    def _update_stable_goal(self, goal) -> None:
        if goal is None:
            return
        self._goal_detection_count += 1
        if self._goal_detection_count >= self._GOAL_STABLE_AFTER:
            if self._stable_goal_bbox is None:
                self._stable_goal_bbox = goal.bbox_xyxy
            else:
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
        class _StableGoal:
            def __init__(self, bbox):
                self.bbox_xyxy = bbox
        return _StableGoal(self._stable_goal_bbox)
 
    # ── Role assignment with ghost detection + post-shot role lock ──────────────
 
    def _identify_roles_by_track_id(
        self,
        players: list[PlayerDetection],
        goal,
        stable_goal_bbox: Optional[tuple] = None,
    ) -> tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """
        Assign shooter / goalkeeper roles, then apply ghost detection.
 
        POST-SHOT ROLE LOCK:
        Once _shot_detected is True, we freeze the role map so that the
        shooter entering the goal area cannot flip the GK/shooter labels.
        After the lock:
          - Track IDs are only looked up in _frozen_role_map.
          - Spatial heuristic is disabled — no new role assignments allowed.
          - Ghost detection still runs normally for occlusion bridging.
 
        CROSSBAR FILTER:
        Any detection whose bbox top (y1) is within CROSSBAR_MARGIN px of the
        goal crossbar is rejected — that region is the crowd/stands, not the
        diving goalkeeper. Applies only to the goal-area region.
        """
        # ── Crossbar + lateral filter: reject crowd and out-of-goal spectators ──
        # Two rejection rules when a detection is near or above the crossbar:
        #   1. bbox top closer than CROSSBAR_MARGIN to the crossbar → upper stand crowd
        #   2. center X outside goal width + LATERAL_MARGIN → corner spectators/refs
        # Detections whose center Y is below the goal (on the field) are always kept.
        CROSSBAR_MARGIN = 40   # px above crossbar = definitely crowd
        LATERAL_MARGIN  = 60   # px outside goal left/right edges = out-of-play
        if stable_goal_bbox is not None and players:
            gx1, gy1, gx2, gy2 = stable_goal_bbox
            crossbar_y = gy1
 
            def _in_play_zone(p) -> bool:
                # Always keep if center is clearly below the goal (field area)
                if p.center[1] > gy2:
                    return True
                # Keep if bbox top is well below the crossbar
                if p.bbox_xyxy[1] >= crossbar_y + CROSSBAR_MARGIN:
                    return True
                # Near/above crossbar: only keep if within goal X range
                if gx1 - LATERAL_MARGIN <= p.center[0] <= gx2 + LATERAL_MARGIN:
                    return True
                # Outside goal laterally AND near crossbar → spectator/ref
                return False
 
            players = [p for p in players if _in_play_zone(p)]
 
        current_track_ids = {p.track_id: p for p in players if p.track_id is not None}
 
        # ── Freeze roles at the moment of kick ────────────────────────────────
        if self._shot_detected and not self._roles_frozen and self._last_role_map:
            self._frozen_role_map = dict(self._last_role_map)
            self._roles_frozen = True
 
        if not players:
            shooter    = self.players_detector.update_ghost(
                "shooter",    None, post_shot=self._shot_detected
            )
            goalkeeper = self.players_detector.update_ghost(
                "goalkeeper", None, post_shot=self._shot_detected
            )
            return shooter, goalkeeper
 
        confirmed_shooter    = None
        confirmed_goalkeeper = None
 
        # ── Use frozen map post-shot, live map pre-shot ───────────────────────
        role_map = self._frozen_role_map if self._roles_frozen else self._last_role_map
 
        for track_id, role in role_map.items():
            if track_id in current_track_ids:
                p = current_track_ids[track_id]
                if role == "shooter":
                    confirmed_shooter = p
                elif role == "goalkeeper":
                    confirmed_goalkeeper = p
 
        # ── Spatial heuristic — only allowed PRE-shot ─────────────────────────
        if not self._roles_frozen:
            available = [
                p for p in players
                if p != confirmed_shooter and p != confirmed_goalkeeper
            ]
 
            if available:
                goal_center = None
                if goal is not None:
                    gx1, gy1, gx2, gy2 = goal.bbox_xyxy
                    goal_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
 
                if confirmed_goalkeeper is None:
                    if goal_center is not None:
                        confirmed_goalkeeper = min(
                            available,
                            key=lambda p: self._distance(p.center, goal_center)
                                + max(0.0, p.center[1] - goal_center[1]) * 0.6,
                        )
                    else:
                        confirmed_goalkeeper = min(available, key=lambda p: p.center[1])
                    available = [p for p in available if p != confirmed_goalkeeper]
 
                if confirmed_shooter is None and available:
                    confirmed_shooter = max(available, key=lambda p: p.center[1])
 
            # Update live role map only pre-shot
            self._last_role_map = {}
            if confirmed_shooter    is not None and confirmed_shooter.track_id    is not None:
                self._last_role_map[confirmed_shooter.track_id]    = "shooter"
            if confirmed_goalkeeper is not None and confirmed_goalkeeper.track_id is not None:
                self._last_role_map[confirmed_goalkeeper.track_id] = "goalkeeper"
 
        # ── Ghost detection bridges occlusion gaps ─────────────────────────────
        # post_shot=True keeps GK ghost indefinitely after the kick
        shooter    = self.players_detector.update_ghost(
            "shooter",    confirmed_shooter,    post_shot=self._shot_detected
        )
        goalkeeper = self.players_detector.update_ghost(
            "goalkeeper", confirmed_goalkeeper, post_shot=self._shot_detected
        )
        return shooter, goalkeeper
 
    # ── Utilities ─────────────────────────────────────────────────────────────
 
    def _distance(self, a: tuple, b: tuple) -> float:
        return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)
 
    # ── Annotations (self-contained, no drawings module needed) ───────────────
 
    def _draw_annotations(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        annotated = frame.copy()
 
        # Goal mask + 3x3 grid
        if self.last_goal is not None:
            annotated = self.goal_detector.mask_outside_goal_area(annotated, self.last_goal)
            ball_zone = None
            if self.last_ball is not None:
                ball_zone = self.goal_detector.get_ball_zone(self.last_goal, self.last_ball.center)
            annotated = self.goal_detector.annotate_zones(annotated, self.last_goal, ball_zone)
            # Goal bbox outline
            x1, y1, x2, y2 = self.last_goal.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["goal"], 2)
 
        # Shooter bbox
        if self.last_shooter is not None:
            x1, y1, x2, y2 = self.last_shooter.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["shooter"], 2)
            cv2.putText(annotated, f"shooter {self.last_shooter.confidence:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, COLORS["shooter"], 2, cv2.LINE_AA)
 
        # Goalkeeper bbox — color changes to yellow when ghost is active
        if self.last_goalkeeper is not None:
            is_ghost = self.last_goalkeeper.confidence < 0.20
            color = (0, 200, 255) if is_ghost else COLORS["goalkeeper"]
            label = f"GK [ghost] {self.last_goalkeeper.confidence:.2f}" if is_ghost \
                    else f"goalkeeper {self.last_goalkeeper.confidence:.2f}"
            x1, y1, x2, y2 = self.last_goalkeeper.bbox_xyxy
            thickness = 1 if is_ghost else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated, label,
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
 
        # Ball bbox + trajectory
        if self.last_ball is not None:
            x1, y1, x2, y2 = self.last_ball.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["ball"], 2)
            cv2.putText(annotated, f"ball {self.last_ball.confidence:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, COLORS["ball"], 2, cv2.LINE_AA)
 
        # Ball trajectory trail
        trail = self._ball_trajectory[-30:]
        for i in range(1, len(trail)):
            cv2.line(annotated, trail[i-1], trail[i], COLORS["ball"], 2)
 
        # HUD — frame counter + shot state + role lock indicator
        h = annotated.shape[0]
        if self._roles_frozen:
            shot_label = "ROLES LOCKED"
        elif self._shot_detected:
            shot_label = "SHOT DETECTED"
        else:
            shot_label = "pre-shot"
        cv2.putText(annotated, f"Frame: {frame_idx}  |  {shot_label}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLORS["text"], 1, cv2.LINE_AA)
 
        # Ghost frame counter top-right
        gk_state = self.players_detector._ghost.get("goalkeeper", {})
        missed = gk_state.get("missed", 0)
        if missed > 0:
            cv2.putText(annotated, f"GK ghost: {missed}f",
                        (annotated.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
 
        return annotated