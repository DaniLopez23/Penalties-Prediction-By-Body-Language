"""Main penalty kick analysis pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from utils.drawings import draw_metrics_text, draw_pose, draw_role_detections, draw_trajectory

from .video_io import VideoReader, VideoWriter
from .detectors.ball_detector import BallDetector
from .detectors.players_detector import PlayersDetector, PlayerDetection
from .detectors.goal_detector import GoalDetector
from .pose.pose_estimator import PoseEstimator
from .penalty_metrics import MetricsCalculator, PenaltyMetrics


class PenaltyPipeline:
    """Simple penalty kick analysis pipeline.
    
    Processes video frames to detect players, ball, and goal, estimate poses,
    and calculate penalty kick metrics.
    """
    
    def __init__(
        self,
        ball_model: str = "yolov8s.pt",
        players_model: Optional[str] = None,
        pose_model: str = "yolov8s-pose.pt",
        ball_confidence: float = 0.2,
        players_confidence: float = 0.25,
        pose_confidence: float = 0.25,
        process_every_n_frames: int = 2
    ):
        """Initialize penalty pipeline with detectors and estimators.
        
        Args:
            ball_model: Path to YOLO model for ball detection.
            players_model: Path to YOLO model for player detection. Defaults to ball_model.
            pose_model: Path to YOLO pose model.
            ball_confidence: Confidence threshold for ball detector.
            players_confidence: Confidence threshold for players detector.
            pose_confidence: Confidence threshold for pose estimator.
            process_every_n_frames: Skip frames for efficiency (process 1 per N frames).
        """
        self.ball_detector = BallDetector(ball_model, ball_confidence)
        self.players_detector = PlayersDetector(players_model or ball_model, players_confidence)
        self.goal_detector = GoalDetector()
        self.pose_estimator = PoseEstimator(pose_model, pose_confidence)
        self.metrics_calc = MetricsCalculator()
        self.process_every_n_frames = process_every_n_frames
        
        self.last_metrics: Optional[PenaltyMetrics] = None
        self.last_ball = None
        self.last_goal = None
        self.last_shooter = None
        self.last_goalkeeper = None
        self.last_shooter_pose = None
        self.last_goalkeeper_pose = None
        self.role_track_max_dist = 160.0
        self.role_hold_frames = 6
        self.shooter_missing_frames = 0
        self.goalkeeper_missing_frames = 0
        # Hysteresis / confirmation to avoid rapid role switching
        self.pending_shooter: Optional[PlayerDetection] = None
        self.pending_goalkeeper: Optional[PlayerDetection] = None
        self.shooter_confirm = 0
        self.goalkeeper_confirm = 0
        self.confirm_threshold = 2
    
    def process_video(
        self,
        input_video: str | Path,
        output_video: Optional[str | Path] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None
    ) -> Path:
        """Process video and save annotated output.
        
        Args:
            input_video: Path to input video file.
            output_video: Path to output annotated video. If None, saves to data/cv_output/.
            show_preview: Display preview window during processing.
            max_frames: Maximum frames to process (for testing). None = process all.
            
        Returns:
            Path to output video file.
            
        Raises:
            FileNotFoundError: If input video does not exist.
            RuntimeError: If video cannot be opened.
        """
        video_reader = VideoReader(input_video)
        
        if output_video is None:
            output_video = Path("data/cv_output") / f"annotated_{Path(input_video).stem}.mp4"
        
        video_writer = VideoWriter(
            output_video,
            fps=video_reader.fps,
            width=video_reader.width,
            height=video_reader.height
        )
        
        frame_count = 0
        
        for frame_idx, frame in video_reader:
            # Always start with a copy of the frame for display/writing
            annotated = frame.copy()
            
            # Skip frames based on process_every_n_frames
            should_process = (frame_idx % self.process_every_n_frames) == 0
            
            if should_process:
                # Run detections
                ball = self.ball_detector.detect(frame)
                players = self.players_detector.detect(frame)
                goal = self.goal_detector.detect(frame)
                
                # Identify roles (shooter vs goalkeeper)
                shooter, goalkeeper = self._identify_roles(players, goal, frame.shape)
                
                # Estimate poses
                shooter_pose = None
                if shooter:
                    shooter_pose = self.pose_estimator.estimate(frame, shooter.bbox_xyxy)
                
                goalkeeper_pose = None
                if goalkeeper:
                    goalkeeper_pose = self.pose_estimator.estimate(frame, goalkeeper.bbox_xyxy)

                self.last_ball = ball
                self.last_goal = goal
                self.last_shooter = shooter
                self.last_goalkeeper = goalkeeper
                self.last_shooter_pose = shooter_pose
                self.last_goalkeeper_pose = goalkeeper_pose
                
                # Calculate metrics
                self.last_metrics = self.metrics_calc.update(
                    frame_idx=frame_idx,
                    ball_center=ball.center if ball else None,
                    goalkeeper_center=goalkeeper.center if goalkeeper else None,
                    shooter_pose=shooter_pose,
                    goalkeeper_pose=goalkeeper_pose,
                    fps=video_reader.fps
                )
            
                # Draw annotations
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
    

    def _identify_roles(
        self,
        players: list[PlayerDetection],
        goal,
        frame_shape: tuple[int, int, int],
    ):
        """Identify shooter and goalkeeper with simple spatial heuristics + temporal tracking."""
        if not players:
            return (
                self._track_player("shooter", None, []),
                self._track_player("goalkeeper", None, []),
            )

        h, w = frame_shape[:2]
        goal_center = None
        goal_bbox = None
        if goal is not None:
            goal_bbox = goal.bbox_xyxy
            gx1, gy1, gx2, gy2 = goal_bbox
            goal_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)

        goalkeeper_candidate = None
        if goal_center is not None:
            goalkeeper_candidate = min(
                players,
                key=lambda p: self._distance(p.center, goal_center) + max(0.0, p.center[1] - goal_center[1]) * 0.6,
            )
        else:
            # Fallback when goal is missing: goalkeeper is typically higher (farther) in the frame.
            goalkeeper_candidate = min(players, key=lambda p: p.center[1])

        remaining_players = [p for p in players if p is not goalkeeper_candidate]

        shooter_candidate = None
        if remaining_players:
            shooter_candidate = max(
                remaining_players,
                key=lambda p: p.center[1] - 140.0 * self._goal_overlap_ratio(p, goal_bbox),
            )
        elif players:
            # One-player fallback: treat lower player as shooter.
            shooter_candidate = max(players, key=lambda p: p.center[1])

        tracked_goalkeeper = self._track_player("goalkeeper", goalkeeper_candidate, players)
        shooter_pool = [p for p in players if not self._same_detection(p, tracked_goalkeeper)]
        tracked_shooter = self._track_player("shooter", shooter_candidate, shooter_pool)

        # Last safety net in sparse/noisy frames.
        if tracked_shooter is None and shooter_pool:
            tracked_shooter = max(shooter_pool, key=lambda p: p.center[1])
        if tracked_goalkeeper is None and players:
            tracked_goalkeeper = min(players, key=lambda p: p.center[1])

        if tracked_shooter is not None and tracked_goalkeeper is not None and self._same_detection(tracked_shooter, tracked_goalkeeper):
            alternate = [p for p in players if not self._same_detection(p, tracked_goalkeeper)]
            if alternate:
                tracked_shooter = max(alternate, key=lambda p: p.center[1])

        return tracked_shooter, tracked_goalkeeper

    def _track_player(self, role: str, candidate: Optional[PlayerDetection], players: list[PlayerDetection]):
        """Nearest-neighbor temporal tracking for one role (shooter or goalkeeper)."""
        prev = self.last_shooter if role == "shooter" else self.last_goalkeeper
        missing = self.shooter_missing_frames if role == "shooter" else self.goalkeeper_missing_frames

        # If we have a previous and current detections, prefer the one closest to previous.
        if prev is not None and players:
            nearest = min(players, key=lambda p: self._distance(p.center, prev.center))
            d_nearest = self._distance(nearest.center, prev.center)

            # If it's essentially the same (very close), accept immediately.
            if d_nearest < 40.0:
                if role == "shooter":
                    self.shooter_missing_frames = 0
                    self.pending_shooter = None
                    self.shooter_confirm = 0
                else:
                    self.goalkeeper_missing_frames = 0
                    self.pending_goalkeeper = None
                    self.goalkeeper_confirm = 0
                return nearest

            # If within tracking distance but different, require confirmation across frames.
            if d_nearest <= self.role_track_max_dist:
                if role == "shooter":
                    if self.pending_shooter is None or not self._same_detection(self.pending_shooter, nearest):
                        self.pending_shooter = nearest
                        self.shooter_confirm = 1
                    else:
                        self.shooter_confirm += 1

                    if self.shooter_confirm >= self.confirm_threshold:
                        self.pending_shooter = None
                        self.shooter_confirm = 0
                        self.shooter_missing_frames = 0
                        return nearest
                    # wait for confirmation, keep prev for now
                    return prev
                else:
                    if self.pending_goalkeeper is None or not self._same_detection(self.pending_goalkeeper, nearest):
                        self.pending_goalkeeper = nearest
                        self.goalkeeper_confirm = 1
                    else:
                        self.goalkeeper_confirm += 1

                    if self.goalkeeper_confirm >= self.confirm_threshold:
                        self.pending_goalkeeper = None
                        self.goalkeeper_confirm = 0
                        self.goalkeeper_missing_frames = 0
                        return nearest
                    return prev

        # If no prev but we have a candidate from heuristics, accept it.
        if prev is None and candidate is not None:
            if role == "shooter":
                self.shooter_missing_frames = 0
            else:
                self.goalkeeper_missing_frames = 0
            return candidate

        # If previous exists and short disappearance, hold previous
        if prev is not None and missing < self.role_hold_frames:
            if role == "shooter":
                self.shooter_missing_frames += 1
            else:
                self.goalkeeper_missing_frames += 1
            return prev

        # Reset and give up
        if role == "shooter":
            self.shooter_missing_frames = 0
            self.pending_shooter = None
            self.shooter_confirm = 0
        else:
            self.goalkeeper_missing_frames = 0
            self.pending_goalkeeper = None
            self.goalkeeper_confirm = 0
        return None

    def _goal_overlap_ratio(self, player: PlayerDetection, goal_bbox) -> float:
        if goal_bbox is None:
            return 0.0

        px1, py1, px2, py2 = player.bbox_xyxy
        gx1, gy1, gx2, gy2 = goal_bbox
        ix1 = max(px1, gx1)
        iy1 = max(py1, gy1)
        ix2 = min(px2, gx2)
        iy2 = min(py2, gy2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = float((ix2 - ix1) * (iy2 - iy1))
        player_area = max(1.0, float((px2 - px1) * (py2 - py1)))
        return inter_area / player_area

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _same_detection(self, a: Optional[PlayerDetection], b: Optional[PlayerDetection]) -> bool:
        if a is None or b is None:
            return False
        return a.bbox_xyxy == b.bbox_xyxy
    
    def _draw_annotations(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw detections and metrics on frame.
        
        Args:
            frame: Input frame.
            frame_idx: Current frame index.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        draw_role_detections(
            annotated,
            shooter=self.last_shooter,
            goalkeeper=self.last_goalkeeper,
            ball=self.last_ball,
            goal=self.last_goal,
        )

        draw_pose(annotated, self.last_shooter_pose, color=(255, 255, 255))
        draw_pose(annotated, self.last_goalkeeper_pose, color=(255, 120, 120))

        if self.last_metrics is not None:
            draw_trajectory(annotated, self.last_metrics.ball_trajectory, color=(0, 255, 255), max_points=30)

        draw_metrics_text(annotated, self.last_metrics, frame_idx)
        
        return annotated
