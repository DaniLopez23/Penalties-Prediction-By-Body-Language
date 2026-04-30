"""Main penalty kick analysis pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from utils.drawings import draw_metrics_text, draw_pose, draw_role_detections, draw_trajectory

from .video_io import VideoReader, VideoWriter
from .detectors.ball_detector import BallDetector
from .detectors.players_detector import PlayersDetector
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
            # Skip frames based on process_every_n_frames
            should_process = (frame_idx % self.process_every_n_frames) == 0
            
            if should_process:
                # Run detections
                ball = self.ball_detector.detect(frame)
                players = self.players_detector.detect(frame)
                goal = self.goal_detector.detect(frame)
                
                # Identify roles (shooter vs goalkeeper)
                shooter, goalkeeper = self._identify_roles(players, goal)
                
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
    
    def _identify_roles(self, players, goal):
        """Identify shooter and goalkeeper from detected players.
        
        Simple heuristic: the player closest to goal = goalkeeper.
        Assumes goal_x is available from goal detection.
        
        Args:
            players: List of PlayerDetection objects.
            goal: GoalDetection object or None.
            
        Returns:
            Tuple of (shooter, goalkeeper) PlayerDetection or None.
        """
        if not players:
            return None, None
        
        # Determine goal x-coordinate
        if goal:
            goal_x = (goal.bbox_xyxy[0] + goal.bbox_xyxy[2]) / 2
        else:
            # Fallback: assume goal is at rightmost x
            goal_x = max(p.center[0] for p in players)
        
        # Sort by distance to goal
        sorted_players = sorted(
            players,
            key=lambda p: abs(p.center[0] - goal_x)
        )
        
        if len(sorted_players) >= 2:
            # Closer to goal = goalkeeper
            return sorted_players[1], sorted_players[0]
        elif len(sorted_players) == 1:
            # Only one player detected
            return None, sorted_players[0]
        
        return None, None
    
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
