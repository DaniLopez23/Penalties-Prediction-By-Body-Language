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
from .models import ModelConfig


class PenaltyPipeline:
    """Simple penalty kick analysis pipeline.

    Processes video frames to detect players, ball, and goal, estimate poses,
    and calculate penalty kick metrics.
    """

    def __init__(
        self,
        ball_model: Optional[str] = None,
        players_model: Optional[str] = None,
        pose_model: Optional[str] = None,
        ball_confidence: Optional[float] = None,
        players_confidence: Optional[float] = None,
        pose_confidence: Optional[float] = None,
        process_every_n_frames: Optional[int] = None,
    ):
        """Initialize penalty pipeline with detectors and estimators.
        
        All parameters are optional. If not provided, default values from ModelConfig are used.
        Models are integrated directly - no need to pass parameters unless using custom models.

        Args:
            ball_model: Path to YOLO model for ball detection (default: ModelConfig.BALL_MODEL).
            players_model: Path to YOLO model for player detection (default: ModelConfig.PLAYERS_MODEL).
            pose_model: Path to YOLO pose model (default: ModelConfig.POSE_MODEL).
            ball_confidence: Confidence threshold for ball detector (default: ModelConfig.BALL_CONFIDENCE).
            players_confidence: Confidence threshold for players detector (default: ModelConfig.PLAYERS_CONFIDENCE).
            pose_confidence: Confidence threshold for pose estimator (default: ModelConfig.POSE_CONFIDENCE).
            process_every_n_frames: Skip frames for efficiency (default: ModelConfig.PROCESS_EVERY_N_FRAMES).
        """
        # Use provided values or fall back to ModelConfig defaults
        ball_model = ball_model or ModelConfig.get_ball_model_path()
        players_model = players_model or ModelConfig.get_players_model_path()
        pose_model = pose_model or ModelConfig.get_pose_model_path()
        ball_confidence = ball_confidence if ball_confidence is not None else ModelConfig.BALL_CONFIDENCE
        players_confidence = players_confidence if players_confidence is not None else ModelConfig.PLAYERS_CONFIDENCE
        pose_confidence = pose_confidence if pose_confidence is not None else ModelConfig.POSE_CONFIDENCE
        process_every_n_frames = process_every_n_frames or ModelConfig.PROCESS_EVERY_N_FRAMES
        
        self.ball_detector = BallDetector(ball_model, ball_confidence)
        self.players_detector = PlayersDetector(players_model, players_confidence)
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
        
        # Track ID mapping for roles (ByteTrack-based instead of manual tracking)
        # Format: {track_id: role} where role is "shooter" or "goalkeeper"
        self._last_role_map: dict[int, str] = {}

        # ── Contexto de portería para el detector de balón ───────────────────
        # La portería no se mueve: una vez detectada con estabilidad se congela
        # para no alimentar bboxes ruidosas al detector de balón.
        self._stable_goal_bbox: Optional[tuple[int, int, int, int]] = None
        self._goal_detection_count: int = 0
        self._GOAL_STABLE_AFTER: int = 4   # frames con detección antes de congelar

    # ─────────────────────────────────────────────────────────────────────────
    # Proceso principal
    # ─────────────────────────────────────────────────────────────────────────

    def process_video(
        self,
        input_video: str | Path,
        output_video: Optional[str | Path] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
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
            height=video_reader.height,
        )

        frame_count = 0

        for frame_idx, frame in video_reader:
            should_process = (frame_idx % self.process_every_n_frames) == 0

            if should_process:
                # ── Detección de portería ────────────────────────────────────
                goal = self.goal_detector.detect(frame)
                self._update_stable_goal(goal)

                # Alimentar el bbox estable de portería al detector de balón
                # para que pueda filtrar falsos positivos laterales.
                self.ball_detector.set_goal_bbox(self._stable_goal_bbox)

                # Aplicar máscara binaria rápida en lugar de blur costoso.
                # Se conserva la zona del arco y lo que queda por debajo,
                # y se oscurecen los laterales y la parte superior para reducir ruido.
                effective_goal = goal if goal is not None else self._make_goal_from_stable()
                analysis_frame = self.goal_detector.mask_outside_goal_area(
                    frame,
                    effective_goal,
                )

                # ── Resto de detecciones ─────────────────────────────────────
                ball = self.ball_detector.detect(analysis_frame)
                players = self.players_detector.detect(analysis_frame)

                # Identify roles using ByteTrack track_ids (simplified logic)
                shooter, goalkeeper = self._identify_roles_by_track_id(players, effective_goal)

                # Estimate poses only on detected players (lazy evaluation)
                shooter_pose = None
                if shooter:
                    shooter_pose = self.pose_estimator.estimate(frame, shooter.bbox_xyxy)

                goalkeeper_pose = None
                if goalkeeper:
                    goalkeeper_pose = self.pose_estimator.estimate(frame, goalkeeper.bbox_xyxy)

                self.last_ball = ball
                self.last_goal = effective_goal
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
                    fps=video_reader.fps,
                )

            # Draw annotations on every frame so skipped frames keep the last
            # available detections instead of alternating between annotated and raw.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Estabilización de portería
    # ─────────────────────────────────────────────────────────────────────────

    def _update_stable_goal(self, goal) -> None:
        """
        Acumula detecciones de portería y congela el bbox cuando hay suficientes
        frames consecutivos con detección. Esto evita pasar bboxes ruidosas o
        ausentes al detector de balón.
        """
        if goal is None:
            # Si ya tenemos un bbox estable, lo conservamos (la portería no se mueve)
            return

        self._goal_detection_count += 1

        if self._goal_detection_count >= self._GOAL_STABLE_AFTER:
            # Actualizar con un suavizado ligero si ya había uno
            if self._stable_goal_bbox is None:
                self._stable_goal_bbox = goal.bbox_xyxy
            else:
                # Promedio ponderado para evitar saltos bruscos en la detección
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
        """Devuelve un objeto mínimo compatible con las funciones de rol a partir del bbox estable."""
        if self._stable_goal_bbox is None:
            return None

        # Creamos un namespace simple para evitar depender de la clase GoalDetection
        class _StableGoal:
            def __init__(self, bbox):
                self.bbox_xyxy = bbox

        return _StableGoal(self._stable_goal_bbox)

    # ─────────────────────────────────────────────────────────────────────────
    # Identificación de roles (Optimized with ByteTrack)
    # ─────────────────────────────────────────────────────────────────────────

    def _identify_roles_by_track_id(
        self,
        players: list[PlayerDetection],
        goal,
    ) -> tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """Identify shooter and goalkeeper using ByteTrack track_ids.
        
        Strategy:
        1. First frame: assign roles based on spatial proximity to goal
        2. Subsequent frames: maintain role assignment using track_ids
        3. If track_id reappears, reuse its previous role
        4. If new track_id appears, use spatial heuristic
        
        Args:
            players: List of detected players with track_ids from ByteTrack.
            goal: GoalDetection object or None.
            
        Returns:
            Tuple of (shooter, goalkeeper) PlayerDetection objects.
        """
        if not players:
            return None, None
        
        # Extract track_ids from current detections
        current_track_ids = {p.track_id: p for p in players if p.track_id is not None}
        
        # Try to match existing roles by track_id
        shooter = None
        goalkeeper = None
        
        for track_id, role in self._last_role_map.items():
            if track_id in current_track_ids:
                player = current_track_ids[track_id]
                if role == "shooter":
                    shooter = player
                elif role == "goalkeeper":
                    goalkeeper = player
        
        # For unmatched players, use spatial heuristic to assign roles
        available_players = [
            p for p in players 
            if p != shooter and p != goalkeeper
        ]
        
        if available_players:
            goal_center = None
            if goal is not None:
                gx1, gy1, gx2, gy2 = goal.bbox_xyxy
                goal_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
            
            # Assign goalkeeper: closest to goal (typically standing in front)
            if goalkeeper is None:
                if goal_center is not None:
                    goalkeeper = min(
                        available_players,
                        key=lambda p: self._distance(p.center, goal_center)
                        + max(0.0, p.center[1] - goal_center[1]) * 0.6,
                    )
                    available_players.remove(goalkeeper)
                else:
                    goalkeeper = min(available_players, key=lambda p: p.center[1])
                    available_players.remove(goalkeeper)
            
            # Assign shooter: farthest from goal (typically running toward it)
            if shooter is None and available_players:
                shooter = max(available_players, key=lambda p: p.center[1])
                available_players.remove(shooter)
            elif shooter is None and available_players:
                shooter = available_players[0]
        
        # Update role map for next frame
        self._last_role_map = {}
        if shooter is not None and shooter.track_id is not None:
            self._last_role_map[shooter.track_id] = "shooter"
        if goalkeeper is not None and goalkeeper.track_id is not None:
            self._last_role_map[goalkeeper.track_id] = "goalkeeper"
        
        return shooter, goalkeeper

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────────

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

    def _same_detection(
        self, a: Optional[PlayerDetection], b: Optional[PlayerDetection]
    ) -> bool:
        if a is None or b is None:
            return False
        return a.bbox_xyxy == b.bbox_xyxy

    # ─────────────────────────────────────────────────────────────────────────
    # Anotaciones
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw detections and metrics on frame."""
        annotated = frame.copy()

        if self.last_goal is not None:
            # Apply fast binary mask for visualization (replaces slow blur)
            annotated = self.goal_detector.mask_outside_goal_area(annotated, self.last_goal)

            x1, y1, x2, _ = self.last_goal.bbox_xyxy
            frame_h, frame_w = annotated.shape[:2]
            keep_x1 = max(0, x1 - 24)
            keep_x2 = min(frame_w - 1, x2 + 24)
            keep_y1 = max(0, y1 - 18)

            cv2.rectangle(
                annotated,
                (keep_x1, keep_y1),
                (keep_x2, frame_h - 1),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                annotated,
                "analysis mask",
                (keep_x1, max(20, keep_y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        draw_role_detections(
            annotated,
            shooter=self.last_shooter,
            goalkeeper=self.last_goalkeeper,
            ball=self.last_ball,
            goal=self.last_goal,
        )

        ball_zone = None

        if self.last_goal is not None and self.last_ball is not None:
            ball_zone = self.goal_detector.get_ball_zone(
                self.last_goal,
                self.last_ball.center,
            )

        annotated = self.goal_detector.annotate_zones(
            annotated,
            self.last_goal,
            ball_zone,
        )
        
        draw_pose(annotated, self.last_shooter_pose, color=(255, 255, 255))
        draw_pose(annotated, self.last_goalkeeper_pose, color=(255, 120, 120))

        if self.last_metrics is not None:
            draw_trajectory(annotated, self.last_metrics.ball_trajectory, color=(0, 255, 255), max_points=30)

        draw_metrics_text(annotated, self.last_metrics, frame_idx)

        return annotated