from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.analysis.pose import PoseAnalyzer
from src.analysis.penalty import PenaltyAnalyzer
from src.config import PipelineConfig
from src.detectors.goal import GoalDetector
from src.detectors.yolo import YOLODetector, YOLOPoseEstimator
from src.models import FrameAnalysisRecord, PenaltyAnalysisState, PoseMetrics
from src.preprocessing.roi import PlayAreaMasker
from src.tracking.ball import BallTracker
from src.tracking.roles import PlayerRoleAssigner
from src.visualization.annotator import FrameAnnotator


ProgressCallback = Callable[[int, Optional[int], PenaltyAnalysisState], None]


class PenaltyAnalysisPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = YOLODetector(config)
        self.pose_estimator = YOLOPoseEstimator(config)
        self.goal_detector = GoalDetector(config)
        self.play_area_masker = PlayAreaMasker(config)
        self.role_assigner = PlayerRoleAssigner(config)
        self.ball_tracker = BallTracker(config)
        self.pose_analyzer = PoseAnalyzer(config.models.pose_keypoint_confidence)
        self.penalty_analyzer = PenaltyAnalyzer()
        self.annotator = FrameAnnotator(config)
        self.analysis_history: list[FrameAnalysisRecord] = []
        self.last_analysis_state: Optional[PenaltyAnalysisState] = None
        self.video_info: dict[str, float | int | str | None] = {}
        self.shot_frame_index: Optional[int] = None
        self.shot_time_sec: Optional[float] = None

    def process_video(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        show_window: Optional[bool] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        self._reset_analysis_state()
        video_cfg = self.config.video
        source = Path(input_path or video_cfg.input_path)
        if not source.exists():
            raise FileNotFoundError(f"Input video not found: {source}")

        frame_limit = max_frames if max_frames is not None else video_cfg.max_frames
        destination = output_path or self._default_output_path(source, frame_limit)
        destination.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or video_cfg.fallback_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = self._total_frames(source_frame_count, frame_limit)
        self.video_info = {
            "source": str(source),
            "fps": float(fps),
            "width": width,
            "height": height,
            "frame_count": source_frame_count,
            "processed_frame_count": total_frames,
            "duration_sec": (source_frame_count / fps) if source_frame_count > 0 and fps else None,
        }
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*video_cfg.output_codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {destination}")

        should_show = video_cfg.show_window if show_window is None else show_window

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_limit is not None and frame_index > frame_limit:
                    break

                annotated, analysis_state, pose_metrics = self._process_frame(frame, frame_index)
                self._record_analysis(frame_index, fps, analysis_state, pose_metrics)
                writer.write(annotated)
                if progress_callback is not None:
                    progress_callback(frame_index, total_frames, analysis_state)

                if should_show:
                    cv2.imshow(video_cfg.window_name, self._display_frame(annotated))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(video_cfg.quit_key):
                        break
        finally:
            capture.release()
            writer.release()
            if should_show:
                cv2.destroyWindow(video_cfg.window_name)

        return destination

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, PenaltyAnalysisState, dict[str, PoseMetrics]]:
        goal = self.goal_detector.detect(frame)
        play_area_mask = self.play_area_masker.build_mask(frame.shape, goal)
        inference_frame = self.play_area_masker.apply_for_detection(frame, play_area_mask)

        detections = self.detector.detect(inference_frame)
        detections = self.play_area_masker.filter_detections(detections, play_area_mask)
        persons = [det for det in detections if det.class_id == self.config.models.coco_person_class_id]
        balls = [det for det in detections if det.class_id == self.config.models.coco_ball_class_id]

        ball = self.ball_tracker.update(
            inference_frame,
            balls,
            persons,
            valid_mask=play_area_mask,
        )
        role_assignments = self.role_assigner.assign(
            persons,
            frame.shape,
            goal,
            ball.center if ball is not None and ball.observed else None,
        )
        poses = self.pose_estimator.detect(inference_frame, frame_index)
        pose_assignments = self.pose_analyzer.assign_poses(poses, role_assignments)
        pose_metrics = {
            role: self.pose_analyzer.metrics(pose)
            for role, pose in pose_assignments.items()
        }
        analysis_state = self.penalty_analyzer.update(goal, ball, pose_metrics)
        visual_frame = self.play_area_masker.apply_blur(frame, play_area_mask)

        annotated = self.annotator.draw(
            visual_frame,
            goal,
            role_assignments,
            pose_assignments,
            pose_metrics,
            ball,
            self.ball_tracker.trail,
            frame_index,
            analysis_state,
        )
        return annotated, analysis_state, pose_metrics

    def _default_output_path(self, source: Path, frame_limit: Optional[int] = None) -> Path:
        video_cfg = self.config.video
        frame_suffix = f"_first{frame_limit}" if frame_limit is not None else ""
        return video_cfg.output_dir / f"{source.stem}{video_cfg.output_suffix}{frame_suffix}.mp4"

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        scale = self.config.video.display_scale
        if abs(scale - 1.0) < 1e-6:
            return frame
        width = max(1, int(round(frame.shape[1] * scale)))
        height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _reset_analysis_state(self) -> None:
        self.role_assigner = PlayerRoleAssigner(self.config)
        self.ball_tracker = BallTracker(self.config)
        self.penalty_analyzer = PenaltyAnalyzer()
        self.pose_estimator.last_poses = []
        self.analysis_history = []
        self.last_analysis_state = None
        self.video_info = {}
        self.shot_frame_index = None
        self.shot_time_sec = None

    @staticmethod
    def _total_frames(source_frame_count: int, frame_limit: Optional[int]) -> Optional[int]:
        if source_frame_count <= 0:
            return frame_limit
        if frame_limit is None:
            return source_frame_count
        return min(source_frame_count, frame_limit)

    def _record_analysis(
        self,
        frame_index: int,
        fps: float,
        analysis_state: PenaltyAnalysisState,
        pose_metrics: dict[str, PoseMetrics],
    ) -> None:
        time_sec = (frame_index - 1) / fps if fps else 0.0
        if self.shot_frame_index is None and analysis_state.shot_state == "shot":
            self.shot_frame_index = frame_index
            self.shot_time_sec = time_sec

        striker_metrics = pose_metrics.get("striker")
        self.analysis_history.append(
            FrameAnalysisRecord(
                frame_index=frame_index,
                time_sec=time_sec,
                shot_state=analysis_state.shot_state,
                ball_zone=analysis_state.ball_zone,
                goalkeeper_direction=analysis_state.goalkeeper_direction,
                striker_shoulder_angle_deg=(
                    striker_metrics.shoulder_angle_deg if striker_metrics else None
                ),
                striker_body_lean_deg=analysis_state.striker_body_lean_deg,
                goalkeeper_lean_deg=analysis_state.goalkeeper_lean_deg,
            )
        )
        self.last_analysis_state = analysis_state
