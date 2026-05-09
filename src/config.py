from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_VIDEO = DATA_DIR / "penalties_mbappe_1.mp4"
DEFAULT_OUTPUT_DIR = DATA_DIR / "output_videos"


@dataclass(frozen=True)
class VideoConfig:
    input_path: Path = DEFAULT_INPUT_VIDEO
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_suffix: str = "_annotated"
    output_codec: str = "mp4v"
    fallback_fps: float = 60.0
    show_window: bool = True
    window_name: str = "Penalty Analysis"
    display_scale: float = 1.0
    quit_key: str = "q"
    max_frames: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    detector_model: str = "yolov8n.pt"
    pose_model: str = "yolov8n-pose.pt"
    tracker: str = "botsort.yaml"
    device: Optional[str] = None
    image_size: int = 960
    detector_confidence: float = 0.08
    person_confidence: float = 0.25
    ball_confidence: float = 0.08
    iou_threshold: float = 0.45
    pose_confidence: float = 0.25
    pose_keypoint_confidence: float = 0.30
    pose_stride: int = 1
    coco_person_class_id: int = 0
    coco_ball_class_id: int = 32


@dataclass(frozen=True)
class GoalConfig:
    white_saturation_max: int = 30
    white_value_min: int = 200
    goal_ratio: float = 3.0
    goal_shrink_x: float = 0.02
    goal_shrink_y: float = 0.01
    goal_bottom_extend_ratio: float = 0.08
    goal_erode_iterations: int = 1
    goal_dilate_iterations: int = 2
    goal_median_blur: int = 5
    min_contour_area_ratio: float = 0.0005
    min_goal_width_ratio: float = 0.20
    max_goal_width_ratio: float = 0.82
    min_goal_height_ratio: float = 0.10
    goal_roi_top_ratio: float = 0.05
    goal_roi_bottom_ratio: float = 0.55
    goal_roi_x_margin_ratio: float = 0.0
    tracking_smoothing: float = 0.35
    max_tracking_shift_ratio: float = 0.08
    max_tracking_size_change_ratio: float = 0.25
    keep_last_for_frames: int = 20
    use_fallback_goal: bool = True
    fallback_x1_ratio: float = 0.29
    fallback_y1_ratio: float = 0.13
    fallback_x2_ratio: float = 0.70
    fallback_y2_ratio: float = 0.38


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool = True
    top_expand_goal_width: float = 0.04
    bottom_width_goal_multiplier: float = 1.2
    bottom_y_ratio: float = 1.0
    detection_fill_value: int = 0
    blur_kernel: int = 45
    show_roi_border: bool = True
    roi_border_color: tuple[int, int, int] = (130, 130, 130)


@dataclass(frozen=True)
class RoleConfig:
    lock_track_ids: bool = True
    lost_reassign_frames: int = 10
    min_assignment_score: float = 0.35
    goalkeeper_goal_margin_ratio: float = 0.18
    striker_bottom_weight: float = 2.2
    striker_center_weight: float = 1.1
    striker_size_weight: float = 0.8
    striker_ball_weight: float = 2.0
    striker_ball_max_distance_ratio: float = 0.30
    goalkeeper_goal_weight: float = 2.5
    goalkeeper_center_weight: float = 1.3
    goalkeeper_depth_weight: float = 1.0
    goalkeeper_lock_goal_margin_ratio: float = 0.55
    goalkeeper_reassign_after_frames: int = 35


@dataclass(frozen=True)
class BallConfig:
    max_missing_frames: int = 18
    min_motion_area_ratio: float = 0.000012
    max_motion_area_ratio: float = 0.0025
    motion_threshold: int = 28
    motion_blur_kernel: int = 5
    contour_circularity_weight: float = 0.25
    yolo_candidate_weight: float = 1.0
    prediction_distance_weight: float = 0.55
    player_overlap_penalty: float = 0.75
    player_foot_overlap_penalty: float = 1.2
    motion_candidate_max_aspect_ratio: float = 2.6
    max_prediction_distance_ratio: float = 0.18
    max_candidate_jump_ratio: float = 0.22
    last_position_weight: float = 0.35
    reject_outside_roi: bool = True
    trail_length: int = 28
    min_ball_radius: int = 3
    max_ball_radius_ratio: float = 0.025


@dataclass(frozen=True)
class DrawConfig:
    goal_color: tuple[int, int, int] = (70, 220, 255)
    goal_zone_color: tuple[int, int, int] = (40, 130, 240)
    striker_color: tuple[int, int, int] = (255, 180, 40)
    goalkeeper_color: tuple[int, int, int] = (70, 255, 120)
    ball_color: tuple[int, int, int] = (35, 90, 255)
    predicted_ball_color: tuple[int, int, int] = (170, 170, 170)
    pose_color: tuple[int, int, int] = (240, 240, 240)
    text_color: tuple[int, int, int] = (255, 255, 255)
    label_bg: tuple[int, int, int] = (25, 25, 25)
    line_thickness: int = 2
    font_scale: float = 0.55


@dataclass(frozen=True)
class PipelineConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)


DEFAULT_CONFIG = PipelineConfig()
