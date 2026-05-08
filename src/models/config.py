"""Small, central configuration for the penalty video pipeline."""

from __future__ import annotations

from pathlib import Path


class ModelConfig:
    """Defaults for behind-the-shooter penalty videos."""

    PLAYERS_MODEL: str = "yolo11l-pose.pt"
    POSE_MODEL: str = PLAYERS_MODEL
    BALL_MODEL: str = "yolo11s.pt"

    PERSON_CLASS_ID: int = 0
    BALL_CLASS_ID: int = 32

    PLAYERS_CONFIDENCE: float = 0.20
    POSE_CONFIDENCE: float = PLAYERS_CONFIDENCE
    BALL_CONFIDENCE: float = 0.25

    PLAYERS_IMGSZ: int = 1280
    BALL_IMGSZ: int = 1536

    PLAYERS_TRACKER: str = "botsort.yaml"
    PLAYERS_DEVICE: str | None = None
    BALL_DEVICE: str | None = None
    PLAYERS_HALF: bool = False
    BALL_HALF: bool = False
    BALL_TRACKER: str = "bytetrack.yaml"

    # Heavy inference cadence. The output is still written every frame.
    PROCESS_EVERY_N_FRAMES: int = 1
    GOAL_DETECT_EVERY_N_FRAMES: int = 30
    PLAYERS_TRACK_EVERY_N_FRAMES: int = 5
    BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT: int = 2
    BALL_DETECT_EVERY_N_FRAMES_POST_SHOT: int = 1
    BALL_DETECT_EVERY_N_FRAMES: int = BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT
    BALL_REACQUIRE_EVERY_N_FRAMES: int = 2
    BALL_FORCE_DETECT_FRAMES_AFTER_SHOT: int = 8

    # Input mask around goal plus the field below it.
    MASK_BLUR_KERNEL: int = 80
    MASK_SIDE_PADDING_RATIO: float = 0.02
    MASK_TOP_PADDING_RATIO: float = 0.02
    MASK_FIELD_BELOW_GOAL_RATIO: float = 0.02
    PLAYER_MASK_SIDE_PADDING_RATIO: float = 0.45
    PLAYER_MASK_TOP_PADDING_RATIO: float = 0.35
    PLAYER_MASK_FIELD_BELOW_GOAL_RATIO: float = 0.20

    # Simple football rules for this camera angle.
    PLAYERS_MIN_AREA_RATIO: float = 0.00035
    SHOOTER_MIN_Y_RATIO: float = 0.45
    SHOOTER_CENTER_X_MIN: float = 0.25
    SHOOTER_CENTER_X_MAX: float = 0.75
    SHOOTER_BOTTOM_Y_MIN: float = 0.45
    GOALKEEPER_MAX_GOAL_DIST_RATIO: float = 1.15
    ROLE_LOCK_MAX_UPDATE_DIST: int = 180
    ROLE_LOCK_MAX_IOU: float = 0.20
    ROLE_LOCK_MAX_VERTICAL_DRIFT: int = 140
    ROLE_LOCK_MAX_HORIZONTAL_RECOVERY_DIST: int = 520
    ROLE_MIN_ROLE_CENTER_DIST: int = 120

    # Ball detector: YOLO candidates + tracker IDs + light contextual scoring.
    BALL_ACCEPT_CONFIDENCE: float = 0.25
    BALL_CANDIDATE_CONFIDENCE: float = 0.08
    BALL_MIN_AREA_RATIO: float = 0.000006
    BALL_MAX_AREA_RATIO: float = 0.004
    BALL_MIN_ASPECT_RATIO: float = 0.45
    BALL_MAX_ASPECT_RATIO: float = 2.20
    BALL_MAX_JUMP_PRE_SHOT: int = 180
    BALL_MAX_JUMP_POST_SHOT: int = 420
    BALL_PREDICTION_GATE_BASE: int = 90
    BALL_PREDICTION_GATE_PER_MISS: int = 45
    BALL_PRE_SHOT_SHOOTER_RADIUS_RATIO: float = 0.75
    BALL_GOAL_PATH_PADDING_RATIO: float = 0.35
    BALL_REJECT_ABOVE_GOAL_MARGIN_RATIO: float = 0.08
    BALL_REJECT_BEHIND_GOAL: bool = True
    BALL_POST_SHOT_REJECT_PENALTY_SPOT_RADIUS_RATIO: float = 0.10
    BALL_POST_SHOT_MIN_PROGRESS_RATIO: float = 0.03
    BALL_POST_SHOT_BACKTRACK_TOLERANCE_RATIO: float = 0.04
    BALL_SEARCH_RADIUS_PRE_SHOT_RATIO: float = 0.18
    BALL_SEARCH_RADIUS_POST_SHOT_RATIO: float = 0.12
    BALL_MAX_PREDICTED_HOLD_POST_SHOT: int = 4
    BALL_MIN_POST_SHOT_SPEED: float = 8.0
    BALL_MAX_MISSED_PRE_SHOT: int = 8
    BALL_MAX_MISSED_POST_SHOT: int = 5
    SHOT_VELOCITY_THRESHOLD: float = 18.0

    ROLE_STABLE_FRAMES_BEFORE_FREEZE: int = 1
    PLAYERS_MASK_TOP_PADDING: int = 0
    PLAYERS_MASK_SIDE_PADDING: int = 0
    GK_MAX_GHOST_FRAMES_POST_SHOT: int = 0

    @staticmethod
    def get_model_dir() -> Path:
        return Path.cwd()

    @classmethod
    def get_ball_model_path(cls) -> str:
        return cls.BALL_MODEL

    @classmethod
    def get_players_model_path(cls) -> str:
        return cls.PLAYERS_MODEL

    @classmethod
    def get_pose_model_path(cls) -> str:
        return cls.POSE_MODEL


DEFAULT_CONFIG = ModelConfig()
