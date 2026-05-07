"""Centralized model and runtime configuration for the penalty pipeline.

Current pipeline strategy:
- People, pose and player track IDs come from a single YOLO pose tracker call.
- Ball detection is refreshed periodically with YOLO and propagated between
  refreshes with a Kalman filter plus physics gates.
- Goal detection is refreshed less often and stabilized in the pipeline.
"""

from pathlib import Path


class ModelConfig:
    """Default models and knobs tuned for 1080p penalty footage."""

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    # YOLO11 pose gives player boxes, keypoints and track IDs in one pass.
    # Use yolo11m-pose.pt if runtime is too slow; keep the large model when
    # goalkeeper dives and low-confidence poses are more important than speed.
    PLAYERS_MODEL: str = "yolo11l-pose.pt"
    POSE_MODEL: str = PLAYERS_MODEL

    # COCO sports-ball detector baseline. A fine-tuned ball model should keep
    # the same interface and replace this path when available.
    BALL_MODEL: str = "yolo11m.pt"

    PERSON_CLASS_ID: int = 0
    BALL_CLASS_ID: int = 32

    # ------------------------------------------------------------------
    # Inference thresholds and sizes
    # ------------------------------------------------------------------
    # Keep people permissive: play-zone filtering, role assignment and tracker
    # continuity remove most false positives after model inference.
    PLAYERS_CONFIDENCE: float = 0.15
    POSE_CONFIDENCE: float = 0.25

    # Keep ball confidence low because the post-shot ball is small, blurred and
    # often partially occluded by net/goalkeeper. Physics gates validate it.
    BALL_CONFIDENCE: float = 0.15

    PLAYERS_IMGSZ: int = 1280
    BALL_IMGSZ: int = 1280

    # ------------------------------------------------------------------
    # Tracking cadence
    # ------------------------------------------------------------------
    # Player tracking is delegated to Ultralytics BoT-SORT. Keep this at 1 for
    # maximum ID stability; raise it only after validating that IDs survive the
    # goalkeeper dive.
    PLAYERS_TRACKER: str = "botsort.yaml"
    PLAYERS_TRACK_EVERY_N_FRAMES: int = 5
    PROCESS_EVERY_N_FRAMES: int = 1

    # Ball detection has separate modes. Pre-shot can be cheaper, post-shot and
    # re-acquisition should be aggressive because the ball is fast and blurred.
    BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT: int = 10
    BALL_DETECT_EVERY_N_FRAMES_POST_SHOT: int = 2
    BALL_REACQUIRE_EVERY_N_FRAMES: int = 4
    BALL_FORCE_DETECT_FRAMES_AFTER_SHOT: int = 8
    BALL_DETECT_EVERY_N_FRAMES: int = BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT

    GOAL_DETECT_EVERY_N_FRAMES: int = 10

    # Optional Ultralytics runtime hints. Leave device as None to let
    # Ultralytics choose automatically.
    PLAYERS_DEVICE: str | None = None
    BALL_DEVICE: str | None = None
    PLAYERS_HALF: bool = False
    BALL_HALF: bool = False

    # ------------------------------------------------------------------
    # Player mask and detector filters
    # ------------------------------------------------------------------
    # The blur should start just outside the posts, but keep extra vertical
    # room above the crossbar so YOLO still sees a diving goalkeeper.
    PLAYERS_MASK_TOP_PADDING: int = 110
    PLAYERS_MASK_SIDE_PADDING: int = 35
    PLAYERS_MASK_KEEP_FIELD_BELOW_GOAL: int = 80
    MASK_BLUR_KERNEL: int = 35

    PLAYERS_CENTRAL_X_MIN: float = 0.05
    PLAYERS_CENTRAL_X_MAX: float = 0.95
    PLAYERS_TOP_IGNORE_RATIO: float = 0.08
    PLAYERS_MIN_AREA_RATIO: float = 0.00035

    # ------------------------------------------------------------------
    # Ball tracking/re-acquisition
    # ------------------------------------------------------------------
    BALL_BASE_MAX_DIST: int = 280
    BALL_MAX_DIST_PER_MISS: int = 30
    BALL_ROI_RADIUS: int = 420
    BALL_MAX_MISSED_PRE_SHOT: int = 24
    BALL_MAX_MISSED_POST_SHOT: int = 10
    BALL_MIN_SPEED_FOR_DIR_FILTER: float = 14.0
    BALL_VELOCITY_EMA_ALPHA: float = 0.20

    BALL_MIN_AREA: int = 20
    BALL_MAX_AREA: int = 8000
    BALL_AREA_RATIO_MIN: float = 0.55
    BALL_AREA_RATIO_MAX: float = 1.90
    BALL_MAX_ASPECT_RATIO: float = 1.6

    BALL_NEAR_GOAL_Y_FRACTION: float = 0.45
    BALL_NEAR_GOAL_MAX_AREA: int = 4000
    BALL_NEAR_GOAL_MIN_CONF: float = 0.20
    BALL_NEAR_GOAL_MAX_ASPECT: float = 1.30

    BALL_REACQ_MIN_MISSED: int = 3
    BALL_REACQ_CONFIRM_FRAMES: int = 1
    BALL_REACQ_MAX_DRIFT: int = 140
    BALL_DECEL_PER_MISS: float = 0.92
    BALL_ELLIPSE_R_ALONG_BASE: int = 60
    BALL_ELLIPSE_R_PERP_BASE: int = 40
    BALL_ELLIPSE_R_PERP_GROW: int = 8

    # ------------------------------------------------------------------
    # Role continuity and occlusion recovery
    # ------------------------------------------------------------------
    GK_MAX_GHOST_FRAMES: int = 8
    GK_MAX_GHOST_FRAMES_POST_SHOT: int = 60
    SHOOTER_MAX_GHOST_FRAMES: int = 4
    GHOST_CONF_DECAY: float = 0.80
    GHOST_CONF_DECAY_POST_SHOT: float = 0.98

    ROLE_CROSSBAR_MARGIN: int = 40
    ROLE_LATERAL_MARGIN: int = 80
    ROLE_GK_REACQUIRE_MAX_DIST: int = 420
    ROLE_SHOOTER_REACQUIRE_MAX_DIST: int = 360
    ROLE_STABLE_FRAMES_BEFORE_FREEZE: int = 3

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
