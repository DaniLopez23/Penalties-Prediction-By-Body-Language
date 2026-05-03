"""Centralized YOLO model configuration for penalty prediction pipeline.

This module manages all YOLO models used in the penalty prediction pipeline:
- Ball detection (yolov8m.pt): Optimized for detecting soccer balls
- Players detection (yolov8m.pt): For detecting shooter and goalkeeper
- Pose estimation (yolov8m-pose.pt): For body pose keypoint estimation

All models are automatically downloaded on first use.
"""

from pathlib import Path
from typing import Optional


class ModelConfig:
    """Centralized configuration for all YOLO models.
    
    Models are chosen based on optimal performance for penalty kick analysis:
    - yolov8m (medium): Better balance between speed and accuracy than yolov8s
    - Pose model: m-size for accurate body keypoint detection
    
    Comparison:
    - yolov8n (nano): Fastest but lower accuracy
    - yolov8s (small): Fast with decent accuracy
    - yolov8m (medium): Better accuracy, ~2x slower than yolov8s (RECOMMENDED)
    - yolov8l (large): High accuracy but slow, overkill for this task
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Model filenames - stored in models/ directory (set via YOLO_HOME)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Ball detection: Uses medium model for better accuracy detecting small object
    # Models are automatically downloaded to MODELS_DIR on first use
    BALL_MODEL: str = "yolov8m.pt"
    
    # Players detection: Medium model to accurately distinguish shooter/goalkeeper
    PLAYERS_MODEL: str = "yolov8m.pt"
    
    # Pose estimation: Medium pose model for accurate keypoint detection
    POSE_MODEL: str = "yolov8m-pose.pt"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Confidence thresholds (permissive by default - filtering done elsewhere)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Ball detection confidence threshold
    BALL_CONFIDENCE: float = 0.5
    
    # Players detection confidence threshold (permissive, filtered by geometry)
    PLAYERS_CONFIDENCE: float = 0.25
    
    # Pose estimation confidence threshold
    POSE_CONFIDENCE: float = 0.25
    
    # ─────────────────────────────────────────────────────────────────────────
    # Processing parameters
    # ─────────────────────────────────────────────────────────────────────────
    
    # Process every N frames for efficiency
    PROCESS_EVERY_N_FRAMES: int = 2
    
    # ─────────────────────────────────────────────────────────────────────────
    # Model directory (where to store/look for model files)
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_model_dir() -> Path:
        """Get the directory where YOLO models are stored.
        
        Models are downloaded automatically by Ultralytics on first use.
        
        Returns:
            Path to model directory (project root).
        """
        return Path.cwd()
    
    @classmethod
    def get_ball_model_path(cls) -> str:
        """Get full path to ball detection model.
        
        Returns:
            Path to ball detection model file.
        """
        return cls.BALL_MODEL
    
    @classmethod
    def get_players_model_path(cls) -> str:
        """Get full path to players detection model.
        
        Returns:
            Path to players detection model file.
        """
        return cls.PLAYERS_MODEL
    
    @classmethod
    def get_pose_model_path(cls) -> str:
        """Get full path to pose estimation model.
        
        Returns:
            Path to pose estimation model file.
        """
        return cls.POSE_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration instance
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = ModelConfig()
