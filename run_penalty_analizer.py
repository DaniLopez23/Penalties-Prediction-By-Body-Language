from __future__ import annotations

from pathlib import Path

from src.pipeline import PenaltyPipeline
from src.models import ModelConfig


DEFAULT_INPUT = Path("data/penalties_mbappe_1.mp4")
DEFAULT_OUTPUT_DIR = Path("data/cv_output")
INPUT_VIDEO = DEFAULT_INPUT
OUTPUT_VIDEO = None
SHOW_WINDOW = True
MAX_FRAMES = None


def main() -> None:
	input_video = INPUT_VIDEO
	if not input_video.exists():
		raise FileNotFoundError(f"Input video was not found: {input_video}")

	output_path = OUTPUT_VIDEO
	if output_path is None:
		DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		output_path = DEFAULT_OUTPUT_DIR / f"annotated_{input_video.stem}.mp4"

	# Print model configuration
	print("=" * 70)
	print("YOLO Models Configuration")
	print("=" * 70)
	print(f"Ball Model:              {ModelConfig.BALL_MODEL}")
	print(f"Ball Confidence:         {ModelConfig.BALL_CONFIDENCE}")
	print(f"Ball Image Size:         {ModelConfig.BALL_IMGSZ}")
	print(f"People+Pose Model:       {ModelConfig.PLAYERS_MODEL}")
	print(f"People+Pose Confidence:  {ModelConfig.PLAYERS_CONFIDENCE}")
	print(f"People+Pose Image Size:  {ModelConfig.PLAYERS_IMGSZ}")
	print(f"People Tracker:          {ModelConfig.PLAYERS_TRACKER}")
	print(f"People Track Cadence:    every {ModelConfig.PLAYERS_TRACK_EVERY_N_FRAMES} frame(s)")
	print("Ball Tracking:           YOLO refresh + OpenCV Kalman between detections")
	print(f"Ball YOLO Pre-Shot:      every {ModelConfig.BALL_DETECT_EVERY_N_FRAMES_PRE_SHOT} frame(s)")
	print(f"Ball YOLO Post-Shot:     every {ModelConfig.BALL_DETECT_EVERY_N_FRAMES_POST_SHOT} frame(s)")
	print(f"Ball Reacquire Cadence:  every {ModelConfig.BALL_REACQUIRE_EVERY_N_FRAMES} frame(s)")
	print(f"Ball ROI Radius:         {ModelConfig.BALL_ROI_RADIUS}")
	print(f"Ball Max Miss Post-Shot: {ModelConfig.BALL_MAX_MISSED_POST_SHOT}")
	print(f"Goal Refresh Frames:     {ModelConfig.GOAL_DETECT_EVERY_N_FRAMES}")
	print(f"Role Stable Frames:      {ModelConfig.ROLE_STABLE_FRAMES_BEFORE_FREEZE}")
	print(f"Players Mask Top Pad:    {ModelConfig.PLAYERS_MASK_TOP_PADDING}")
	print(f"Players Mask Side Pad:   {ModelConfig.PLAYERS_MASK_SIDE_PADDING}")
	print(f"Mask Blur Kernel:        {ModelConfig.MASK_BLUR_KERNEL}")
	print(f"GK Ghost Post-Shot:      {ModelConfig.GK_MAX_GHOST_FRAMES_POST_SHOT} frames")
	print("=" * 70)
	print()

	# Initialize pipeline with default configuration from ModelConfig
	pipeline = PenaltyPipeline()

	result = pipeline.process_video(
		input_video=input_video,
		output_video=output_path,
		show_preview=SHOW_WINDOW,
		max_frames=MAX_FRAMES,
	)

	print(f"Video annotated saved to: {result}")


if __name__ == "__main__":
	main()
