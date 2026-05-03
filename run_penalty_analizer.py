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
	print(f"Ball Detection Model:   {ModelConfig.BALL_MODEL}")
	print(f"Ball Confidence:        {ModelConfig.BALL_CONFIDENCE}")
	print(f"Players Detection Model: {ModelConfig.PLAYERS_MODEL}")
	print(f"Players Confidence:     {ModelConfig.PLAYERS_CONFIDENCE}")
	print(f"Pose Estimation Model:  {ModelConfig.POSE_MODEL}")
	print(f"Pose Confidence:        {ModelConfig.POSE_CONFIDENCE}")
	print(f"Process Every N Frames: {ModelConfig.PROCESS_EVERY_N_FRAMES}")
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
