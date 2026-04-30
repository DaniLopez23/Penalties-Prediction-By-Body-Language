from __future__ import annotations

from pathlib import Path

from src.pipeline import PenaltyPipeline


DEFAULT_INPUT = Path("data/penalties_mbappe_2.mp4")
DEFAULT_OUTPUT_DIR = Path("data/cv_output")
INPUT_VIDEO = DEFAULT_INPUT
OUTPUT_VIDEO = None
SHOW_WINDOW = True
MAX_FRAMES = None
PROCESS_EVERY_N_FRAMES = 2
BALL_MODEL = "yolov8s.pt"
PLAYERS_MODEL = None
POSE_MODEL = "yolov8s-pose.pt"
BALL_CONFIDENCE = 0.2
PLAYERS_CONFIDENCE = 0.25
POSE_CONFIDENCE = 0.25


def main() -> None:
	input_video = INPUT_VIDEO
	if not input_video.exists():
		raise FileNotFoundError(f"Input video was not found: {input_video}")

	output_path = OUTPUT_VIDEO
	if output_path is None:
		DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		output_path = DEFAULT_OUTPUT_DIR / f"annotated_{input_video.stem}.mp4"

	pipeline = PenaltyPipeline(
		ball_model=BALL_MODEL,
		players_model=PLAYERS_MODEL,
		pose_model=POSE_MODEL,
		ball_confidence=BALL_CONFIDENCE,
		players_confidence=PLAYERS_CONFIDENCE,
		pose_confidence=POSE_CONFIDENCE,
		process_every_n_frames=PROCESS_EVERY_N_FRAMES,
	)

	result = pipeline.process_video(
		input_video=input_video,
		output_video=output_path,
		show_preview=SHOW_WINDOW,
		max_frames=MAX_FRAMES,
	)

	print(f"Video annotated saved to: {result}")


if __name__ == "__main__":
	main()
