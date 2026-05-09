from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.pipeline import PenaltyAnalysisPipeline


INPUT_VIDEO = Path("data/penalties_mbappe_0.mp4")
OUTPUT_VIDEO: Path | None = None
SHOW_WINDOW = True
DISPLAY_SCALE = 1.0
MAX_FRAMES: int | None = None

DETECTOR_MODEL = "yolo11m.pt"
POSE_MODEL = "yolo11s-pose.pt"

TRACKER = "botsort.yaml"
DEVICE = "cpu"
IMAGE_SIZE = 960
PERSON_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.08
POSE_CONFIDENCE = 0.25
POSE_STRIDE = 1


def build_config() -> PipelineConfig:
    video = replace(
        DEFAULT_CONFIG.video,
        input_path=INPUT_VIDEO,
        show_window=SHOW_WINDOW,
        display_scale=DISPLAY_SCALE,
        max_frames=MAX_FRAMES,
    )
    models = replace(
        DEFAULT_CONFIG.models,
        detector_model=DETECTOR_MODEL,
        pose_model=POSE_MODEL,
        tracker=TRACKER,
        device=DEVICE,
        image_size=IMAGE_SIZE,
        person_confidence=PERSON_CONFIDENCE,
        ball_confidence=BALL_CONFIDENCE,
        pose_confidence=POSE_CONFIDENCE,
        pose_stride=max(1, POSE_STRIDE),
        detector_confidence=min(PERSON_CONFIDENCE, BALL_CONFIDENCE),
    )
    return replace(DEFAULT_CONFIG, video=video, models=models)


def main() -> None:
    config = build_config()
    pipeline = PenaltyAnalysisPipeline(config)
    output_path = pipeline.process_video(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        show_window=SHOW_WINDOW,
        max_frames=MAX_FRAMES,
    )
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()

