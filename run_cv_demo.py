from pathlib import Path

from cv_pipeline.config import OUTPUT_DIR
from cv_pipeline.main import process_video


INPUT_VIDEO = Path("data/penalties_mbappe_2.mp4")
OUTPUT_VIDEO = None
DETECTION_MODEL = "yolov8s.pt"
POSE_MODEL = "yolov8s-pose.pt"
PERSON_MODEL = None
BALL_MODEL = None
GOAL_MODEL = None
CONFIDENCE = 0.2
DETECTION_IMGSZ = 1280
MAX_FRAMES = None
SHOW_WINDOW = True
INTERPOLATE_BALL = True
RUN_POSE = True
PROCESS_EVERY_N_FRAMES = 2
POSE_EVERY_N_FRAMES = 2


def main() -> None:
    output_path = Path(OUTPUT_VIDEO) if OUTPUT_VIDEO else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = process_video(
        input_video=INPUT_VIDEO,
        output_video=output_path,
        show=SHOW_WINDOW,
        det_model_path=DETECTION_MODEL,
        pose_model_path=POSE_MODEL,
        person_model_path=PERSON_MODEL,
        ball_model_path=BALL_MODEL,
        goal_model_path=GOAL_MODEL,
        confidence=CONFIDENCE,
        detection_imgsz=DETECTION_IMGSZ,
        max_frames=MAX_FRAMES,
        interpolate_ball=INTERPOLATE_BALL,
        run_pose=RUN_POSE,
        process_every_n_frames=PROCESS_EVERY_N_FRAMES,
        pose_every_n_frames=POSE_EVERY_N_FRAMES,
    )

    if result is not None:
        print(f"Vídeo anotado guardado en: {result}")


if __name__ == "__main__":
    main()