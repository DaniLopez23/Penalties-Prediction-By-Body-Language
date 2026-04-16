from pathlib import Path

import cv2

from .config import DEFAULT_MODEL_NAME, OUTPUT_DIR
from .ball_detection import BallDetector, BallDetection
from .goal_detection import GoalDetection, GoalDetector
from .players_detection import KEYPOINT_CONNECTIONS, Keypoint, PlayerDetection, PlayersDetections, PlayersDetector


PERSON_COLOR = (255, 165, 0)
BALL_COLOR = (0, 255, 255)
LAUNCHER_COLOR = (0, 255, 0)
GOALKEEPER_COLOR = (0, 0, 255)
GOAL_COLOR = (255, 255, 0)
POSE_COLOR = (255, 255, 255)
KEYPOINT_COLOR = (0, 215, 255)


def process_video(
    input_video: str | Path,
    output_video: str | Path | None = None,
    show: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    confidence: float = 0.25,
    max_frames: int | None = None,
) -> Path | None:
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el vídeo de entrada: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_video is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"annotated_{input_path.stem}.mp4"
    else:
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    ball_detector = BallDetector(model_name=model_name, confidence=confidence)
    players_detector = PlayersDetector(model_name="yolov8n-pose.pt", confidence=confidence)
    goal_detector = GoalDetector()

    processed_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        goal_detection = goal_detector.detect(frame)
        ball_detection = ball_detector.detect(frame)
        player_detections = players_detector.detect(frame, goal_detection, ball_detection)
        annotated_frame = _draw_annotations(frame, goal_detection, ball_detection, player_detections)

        writer.write(annotated_frame)

        if show:
            cv2.imshow("Football CV", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed_frames += 1
        if max_frames is not None and processed_frames >= max_frames:
            break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return output_path


def _draw_annotations(
    frame,
    goal_detection: GoalDetection | None,
    ball_detection: BallDetection | None,
    player_detections: PlayersDetections,
):
    annotated = frame.copy()

    if goal_detection is not None:
        _draw_box(annotated, goal_detection.bbox, GOAL_COLOR, f"Portería ({goal_detection.side})")

    if ball_detection is not None:
        _draw_box(annotated, ball_detection.bbox, BALL_COLOR, f"Balón {ball_detection.confidence:.2f}")

    if player_detections.launcher is not None:
        _draw_player(annotated, player_detections.launcher, LAUNCHER_COLOR, "Lanzador")

    if player_detections.goalkeeper is not None:
        _draw_player(annotated, player_detections.goalkeeper, GOALKEEPER_COLOR, "Portero")

    cv2.putText(
        annotated,
        "Pulsa Q para salir",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def _draw_box(frame, bbox, color, label):
    x, y, width, height = bbox
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    text_origin = (x, max(20, y - 8))
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_player(frame, player: PlayerDetection, color, label: str) -> None:
    _draw_box(frame, player.bbox, color, label)
    _draw_pose(frame, player.keypoints)


def _draw_pose(frame, keypoints: list[Keypoint]) -> None:
    if not keypoints:
        return

    for start_index, end_index in KEYPOINT_CONNECTIONS:
        if start_index >= len(keypoints) or end_index >= len(keypoints):
            continue

        start = keypoints[start_index]
        end = keypoints[end_index]
        if start.confidence < 0.2 or end.confidence < 0.2:
            continue

        cv2.line(frame, (int(start.x), int(start.y)), (int(end.x), int(end.y)), POSE_COLOR, 2)

    for keypoint in keypoints:
        if keypoint.confidence < 0.2:
            continue
        cv2.circle(frame, (int(keypoint.x), int(keypoint.y)), 3, KEYPOINT_COLOR, -1)
