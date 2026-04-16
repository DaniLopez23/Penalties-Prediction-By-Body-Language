from __future__ import annotations

from pathlib import Path
import time

import cv2

from .analytics import PenaltyAnalytics
from .config import OUTPUT_DIR
from .detection import PenaltyDetector
from .goal_detection import GoalPostDetector
from .pose import PoseEstimator, draw_pose
from .tracking import MultiObjectTracker, TrackedObject


COLORS = {
    "shooter": (0, 200, 0),
    "goalkeeper": (0, 0, 255),
    "ball": (0, 255, 255),
    "goal": (255, 255, 0),
}


def process_video(
    input_video: str | Path,
    output_video: str | Path | None = None,
    show: bool = True,
    det_model_path: str = "yolov8s.pt",
    pose_model_path: str = "yolov8s-pose.pt",
    person_model_path: str | None = None,
    ball_model_path: str | None = None,
    goal_model_path: str | None = None,
    confidence: float = 0.2,
    detection_imgsz: int = 1280,
    max_frames: int | None = None,
    interpolate_ball: bool = True,
    run_pose: bool = True,
    process_every_n_frames: int = 2,
    pose_every_n_frames: int = 2,
) -> Path | None:
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video was not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

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

    detector = PenaltyDetector(
        model_path=det_model_path,
        person_model_path=person_model_path,
        ball_model_path=ball_model_path,
        goal_model_path=goal_model_path,
        confidence=confidence,
        imgsz=detection_imgsz,
        use_goal_roi=True,
        use_goal_hough_validation=True,
    )
    goal_detector = GoalPostDetector()
    tracker = MultiObjectTracker(frame_rate=fps, interpolate_ball=interpolate_ball)
    pose_estimator = PoseEstimator(model_path=pose_model_path, confidence=confidence, imgsz=960) if run_pose else None
    analytics = PenaltyAnalytics()

    processed_frames = 0
    frame_idx = 0
    last_tracked = None
    last_poses = {}
    last_metrics = None
    ema_fps = 0.0

    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        should_process = (frame_idx % max(1, process_every_n_frames)) == 0
        if should_process:
            last_ball = tracker.last_objects_by_role.get("ball")
            last_ball_center = last_ball.center if last_ball is not None else None
            last_goalkeeper = tracker.last_objects_by_role.get("goalkeeper")
            last_goalkeeper_center = last_goalkeeper.center if last_goalkeeper is not None else None

            goal_detection = goal_detector.detect(frame)
            goal_bbox = goal_detection.bbox_xyxy if goal_detection is not None else None

            frame_detections = detector.detect(
                frame,
                goal_bbox_xyxy=goal_bbox,
                last_ball_center=last_ball_center,
                last_goalkeeper_center=last_goalkeeper_center,
            )
            tracked = tracker.update(frame_detections)

            poses = {}
            if pose_estimator is not None and (frame_idx % max(1, pose_every_n_frames)) == 0:
                poses = pose_estimator.estimate(frame, tracked)
            elif last_poses:
                poses = last_poses

            metrics = analytics.update(tracked, poses)
            last_tracked = tracked
            last_poses = poses
            last_metrics = metrics
        elif last_tracked is not None and last_metrics is not None:
            tracked = last_tracked
            poses = last_poses
            metrics = last_metrics
        else:
            frame_idx += 1
            continue

        annotated = frame.copy()
        _draw_tracked_object(annotated, tracked.shooter, "shooter")
        _draw_tracked_object(annotated, tracked.goalkeeper, "goalkeeper")
        _draw_tracked_object(annotated, tracked.ball, "ball")

        if tracked.goal is not None:
            _draw_bbox(annotated, tracked.goal.bbox_xyxy, COLORS["goal"], "goal")

        if tracked.goal_zones is not None:
            _draw_goal_zones(annotated, tracked.goal_zones)

        if "shooter" in poses:
            draw_pose(annotated, poses["shooter"], color=(255, 255, 255))

        if "goalkeeper" in poses:
            draw_pose(annotated, poses["goalkeeper"], color=(255, 120, 120))

        _draw_trajectory(annotated, metrics.ball_trajectory)
        _draw_goalkeeper_vector(annotated, metrics.goalkeeper_center_of_mass_path)
        elapsed = max(1e-6, time.perf_counter() - loop_start)
        inst_fps = 1.0 / elapsed
        ema_fps = inst_fps if ema_fps <= 0 else (0.9 * ema_fps + 0.1 * inst_fps)
        _draw_metrics(
            annotated,
            metrics.shoulder_angle_deg,
            metrics.shot_direction_zone,
            metrics.goalkeeper_movement,
            metrics.goalkeeper_body_angle_deg,
            metrics.goalkeeper_dive_direction,
            metrics.goalkeeper_reaction_time_frames,
            ema_fps,
            elapsed * 1000.0,
        )

        writer.write(annotated)

        if show:
            cv2.imshow("Penalty Analytics", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed_frames += 1
        frame_idx += 1
        if max_frames is not None and processed_frames >= max_frames:
            break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return output_path


def _draw_tracked_object(frame, obj: TrackedObject | None, role: str) -> None:
    if obj is None:
        return

    label = f"{role}#{obj.track_id} {obj.confidence:.2f}"
    if obj.interpolated:
        label += " (interp)"
    _draw_bbox(frame, obj.bbox_xyxy, COLORS.get(role, (255, 255, 255)), label)


def _draw_bbox(frame, bbox_xyxy: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_trajectory(frame, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return

    for idx in range(1, len(points)):
        thickness = max(1, int(3 * idx / max(1, len(points) - 1)))
        cv2.line(frame, points[idx - 1], points[idx], (0, 255, 255), thickness)


def _draw_goal_zones(frame, zones: dict[str, tuple[int, int, int, int]]) -> None:
    for zone_name, (x1, y1, x2, y2) in zones.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(
            frame,
            zone_name,
            (x1 + 2, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )


def _draw_goalkeeper_vector(frame, com_path: list[tuple[int, int]]) -> None:
    if len(com_path) < 2:
        return

    start = com_path[-2]
    end = com_path[-1]
    cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.25)


def _draw_metrics(
    frame,
    shoulder_angle_deg: float | None,
    shot_direction_zone: str | None,
    goalkeeper_movement: str | None,
    goalkeeper_body_angle_deg: float | None,
    goalkeeper_dive_direction: str | None,
    goalkeeper_reaction_frames: int | None,
    fps: float,
    frame_time_ms: float,
) -> None:
    lines = [
        f"Shooter shoulder angle: {shoulder_angle_deg:.1f} deg" if shoulder_angle_deg is not None else "Shooter shoulder angle: n/a",
        f"Shot direction: {shot_direction_zone or 'n/a'}",
        f"Goalkeeper movement: {goalkeeper_movement or 'n/a'}",
        f"Goalkeeper body angle: {goalkeeper_body_angle_deg:.1f} deg" if goalkeeper_body_angle_deg is not None else "Goalkeeper body angle: n/a",
        f"Goalkeeper dive: {goalkeeper_dive_direction or 'n/a'}",
        f"Goalkeeper reaction: {goalkeeper_reaction_frames} frames" if goalkeeper_reaction_frames is not None else "Goalkeeper reaction: n/a",
        f"FPS: {fps:.1f} | Frame: {frame_time_ms:.1f} ms",
        "Press Q to quit",
    ]

    start_x = 16
    start_y = 26
    line_gap = 22
    font_scale = 0.5
    thickness = 1

    box_width = 430
    box_height = line_gap * len(lines) + 10
    cv2.rectangle(frame, (10, 8), (10 + box_width, 8 + box_height), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 8), (10 + box_width, 8 + box_height), (0, 0, 0), 1)

    y = start_y
    for text in lines:
        cv2.putText(frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_gap
