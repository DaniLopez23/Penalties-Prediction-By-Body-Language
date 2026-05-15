from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from _debug_common import (
    add_common_args,
    build_config,
    draw_box,
    draw_goal,
    draw_label,
    overlay_mask,
    read_frame,
    save_png,
    split_detections,
    tile_images,
)
from src.detectors.goal import GoalDetector
from src.detectors.yolo import YOLODetector
from src.models import BallState, Detection, GoalBox
from src.preprocessing.roi import PlayAreaMasker
from src.tracking.ball import BallTracker


@dataclass
class FrameInputs:
    frame_index: int
    frame: np.ndarray
    inference_frame: np.ndarray
    goal: Optional[GoalBox]
    mask: Optional[np.ndarray]
    persons: list[Detection]
    balls: list[Detection]


@dataclass
class BallDebugStep:
    prediction: Optional[tuple[float, float]]
    last_state_before: Optional[BallState]
    missing_frames_before: int
    yolo_candidates: list[BallState]
    motion_candidates: list[BallState]
    selected_ball: Optional[BallState]
    trail_before: list[tuple[int, int]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un PNG tecnico del tracking/deteccion de balon.")
    add_common_args(parser)
    parser.add_argument("--output-name", default="ball.png")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=4,
        help="Frames anteriores que se procesan antes del frame objetivo para mostrar Kalman/trail reales.",
    )
    parser.add_argument(
        "--warmup-stride",
        type=int,
        default=1,
        help="Procesa 1 de cada N frames de warmup; el frame anterior al objetivo siempre se mantiene.",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=180,
        help="Tamano en pixeles del recorte local alrededor del balon/candidato principal.",
    )
    return parser


def read_frame_window(source: Path, frame_index: int, warmup_frames: int) -> list[tuple[int, np.ndarray]]:
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame, used_index = read_frame(source, frame_index)
        return [(used_index, frame)]

    if not source.exists():
        raise FileNotFoundError(f"No existe la entrada: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        safe_index = max(0, frame_index)
        if total > 0:
            safe_index = min(safe_index, total - 1)
        start_index = max(0, safe_index - max(0, warmup_frames))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        frames: list[tuple[int, np.ndarray]] = []
        current_index = start_index
        while current_index <= safe_index:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append((current_index, frame))
            current_index += 1
        if not frames or frames[-1][0] != safe_index:
            raise RuntimeError(f"No se pudo extraer la ventana hasta el frame {safe_index} de {source}")
        return frames
    finally:
        capture.release()


def select_warmup_frames(
    frame_window: list[tuple[int, np.ndarray]],
    warmup_stride: int,
) -> list[tuple[int, np.ndarray]]:
    if len(frame_window) <= 1:
        return []

    stride = max(1, warmup_stride)
    target_index = frame_window[-1][0]
    selected: list[tuple[int, np.ndarray]] = []
    for index, frame in frame_window[:-1]:
        distance_to_target = target_index - index
        if distance_to_target == 1 or distance_to_target % stride == 0:
            selected.append((index, frame))
    return selected


def blurred_gray(frame: np.ndarray, config) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = max(3, config.ball.motion_blur_kernel | 1)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def motion_threshold_from_previous(
    previous_gray: Optional[np.ndarray],
    current: np.ndarray,
    mask: Optional[np.ndarray],
    config,
) -> np.ndarray:
    if previous_gray is None:
        return np.zeros(current.shape[:2], dtype=np.uint8)
    gray_curr = blurred_gray(current, config)
    diff = cv2.absdiff(previous_gray, gray_curr)
    _, threshold = cv2.threshold(diff, config.ball.motion_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    if mask is not None:
        threshold = cv2.bitwise_and(threshold, mask)
    return threshold


def build_frame_inputs(
    frame_index: int,
    frame: np.ndarray,
    goal_detector: GoalDetector,
    masker: PlayAreaMasker,
    detector: YOLODetector,
    config,
) -> FrameInputs:
    goal = goal_detector.detect(frame)
    mask = masker.build_mask(frame.shape, goal)
    inference_frame = masker.apply_for_detection(frame, mask)
    detections = masker.filter_detections(detector.detect(inference_frame), mask)
    persons, balls = split_detections(detections, config)
    return FrameInputs(frame_index, frame, inference_frame, goal, mask, persons, balls)


def debug_update_ball(tracker: BallTracker, inputs: FrameInputs) -> BallDebugStep:
    last_state_before = tracker.last_state
    missing_frames_before = tracker.missing_frames
    trail_before = list(tracker.trail)

    prediction = tracker._predict()
    yolo_candidates = tracker._yolo_candidates(inputs.balls, prediction, inputs.frame.shape, inputs.mask)
    motion_candidates = tracker._motion_candidates(inputs.inference_frame, prediction, inputs.persons, inputs.mask)
    candidates = yolo_candidates + motion_candidates

    selected_ball: Optional[BallState] = None
    if candidates:
        selected_ball = max(candidates, key=lambda item: item.confidence)
        tracker._correct(selected_ball.center)
        tracker.missing_frames = 0
        tracker.last_state = selected_ball
    elif prediction is not None and tracker.missing_frames < tracker.config.ball.max_missing_frames:
        tracker.missing_frames += 1
        radius = tracker.last_state.radius if tracker.last_state is not None else tracker.config.ball.min_ball_radius
        selected_ball = BallState(prediction, radius, 0.15, "kalman", observed=False)
        tracker.last_state = selected_ball
    else:
        tracker.missing_frames += 1
        if tracker.missing_frames > tracker.config.ball.max_missing_frames:
            tracker.kalman = None
            tracker.trail.clear()

    tracker.previous_gray = blurred_gray(inputs.inference_frame, tracker.config)
    if selected_ball is not None:
        x, y = selected_ball.center
        tracker.trail.append((int(round(x)), int(round(y))))

    return BallDebugStep(
        prediction=prediction,
        last_state_before=last_state_before,
        missing_frames_before=missing_frames_before,
        yolo_candidates=yolo_candidates,
        motion_candidates=motion_candidates,
        selected_ball=selected_ball,
        trail_before=trail_before,
    )


def draw_ball_marker(
    panel: np.ndarray,
    ball: BallState,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x, y = (int(round(v)) for v in ball.center)
    cv2.circle(panel, (x, y), ball.radius, color, thickness, cv2.LINE_AA)
    cv2.circle(panel, (x, y), 2, color, -1, cv2.LINE_AA)
    draw_label(panel, label, (x + ball.radius + 6, y), color=color, scale=0.42)


def draw_point_marker(
    panel: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = (int(round(v)) for v in point)
    cv2.drawMarker(panel, (x, y), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    draw_label(panel, label, (x + 10, y - 6), color=color, scale=0.42)


def draw_trail(panel: np.ndarray, trail: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(trail) < 2:
        return
    for idx in range(1, len(trail)):
        alpha = idx / max(1, len(trail) - 1)
        thickness = max(1, int(round(1 + 3 * alpha)))
        trail_color = tuple(int(channel * alpha) for channel in color)
        cv2.line(panel, trail[idx - 1], trail[idx], trail_color, thickness, cv2.LINE_AA)


def draw_player_penalty_zones(panel: np.ndarray, persons: list[Detection]) -> None:
    for person in persons:
        x1, y1, x2, y2 = person.xyxy
        margin = max(8.0, 0.04 * max(person.width, person.height))
        cv2.rectangle(
            panel,
            (int(round(x1 - margin)), int(round(y1 - margin))),
            (int(round(x2 + margin)), int(round(y2 + margin))),
            (80, 180, 255),
            1,
            cv2.LINE_AA,
        )

        foot_top = person.y1 + person.height * 0.72
        horizontal_margin = max(10.0, person.width * 0.18)
        vertical_margin = max(8.0, person.height * 0.06)
        cv2.rectangle(
            panel,
            (int(round(person.x1 - horizontal_margin)), int(round(foot_top))),
            (int(round(person.x2 + horizontal_margin)), int(round(person.y2 + vertical_margin))),
            (70, 80, 255),
            1,
            cv2.LINE_AA,
        )


def tracking_panel(inputs: FrameInputs, step: BallDebugStep, config) -> np.ndarray:
    panel = inputs.frame.copy()
    draw_trail(panel, step.trail_before, (35, 90, 255))

    height, width = inputs.frame.shape[:2]
    diag = math.hypot(width, height)
    if step.last_state_before is not None:
        last_x, last_y = (int(round(v)) for v in step.last_state_before.center)
        jump_radius = int(round(diag * config.ball.max_candidate_jump_ratio * (1.0 + min(2.0, step.missing_frames_before * 0.25))))
        cv2.circle(panel, (last_x, last_y), jump_radius, (170, 170, 170), 1, cv2.LINE_AA)
        draw_ball_marker(panel, step.last_state_before, (170, 170, 170), "ultimo estado", 1)

    if step.prediction is not None:
        pred_x, pred_y = (int(round(v)) for v in step.prediction)
        prediction_radius = int(round(diag * config.ball.max_prediction_distance_ratio))
        cv2.circle(panel, (pred_x, pred_y), prediction_radius, (255, 120, 255), 1, cv2.LINE_AA)
        draw_point_marker(panel, step.prediction, (255, 120, 255), "prediccion Kalman")

    for candidate in step.yolo_candidates + step.motion_candidates:
        color = (35, 90, 255) if candidate.source == "yolo" else (0, 220, 255)
        draw_ball_marker(panel, candidate, color, f"{candidate.source} score={candidate.confidence:.2f}", 1)

    if step.selected_ball is not None:
        color = (0, 255, 0) if step.selected_ball.observed else (190, 190, 190)
        draw_ball_marker(panel, step.selected_ball, color, f"seleccionado {step.selected_ball.source}", 3)
    else:
        draw_label(panel, "sin balon seleccionado", (16, 58), color=(80, 80, 255))

    draw_label(
        panel,
        "tracking: trail previo + ultimo estado + prediccion Kalman + candidatos puntuados",
        (16, 30),
        color=(0, 255, 0),
    )
    return panel


def zoom_panel(inputs: FrameInputs, motion_mask: np.ndarray, step: BallDebugStep, zoom_size: int) -> np.ndarray:
    focus = focus_point(step, inputs.frame)
    original_crop = crop_with_markers(inputs.frame, focus, zoom_size, step, "frame")
    motion_crop = crop_with_markers(cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR), focus, zoom_size, step, "motion mask")
    target_h = 360
    original_crop = resize_to_height(original_crop, target_h)
    motion_crop = resize_to_height(motion_crop, target_h)
    panel = np.hstack([original_crop, motion_crop])
    draw_label(panel, f"zoom local {zoom_size}px alrededor de {format_point(focus)}", (12, 30), color=(0, 255, 255))
    return panel


def crop_with_markers(
    image: np.ndarray,
    focus: tuple[float, float],
    zoom_size: int,
    step: BallDebugStep,
    title: str,
) -> np.ndarray:
    height, width = image.shape[:2]
    half = max(20, zoom_size // 2)
    cx, cy = (int(round(v)) for v in focus)
    x1 = max(0, min(width - 1, cx - half))
    y1 = max(0, min(height - 1, cy - half))
    x2 = max(x1 + 1, min(width, cx + half))
    y2 = max(y1 + 1, min(height, cy + half))
    crop = image[y1:y2, x1:x2].copy()

    for candidate in step.yolo_candidates + step.motion_candidates:
        color = (35, 90, 255) if candidate.source == "yolo" else (0, 220, 255)
        draw_shifted_ball(crop, candidate, x1, y1, color, 1)
    if step.selected_ball is not None:
        draw_shifted_ball(crop, step.selected_ball, x1, y1, (0, 255, 0), 2)
    if step.prediction is not None:
        px, py = shifted_point(step.prediction, x1, y1)
        cv2.drawMarker(crop, (px, py), (255, 120, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    draw_label(crop, title, (8, 24), color=(255, 255, 255), scale=0.45)
    return crop


def draw_shifted_ball(
    image: np.ndarray,
    ball: BallState,
    x_offset: int,
    y_offset: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x, y = shifted_point(ball.center, x_offset, y_offset)
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        cv2.circle(image, (x, y), ball.radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)


def shifted_point(point: tuple[float, float], x_offset: int, y_offset: int) -> tuple[int, int]:
    return int(round(point[0])) - x_offset, int(round(point[1])) - y_offset


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / max(1, image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def focus_point(step: BallDebugStep, frame: np.ndarray) -> tuple[float, float]:
    if step.selected_ball is not None:
        return step.selected_ball.center
    candidates = step.yolo_candidates + step.motion_candidates
    if candidates:
        return max(candidates, key=lambda item: item.confidence).center
    if step.prediction is not None:
        return step.prediction
    height, width = frame.shape[:2]
    return width * 0.5, height * 0.5


def format_point(point: Optional[tuple[float, float]]) -> str:
    if point is None:
        return "-"
    return f"({point[0]:.1f},{point[1]:.1f})"


def format_ball(ball: Optional[BallState]) -> str:
    if ball is None:
        return "-"
    observed = "obs" if ball.observed else "pred"
    return f"{ball.source}/{observed} score={ball.confidence:.2f} center={format_point(ball.center)}"


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    frame_window = read_frame_window(Path(args.input), args.frame, args.warmup_frames)

    goal_detector = GoalDetector(config)
    masker = PlayAreaMasker(config)
    detector = YOLODetector(config)
    tracker = BallTracker(config)

    warmup_frames = select_warmup_frames(frame_window, args.warmup_stride)
    if warmup_frames:
        print(
            f"Warmup tracking: {len(warmup_frames)} frames antes del objetivo "
            f"(ventana={len(frame_window) - 1}, stride={max(1, args.warmup_stride)})",
            flush=True,
        )

    for position, (index, raw_frame) in enumerate(warmup_frames, start=1):
        print(f"  warmup {position}/{len(warmup_frames)}: frame {index}", flush=True)
        warmup_inputs = build_frame_inputs(index, raw_frame, goal_detector, masker, detector, config)
        tracker.update(
            warmup_inputs.inference_frame,
            warmup_inputs.balls,
            warmup_inputs.persons,
            valid_mask=warmup_inputs.mask,
        )

    target_index, target_frame = frame_window[-1]
    print(f"Frame objetivo: {target_index}", flush=True)
    target_inputs = build_frame_inputs(target_index, target_frame, goal_detector, masker, detector, config)
    previous_gray_for_motion = None if tracker.previous_gray is None else tracker.previous_gray.copy()
    step = debug_update_ball(tracker, target_inputs)
    motion_mask = motion_threshold_from_previous(previous_gray_for_motion, target_inputs.inference_frame, target_inputs.mask, config)

    roi_panel = overlay_mask(target_inputs.frame, target_inputs.mask, (40, 180, 255), alpha=0.25)
    draw_goal(roi_panel, target_inputs.goal)
    draw_trail(roi_panel, step.trail_before, (35, 90, 255))
    if step.selected_ball is not None:
        draw_ball_marker(roi_panel, step.selected_ball, (0, 255, 0), "balon seleccionado", 2)
    draw_label(
        roi_panel,
        f"ROI valida para balon; frame={target_inputs.frame_index}; fuera de ROI se descarta si reject_outside_roi={config.ball.reject_outside_roi}",
        (16, 30),
        color=(0, 255, 255),
    )

    candidates_panel = target_inputs.frame.copy()
    draw_player_penalty_zones(candidates_panel, target_inputs.persons)
    for person in target_inputs.persons:
        draw_box(candidates_panel, person.xyxy, (80, 180, 255), f"person {person.confidence:.2f}", 1)
    for ball in target_inputs.balls:
        draw_box(candidates_panel, ball.xyxy, (35, 90, 255), f"YOLO ball {ball.confidence:.2f}", 2)
    for candidate in step.yolo_candidates + step.motion_candidates:
        color = (35, 90, 255) if candidate.source == "yolo" else (0, 220, 255)
        draw_ball_marker(candidates_panel, candidate, color, f"{candidate.source} {candidate.confidence:.2f}", 1)
    draw_label(
        candidates_panel,
        f"Candidatos: YOLO={len(step.yolo_candidates)}, movimiento={len(step.motion_candidates)}; rojo=pies penalizados, naranja=solape jugador",
        (16, 30),
        color=(35, 90, 255),
    )

    final_panel = tracking_panel(target_inputs, step, config)
    local_zoom_panel = zoom_panel(target_inputs, motion_mask, step, args.zoom_size)

    output = tile_images(
        [
            ("1. ROI donde se acepta el balon", roi_panel),
            ("2. Candidatos", candidates_panel),
            ("3. Seguimiento: trail, Kalman y seleccion final", final_panel),
            ("4. Zoom local del balon/candidato", local_zoom_panel),
        ],
        columns=2,
    )
    path = save_png(Path(args.output_dir) / args.output_name, output)
    print(path)


if __name__ == "__main__":
    main()
