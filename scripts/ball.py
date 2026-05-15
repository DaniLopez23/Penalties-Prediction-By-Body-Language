from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _debug_common import (
    add_common_args,
    build_config,
    draw_box,
    draw_goal,
    draw_label,
    overlay_mask,
    read_neighbor_frames,
    save_png,
    split_detections,
    tile_images,
)
from src.detectors.goal import GoalDetector
from src.detectors.yolo import YOLODetector
from src.preprocessing.roi import PlayAreaMasker
from src.tracking.ball import BallTracker


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un PNG tecnico del tracking/deteccion de balon.")
    add_common_args(parser)
    parser.add_argument("--output-name", default="ball.png")
    return parser


def motion_threshold_panel(previous: np.ndarray, current: np.ndarray, mask: np.ndarray | None, config) -> np.ndarray:
    gray_prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    kernel_size = max(3, config.ball.motion_blur_kernel | 1)
    gray_prev = cv2.GaussianBlur(gray_prev, (kernel_size, kernel_size), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (kernel_size, kernel_size), 0)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, threshold = cv2.threshold(diff, config.ball.motion_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    if mask is not None:
        threshold = cv2.bitwise_and(threshold, mask)
    return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    previous, frame, frame_index = read_neighbor_frames(Path(args.input), args.frame)

    goal_detector = GoalDetector(config)
    goal = goal_detector.detect(frame)
    masker = PlayAreaMasker(config)
    mask = masker.build_mask(frame.shape, goal)
    inference_frame = masker.apply_for_detection(frame, mask)

    detector = YOLODetector(config)
    detections = masker.filter_detections(detector.detect(inference_frame), mask)
    persons, balls = split_detections(detections, config)

    tracker = BallTracker(config)
    tracker.previous_gray = cv2.GaussianBlur(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
        (max(3, config.ball.motion_blur_kernel | 1), max(3, config.ball.motion_blur_kernel | 1)),
        0,
    )
    motion_candidates = tracker._motion_candidates(inference_frame, None, persons, mask)
    yolo_candidates = tracker._yolo_candidates(balls, None, frame.shape, mask)
    selected_ball = tracker.update(inference_frame, balls, persons, valid_mask=mask)

    roi_panel = overlay_mask(frame, mask, (40, 180, 255), alpha=0.25)
    draw_goal(roi_panel, goal)
    draw_label(roi_panel, f"ROI valida para balon; frame={frame_index}", (16, 30), color=(0, 255, 255))

    yolo_panel = frame.copy()
    for person in persons:
        draw_box(yolo_panel, person.xyxy, (80, 180, 255), f"person {person.confidence:.2f}", 1)
    for ball in balls:
        draw_box(yolo_panel, ball.xyxy, (35, 90, 255), f"YOLO ball {ball.confidence:.2f}", 2)
    draw_label(yolo_panel, f"YOLO candidates: {len(balls)} balon, {len(persons)} personas", (16, 30), color=(35, 90, 255))

    motion_panel = motion_threshold_panel(previous, inference_frame, mask, config)
    for candidate in motion_candidates:
        x, y = (int(round(v)) for v in candidate.center)
        cv2.circle(motion_panel, (x, y), candidate.radius, (0, 255, 255), 2, cv2.LINE_AA)
        draw_label(motion_panel, f"motion {candidate.confidence:.2f}", (x + 6, y), color=(0, 255, 255), scale=0.42)
    draw_label(
        motion_panel,
        f"absdiff + threshold>{config.ball.motion_threshold}; area {config.ball.min_motion_area_ratio}-{config.ball.max_motion_area_ratio}",
        (16, 30),
        color=(0, 255, 255),
    )

    final_panel = frame.copy()
    for candidate in yolo_candidates + motion_candidates:
        x, y = (int(round(v)) for v in candidate.center)
        color = (35, 90, 255) if candidate.source == "yolo" else (0, 220, 255)
        cv2.circle(final_panel, (x, y), candidate.radius, color, 1, cv2.LINE_AA)
        draw_label(final_panel, f"{candidate.source}:{candidate.confidence:.2f}", (x + 7, y), color=color, scale=0.42)
    if selected_ball is not None:
        x, y = (int(round(v)) for v in selected_ball.center)
        cv2.circle(final_panel, (x, y), selected_ball.radius + 4, (0, 255, 0), 3, cv2.LINE_AA)
        draw_label(final_panel, f"seleccionado {selected_ball.source} conf={selected_ball.confidence:.2f}", (x + 10, y + 14), color=(0, 255, 0))
    else:
        draw_label(final_panel, "sin balon seleccionado", (16, 58), color=(80, 80, 255))
    draw_label(final_panel, "score = YOLO/movimiento + cercania prediccion/ultimo - solape jugador", (16, 30), color=(0, 255, 0))

    output = tile_images(
        [
            ("1. ROI donde se acepta el balon", roi_panel),
            ("2. Detecciones YOLO filtradas", yolo_panel),
            ("3. Candidatos por movimiento entre frames", motion_panel),
            ("4. Fusion de candidatos y seleccion final", final_panel),
        ],
        columns=2,
    )
    path = save_png(Path(args.output_dir) / args.output_name, output)
    print(path)


if __name__ == "__main__":
    main()
