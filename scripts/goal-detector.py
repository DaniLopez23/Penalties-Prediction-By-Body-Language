from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _debug_common import (
    add_common_args,
    build_config,
    draw_goal,
    draw_label,
    read_frame,
    save_png,
    tile_images,
)
from src.detectors.goal import GoalDetector


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un PNG tecnico de la deteccion de porteria.")
    add_common_args(parser)
    parser.add_argument("--output-name", default="goal-detector.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    frame, frame_index = read_frame(Path(args.input), args.frame)

    detector = GoalDetector(config)
    goal_cfg = config.goal
    height, width = frame.shape[:2]

    y_roi1 = int(height * goal_cfg.goal_roi_top_ratio)
    y_roi2 = int(height * goal_cfg.goal_roi_bottom_ratio)
    x_roi1 = int(width * goal_cfg.goal_roi_x_margin_ratio)
    x_roi2 = int(width * (1.0 - goal_cfg.goal_roi_x_margin_ratio))
    roi = frame[y_roi1:y_roi2, x_roi1:x_roi2]

    white_mask = detector._white_mask(roi)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = detector._best_goal_contour(contours, width, height, x_roi1, y_roi1)
    final_goal = detector.detect(frame)

    roi_panel = frame.copy()
    cv2.rectangle(roi_panel, (x_roi1, y_roi1), (x_roi2 - 1, y_roi2 - 1), (255, 190, 60), 2, cv2.LINE_AA)
    draw_label(
        roi_panel,
        f"ROI busqueda: y={goal_cfg.goal_roi_top_ratio:.2f}-{goal_cfg.goal_roi_bottom_ratio:.2f}, sat<={goal_cfg.white_saturation_max}, val>={goal_cfg.white_value_min}",
        (16, 30),
        color=(255, 190, 60),
    )

    mask_panel = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= width * height * goal_cfg.min_contour_area_ratio:
            cv2.drawContours(mask_panel, [contour], -1, (90, 220, 90), 1, cv2.LINE_AA)
    draw_label(mask_panel, f"contornos validos por area minima={goal_cfg.min_contour_area_ratio}", (10, 28), color=(90, 220, 90))

    contour_panel = frame.copy()
    for contour in contours:
        shifted = contour + np.array([[[x_roi1, y_roi1]]], dtype=np.int32)
        cv2.drawContours(contour_panel, [shifted], -1, (90, 90, 255), 1, cv2.LINE_AA)
    if best is not None:
        x, y, w, h = cv2.boundingRect(best)
        cv2.rectangle(contour_panel, (x_roi1 + x, y_roi1 + y), (x_roi1 + x + w, y_roi1 + y + h), (0, 255, 255), 2)
        draw_label(contour_panel, "mejor contorno: area + centro + ratio", (x_roi1 + x, max(22, y_roi1 + y)), color=(0, 255, 255))
    else:
        draw_label(contour_panel, "sin contorno valido", (16, 30), color=(80, 80, 255))

    final_panel = frame.copy()
    draw_goal(final_panel, final_goal)
    draw_label(
        final_panel,
        f"ajuste final: ratio={goal_cfg.goal_ratio:.1f}, shrink=({goal_cfg.goal_shrink_x:.2f},{goal_cfg.goal_shrink_y:.2f}), frame={frame_index}",
        (16, 30),
        color=(70, 220, 255),
    )

    output = tile_images(
        [
            ("1. ROI donde se busca blanco de porteria", roi_panel),
            ("2. Mascara HSV blanca + morfologia", mask_panel),
            ("3. Contornos candidatos y mejor caja cruda", contour_panel),
            ("4. Porteria final suavizable/fallback", final_panel),
        ],
        columns=2,
    )
    path = save_png(Path(args.output_dir) / args.output_name, output)
    print(path)


if __name__ == "__main__":
    main()
