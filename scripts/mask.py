from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from _debug_common import (
    add_common_args,
    build_config,
    draw_goal,
    draw_label,
    overlay_mask,
    read_frame,
    save_png,
    tile_images,
)
from src.detectors.goal import GoalDetector
from src.preprocessing.roi import PlayAreaMasker


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un PNG tecnico de la mascara/ROI de juego.")
    add_common_args(parser)
    parser.add_argument("--output-name", default="mask.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    frame, frame_index = read_frame(Path(args.input), args.frame)

    goal = GoalDetector(config).detect(frame)
    masker = PlayAreaMasker(config)
    mask = masker.build_mask(frame.shape, goal)

    original_panel = frame.copy()
    draw_goal(original_panel, goal)
    draw_label(original_panel, f"frame={frame_index}: la ROI parte de la porteria detectada", (16, 30), color=(70, 220, 255))

    mask_panel = frame.copy()
    if mask is not None:
        mask_panel = overlay_mask(mask_panel, mask, (40, 180, 255), alpha=0.45)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_panel, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    draw_goal(mask_panel, goal)
    draw_label(
        mask_panel,
        f"poligono: top_expand={config.roi.top_expand_goal_width:.2f}, bottom_width={config.roi.bottom_width_goal_multiplier:.2f}x, bottom_y={config.roi.bottom_y_ratio:.2f}",
        (16, 30),
        color=(0, 255, 255),
    )

    detection_panel = masker.apply_for_detection(frame, mask)
    draw_label(detection_panel, f"frame para YOLO: fuera de mascara se rellena con {config.roi.detection_fill_value}", (16, 30), color=(255, 255, 255))

    blur_panel = masker.apply_blur(frame, mask)
    draw_label(blur_panel, f"visualizacion: exterior desenfocado kernel={config.roi.blur_kernel}", (16, 30), color=(255, 255, 255))

    output = tile_images(
        [
            ("1. Frame base + porteria", original_panel),
            ("2. Mascara ROI construida desde la porteria", mask_panel),
            ("3. Imagen que recibe el detector", detection_panel),
            ("4. Imagen final para visualizacion", blur_panel),
        ],
        columns=2,
    )
    path = save_png(Path(args.output_dir) / args.output_name, output)
    print(path)


if __name__ == "__main__":
    main()
