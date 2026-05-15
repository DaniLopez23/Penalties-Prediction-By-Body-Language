from __future__ import annotations

import argparse
from pathlib import Path

import cv2

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
from src.preprocessing.roi import PlayAreaMasker
from src.tracking.roles import PlayerRoleAssigner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un PNG tecnico de asignacion de roles.")
    add_common_args(parser)
    parser.add_argument("--output-name", default="role_assigment.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    frame, frame_index = read_frame(Path(args.input), args.frame)

    goal = GoalDetector(config).detect(frame)
    masker = PlayAreaMasker(config)
    mask = masker.build_mask(frame.shape, goal)
    inference_frame = masker.apply_for_detection(frame, mask)

    detector = YOLODetector(config)
    detections = masker.filter_detections(detector.detect(inference_frame), mask)
    persons, balls = split_detections(detections, config)
    ball_center = max(balls, key=lambda det: det.confidence).center if balls else None

    assigner = PlayerRoleAssigner(config)
    assignments = assigner.assign(persons, frame.shape, goal, ball_center)

    roi_panel = overlay_mask(frame, mask, (40, 180, 255), alpha=0.25)
    draw_goal(roi_panel, goal)
    if ball_center is not None:
        cv2.circle(roi_panel, (int(ball_center[0]), int(ball_center[1])), 6, (35, 90, 255), 2, cv2.LINE_AA)
        draw_label(roi_panel, "balon usado para score striker", (int(ball_center[0]) + 8, int(ball_center[1])), color=(35, 90, 255))
    draw_label(roi_panel, f"frame={frame_index}: roles se calculan sobre personas dentro de la ROI", (16, 30), color=(0, 255, 255))

    scores_panel = frame.copy()
    draw_goal(scores_panel, goal)
    for index, person in enumerate(persons):
        gk_score = assigner._goalkeeper_score(person, frame.shape, goal)
        striker_score = assigner._striker_score(person, frame.shape, goal, ball_center)
        color = (180, 180, 180)
        draw_box(scores_panel, person.xyxy, color, None, 1)
        draw_label(
            scores_panel,
            f"#{index} GK={gk_score:.2f} STR={striker_score:.2f}",
            (int(person.x1), max(22, int(person.y1))),
            color=color,
            scale=0.46,
        )
    draw_label(scores_panel, "score GK: cerca/centrado en porteria; score STR: bajo+centro+tamano+cerca balon", (16, 30))

    final_panel = frame.copy()
    draw_goal(final_panel, goal)
    for role, person in assignments.items():
        color = (70, 255, 120) if role == "goalkeeper" else (255, 180, 40)
        draw_box(final_panel, person.xyxy, color, role, 3)
        cx, cy = (int(round(v)) for v in person.center)
        cv2.circle(final_panel, (cx, cy), 4, color, -1, cv2.LINE_AA)
    unassigned = [person for person in persons if person not in assignments.values()]
    for person in unassigned:
        draw_box(final_panel, person.xyxy, (150, 150, 150), "no asignado", 1)
    draw_label(final_panel, f"asignaciones finales: {', '.join(assignments.keys()) or 'ninguna'}", (16, 30), color=(255, 255, 255))

    rules_panel = frame.copy()
    draw_goal(rules_panel, goal)
    if goal is not None:
        margin = max(goal.width, goal.height) * config.roles.goalkeeper_goal_margin_ratio
        cv2.rectangle(
            rules_panel,
            (int(goal.x1 - margin), int(goal.y1 - margin)),
            (int(goal.x2 + margin), int(goal.y2 + margin)),
            (70, 255, 120),
            1,
            cv2.LINE_AA,
        )
        draw_label(rules_panel, f"margen portero={config.roles.goalkeeper_goal_margin_ratio:.2f}x porteria", (16, 58), color=(70, 255, 120))
    if ball_center is not None:
        radius = int(((frame.shape[1] ** 2 + frame.shape[0] ** 2) ** 0.5) * config.roles.striker_ball_max_distance_ratio)
        cv2.circle(rules_panel, (int(ball_center[0]), int(ball_center[1])), radius, (255, 180, 40), 1, cv2.LINE_AA)
        draw_label(rules_panel, f"radio balon para lanzador={config.roles.striker_ball_max_distance_ratio:.2f} diagonal", (16, 86), color=(255, 180, 40))
    draw_label(rules_panel, f"min_assignment_score={config.roles.min_assignment_score:.2f}; lock_track_ids={config.roles.lock_track_ids}", (16, 30))

    output = tile_images(
        [
            ("1. ROI + balon usado como pista", roi_panel),
            ("2. Scores por persona", scores_panel),
            ("3. Reglas espaciales principales", rules_panel),
            ("4. Roles asignados", final_panel),
        ],
        columns=2,
    )
    path = save_png(Path(args.output_dir) / args.output_name, output)
    print(path)


if __name__ == "__main__":
    main()
