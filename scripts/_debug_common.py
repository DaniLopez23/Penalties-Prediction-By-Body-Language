from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG, ModelConfig, PipelineConfig  # noqa: E402
from src.models import Detection, GoalBox  # noqa: E402


DEFAULT_VIDEO = PROJECT_ROOT / "data" / "penalties_mbappe_1.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "debug_frames"


def default_detector_model() -> str:
    for name in ("yolo11s.pt", "yolo11m.pt", "yolov8n.pt"):
        path = PROJECT_ROOT / name
        if path.exists():
            return str(path)
    return "yolov8n.pt"


def build_config(args: argparse.Namespace) -> PipelineConfig:
    model_cfg = replace(
        DEFAULT_CONFIG.models,
        detector_model=str(args.model),
        detector_confidence=args.detector_confidence,
        person_confidence=args.person_confidence,
        ball_confidence=args.ball_confidence,
        image_size=args.imgsz,
        device=args.device,
    )
    return replace(DEFAULT_CONFIG, models=model_cfg)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Ruta al video o imagen de entrada.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=60,
        help="Indice de frame a extraer si la entrada es un video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde guardar los PNG de diagnostico.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(default_detector_model()),
        help="Modelo YOLO para personas/balon.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Tamano de inferencia YOLO.")
    parser.add_argument("--device", default=None, help="Dispositivo YOLO, por ejemplo cpu, 0, cuda.")
    parser.add_argument("--detector-confidence", type=float, default=0.08)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--ball-confidence", type=float, default=0.08)


def read_frame(source: Path, frame_index: int) -> tuple[np.ndarray, int]:
    if not source.exists():
        raise FileNotFoundError(f"No existe la entrada: {source}")

    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"No se pudo leer la imagen: {source}")
        return frame, 0

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        safe_index = max(0, frame_index)
        if total > 0:
            safe_index = min(safe_index, total - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, safe_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"No se pudo extraer el frame {safe_index} de {source}")
        return frame, safe_index
    finally:
        capture.release()


def read_neighbor_frames(source: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray, int]:
    current, used_index = read_frame(source, frame_index)
    previous_index = max(0, used_index - 1)
    previous, _ = read_frame(source, previous_index)
    return previous, current, used_index


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png(path: Path, frame: np.ndarray) -> Path:
    ensure_output_dir(path.parent)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"No se pudo guardar la imagen: {path}")
    return path


def overlay_mask(frame: np.ndarray, mask: Optional[np.ndarray], color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    output = frame.copy()
    if mask is None:
        return output
    color_layer = np.zeros_like(output)
    color_layer[:, :] = color
    output = np.where(mask[:, :, None] > 0, cv2.addWeighted(output, 1.0 - alpha, color_layer, alpha, 0), output)
    return output


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (25, 25, 25),
    scale: float = 0.5,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(0, y - th - baseline - 5)
    cv2.rectangle(frame, (x, y1), (x + tw + 8, y + 3), bg, -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: Optional[str] = None,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        draw_label(frame, label, (x1, max(18, y1)), color=color)


def draw_goal(frame: np.ndarray, goal: Optional[GoalBox], label: str = "goal") -> None:
    if goal is None:
        draw_label(frame, "goal: no detectada", (16, 28), color=(80, 80, 255))
        return
    status = "detectada" if goal.detected else "fallback"
    draw_box(
        frame,
        (goal.x1, goal.y1, goal.x2, goal.y2),
        (70, 220, 255) if goal.detected else (120, 120, 255),
        f"{label}: {status} conf={goal.confidence:.2f}",
        2,
    )
    for zone, bounds in goal.zone_bounds().items():
        x1, y1, x2, y2 = bounds
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 130, 240), 1, cv2.LINE_AA)
        draw_label(frame, zone, (x1 + 4, y2 - 4), color=(40, 130, 240), scale=0.42)


def tile_images(images: Iterable[tuple[str, np.ndarray]], columns: int = 2, max_width: int = 720) -> np.ndarray:
    items = list(images)
    if not items:
        raise ValueError("No hay imagenes para componer.")

    resized: list[np.ndarray] = []
    for title, image in items:
        panel = image.copy()
        scale = min(1.0, max_width / max(1, panel.shape[1]))
        if scale < 1.0:
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(panel, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(panel)

    cell_w = max(img.shape[1] for img in resized)
    cell_h = max(img.shape[0] for img in resized)
    rows = int(np.ceil(len(resized) / columns))
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 18, dtype=np.uint8)
    for index, panel in enumerate(resized):
        row = index // columns
        col = index % columns
        y = row * cell_h
        x = col * cell_w
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def split_detections(detections: list[Detection], config: PipelineConfig) -> tuple[list[Detection], list[Detection]]:
    persons = [det for det in detections if det.class_id == config.models.coco_person_class_id]
    balls = [det for det in detections if det.class_id == config.models.coco_ball_class_id]
    return persons, balls
