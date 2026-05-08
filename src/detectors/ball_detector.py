"""YOLO ball detector with tracker-backed contextual scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from ultralytics import YOLO

from ..models import ModelConfig


@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None
    predicted: bool = False

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class BallDetector:
    """Detect the ball with YOLO and keep the most plausible tracker output."""

    def __init__(self, model_path: str | None = None, confidence: float | None = None):
        self.model_path = model_path or ModelConfig.get_ball_model_path()
        self.accept_confidence = (
            confidence if confidence is not None else ModelConfig.BALL_ACCEPT_CONFIDENCE
        )
        self.model = YOLO(self.model_path)
        self._goal_bbox: tuple[int, int, int, int] | None = None
        self._shooter_bbox: tuple[int, int, int, int] | None = None
        self._shot_detected = False
        self._max_forward_progress = 0.0
        self.last_reject_reason: str | None = None
        self.last_real_detection: BallDetection | None = None
        self.reset()

    def reset(self) -> None:
        self.last_detection: BallDetection | None = None
        self.last_real_detection: BallDetection | None = None
        self.last_position: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.last_track_id: int | None = None
        self.missed_frames = 0
        self.last_reject_reason = None
        if not self._shot_detected:
            self._max_forward_progress = 0.0

    def set_goal_bbox(self, goal_bbox: tuple[int, int, int, int] | None) -> None:
        self._goal_bbox = goal_bbox

    def set_shooter_bbox(self, shooter_bbox: tuple[int, int, int, int] | None) -> None:
        self._shooter_bbox = shooter_bbox

    def set_shot_detected(self, detected: bool) -> None:
        if detected and not self._shot_detected:
            self._max_forward_progress = max(
                self._max_forward_progress,
                self._progress_for_center_without_frame(
                    self.last_real_detection.center if self.last_real_detection is not None else None
                ),
            )
        if not detected:
            self._max_forward_progress = 0.0
        self._shot_detected = detected

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        track_kwargs = {
            "imgsz": ModelConfig.BALL_IMGSZ,
            "persist": True,
            "tracker": ModelConfig.BALL_TRACKER,
            "conf": ModelConfig.BALL_CANDIDATE_CONFIDENCE,
            "classes": [ModelConfig.BALL_CLASS_ID],
            "verbose": False,
        }
        if ModelConfig.BALL_DEVICE is not None:
            track_kwargs["device"] = ModelConfig.BALL_DEVICE
        if ModelConfig.BALL_HALF:
            track_kwargs["half"] = True
        results = self.model.track(frame, **track_kwargs)

        detections = self._read_detections(results, frame.shape)
        if not detections:
            return self._handle_no_detection()

        selected = self._select_detection(detections, frame.shape)
        if selected is None:
            return self._handle_no_detection()

        self._update_state(selected)
        return selected

    def hold_last(self) -> BallDetection | None:
        return self._make_prediction(include_next_step=True)

    def track_without_detection(self) -> BallDetection | None:
        return self.hold_last()

    def increment_miss(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > self._max_missed_frames():
            self.reset()

    def _make_prediction(self, include_next_step: bool) -> BallDetection | None:
        if self.last_detection is None:
            return None
        if self._shot_detected:
            if self.missed_frames >= ModelConfig.BALL_MAX_PREDICTED_HOLD_POST_SHOT:
                return None
            if self.velocity is None or self._speed() < ModelConfig.BALL_MIN_POST_SHOT_SPEED:
                return None

        center = self._predict_center(include_next_step=include_next_step)
        if center is None:
            center = self.last_detection.center

        x1, y1, x2, y2 = self.last_detection.bbox_xyxy
        width = max(2, x2 - x1)
        height = max(2, y2 - y1)
        cx, cy = center
        return BallDetection(
            bbox_xyxy=(
                int(round(cx - width / 2.0)),
                int(round(cy - height / 2.0)),
                int(round(cx + width / 2.0)),
                int(round(cy + height / 2.0)),
            ),
            confidence=max(0.05, self.last_detection.confidence * 0.85),
            center=(float(cx), float(cy)),
            track_id=self.last_detection.track_id,
            predicted=True,
        )

    def _read_detections(self, results, frame_shape: tuple[int, ...]) -> list[BallDetection]:
        if not results or len(results[0].boxes) == 0:
            return []

        detections: list[BallDetection] = []
        boxes = results[0].boxes
        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy().astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            track_id = int(boxes.id[index].item()) if boxes.id is not None else None
            detection = BallDetection(
                bbox_xyxy=bbox,
                confidence=float(boxes.conf[index].item()),
                center=center,
                track_id=track_id,
            )
            if self._valid_geometry(detection, frame_shape):
                detections.append(detection)
        return detections

    def _select_detection(
        self,
        detections: list[BallDetection],
        frame_shape: tuple[int, ...],
    ) -> BallDetection | None:
        valid: list[BallDetection] = []
        rejected_reasons: list[str] = []
        for detection in detections:
            reason = self._candidate_is_hard_rejected(detection, frame_shape)
            if reason is None:
                valid.append(detection)
            else:
                rejected_reasons.append(reason)

        if not valid:
            self.last_reject_reason = rejected_reasons[0] if rejected_reasons else "no_valid_candidate"
            return None

        scored = [(self._score_detection(det, frame_shape), det) for det in valid]
        score, detection = max(scored, key=lambda item: item[0])
        if self._clearly_impossible_jump(detection) or score < self.accept_confidence:
            self.last_reject_reason = "low_score"
            return None
        self.last_reject_reason = None
        return detection

    def _score_detection(self, detection: BallDetection, frame_shape: tuple[int, ...]) -> float:
        score = detection.confidence

        if detection.track_id is not None and detection.track_id == self.last_track_id:
            score += 0.35

        score += self._prediction_score(detection)
        score += self._shape_score(detection)
        score += self._candidate_zone_score(detection, frame_shape)
        if self._shot_detected and self._has_forward_progress(detection, frame_shape):
            score += 0.20
        score -= self._jump_penalty(detection)
        return score

    def _valid_geometry(self, detection: BallDetection, frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            return False
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False

        frame_area = max(1, w * h)
        area_ratio = detection.area / frame_area
        if not (ModelConfig.BALL_MIN_AREA_RATIO <= area_ratio <= ModelConfig.BALL_MAX_AREA_RATIO):
            return False

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = width / height
        return ModelConfig.BALL_MIN_ASPECT_RATIO <= aspect <= ModelConfig.BALL_MAX_ASPECT_RATIO

    def _predict_center(self, include_next_step: bool = True) -> tuple[float, float] | None:
        if self.last_position is None:
            return None
        if self.velocity is None:
            return self.last_position
        steps = self.missed_frames + (1 if include_next_step else 0)
        steps = max(1, steps)
        return (
            self.last_position[0] + self.velocity[0] * steps,
            self.last_position[1] + self.velocity[1] * steps,
        )

    def _prediction_score(self, detection: BallDetection) -> float:
        predicted = self._predict_center()
        if predicted is None:
            return 0.0

        distance = self._distance(detection.center, predicted)
        gate = (
            ModelConfig.BALL_PREDICTION_GATE_BASE
            + self.missed_frames * ModelConfig.BALL_PREDICTION_GATE_PER_MISS
        )
        if distance <= gate:
            return 0.35 * (1.0 - distance / max(1.0, gate))
        return -min(0.45, (distance - gate) / max(1.0, gate) * 0.35)

    def _shape_score(self, detection: BallDetection) -> float:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = width / height
        aspect_error = abs(np.log(aspect))
        return max(-0.15, 0.10 - 0.12 * aspect_error)

    def _pre_shot_shooter_score(self, detection: BallDetection) -> float:
        if self._shooter_bbox is None:
            return 0.0

        sx1, sy1, sx2, sy2 = self._shooter_bbox
        shooter_h = max(1, sy2 - sy1)
        target = ((sx1 + sx2) / 2.0, sy2 - shooter_h * 0.08)
        radius = max(80.0, shooter_h * ModelConfig.BALL_PRE_SHOT_SHOOTER_RADIUS_RATIO)
        distance = self._distance(detection.center, target)
        if distance <= radius:
            return 0.30 * (1.0 - distance / radius)
        if distance > radius * 2.0:
            return -0.30
        return -0.15 * ((distance - radius) / radius)

    def _goal_path_score(self, detection: BallDetection) -> float:
        if self._shooter_bbox is None or self._goal_bbox is None:
            return 0.0

        sx1, sy1, sx2, sy2 = self._shooter_bbox
        gx1, gy1, gx2, gy2 = self._goal_bbox
        start = np.array([(sx1 + sx2) / 2.0, float(sy2)])
        end = np.array([(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0])
        point = np.array([detection.center[0], detection.center[1]])
        segment = end - start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1.0:
            return 0.0

        t = float(np.dot(point - start, segment) / seg_len_sq)
        closest_t = min(1.15, max(-0.15, t))
        closest = start + closest_t * segment
        distance = float(np.linalg.norm(point - closest))
        goal_w = max(1, gx2 - gx1)
        padding = max(100.0, goal_w * ModelConfig.BALL_GOAL_PATH_PADDING_RATIO)

        if -0.15 <= t <= 1.15 and distance <= padding:
            return 0.35 * (1.0 - distance / padding)
        if distance > padding * 2.0:
            return -0.35
        return -0.18 * ((distance - padding) / padding)

    def _penalty_origin(self, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        if self._shooter_bbox is not None:
            sx1, sy1, sx2, sy2 = self._shooter_bbox
            shooter_h = max(1, sy2 - sy1)
            return ((sx1 + sx2) / 2.0, sy2 - shooter_h * 0.08)
        if self.last_real_detection is not None:
            return self.last_real_detection.center
        h, w = frame_shape[:2]
        return (w / 2.0, h * 0.78)

    def _inside_goal_frame(self, candidate: BallDetection) -> bool:
        if self._goal_bbox is None:
            return False
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal_w = max(1, gx2 - gx1)
        goal_h = max(1, gy2 - gy1)
        x, y = candidate.center
        x_pad = goal_w * 0.06
        y_pad = goal_h * ModelConfig.BALL_REJECT_ABOVE_GOAL_MARGIN_RATIO
        return gx1 - x_pad <= x <= gx2 + x_pad and gy1 - y_pad <= y <= gy2 + y_pad

    def _is_behind_goal_noise(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        if not ModelConfig.BALL_REJECT_BEHIND_GOAL or self._goal_bbox is None:
            return False
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal_w = max(1, gx2 - gx1)
        goal_h = max(1, gy2 - gy1)
        x, y = candidate.center
        above_margin = goal_h * ModelConfig.BALL_REJECT_ABOVE_GOAL_MARGIN_RATIO
        lateral_pad = goal_w * 0.10

        if y < gy1 - above_margin:
            return True
        if y < gy2 - above_margin and not self._inside_goal_frame(candidate):
            return True
        if y < gy2 and not (gx1 - lateral_pad <= x <= gx2 + lateral_pad):
            return True
        return False

    def _has_forward_progress(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        progress = self._progress_ratio(candidate, frame_shape)
        return progress is not None and progress >= ModelConfig.BALL_POST_SHOT_MIN_PROGRESS_RATIO

    def _is_backtracking(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        if not self._shot_detected:
            return False
        progress = self._progress_ratio(candidate, frame_shape)
        if progress is None:
            return False
        tolerance = ModelConfig.BALL_POST_SHOT_BACKTRACK_TOLERANCE_RATIO
        return progress < self._max_forward_progress - tolerance

    def _candidate_zone_score(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> float:
        if self._shot_detected:
            predicted = self._predict_center()
            if predicted is not None:
                diag = self._frame_diag(frame_shape)
                radius = max(80.0, diag * ModelConfig.BALL_SEARCH_RADIUS_POST_SHOT_RATIO)
                distance = self._distance(candidate.center, predicted)
                score = 0.35 * max(-1.0, 1.0 - distance / radius)
            else:
                score = 0.0
            return score + self._goal_path_score(candidate)

        if self._shooter_bbox is None and self.last_real_detection is None:
            return 0.0
        origin = self._penalty_origin(frame_shape)
        radius = self._frame_diag(frame_shape) * ModelConfig.BALL_SEARCH_RADIUS_PRE_SHOT_RATIO
        distance = self._distance(candidate.center, origin)
        if distance <= radius:
            return 0.45 * (1.0 - distance / max(1.0, radius))
        return -0.60 * min(1.0, (distance - radius) / max(1.0, radius))

    def _candidate_is_hard_rejected(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> str | None:
        if not self._shot_detected:
            if self._shooter_bbox is None and self.last_real_detection is None:
                return None
            origin = self._penalty_origin(frame_shape)
            radius = self._frame_diag(frame_shape) * ModelConfig.BALL_SEARCH_RADIUS_PRE_SHOT_RATIO
            if self._distance(candidate.center, origin) > radius:
                return "zone"
            return None

        if self._is_behind_goal_noise(candidate, frame_shape):
            return "behind_goal"
        if self._is_backtracking(candidate, frame_shape):
            return "backtracking"

        origin = self._penalty_origin(frame_shape)
        spot_radius = (
            self._frame_diag(frame_shape)
            * ModelConfig.BALL_POST_SHOT_REJECT_PENALTY_SPOT_RADIUS_RATIO
        )
        progressed = self._max_forward_progress >= ModelConfig.BALL_POST_SHOT_MIN_PROGRESS_RATIO
        if progressed and self._distance(candidate.center, origin) <= spot_radius:
            return "penalty_spot"

        predicted = self._predict_center()
        if predicted is not None:
            search_radius = max(
                self._frame_diag(frame_shape) * ModelConfig.BALL_SEARCH_RADIUS_POST_SHOT_RATIO,
                ModelConfig.BALL_PREDICTION_GATE_BASE
                + self.missed_frames * ModelConfig.BALL_PREDICTION_GATE_PER_MISS,
            )
            if self._distance(candidate.center, predicted) > search_radius * 2.5:
                return "zone"
        return None

    def _progress_ratio(
        self,
        candidate: BallDetection,
        frame_shape: tuple[int, ...],
    ) -> float | None:
        if self._goal_bbox is None:
            return None
        origin = np.array(self._penalty_origin(frame_shape), dtype=float)
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal = np.array([(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0], dtype=float)
        point = np.array(candidate.center, dtype=float)
        segment = goal - origin
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1.0:
            return None
        return float(np.dot(point - origin, segment) / seg_len_sq)

    def _progress_for_center_without_frame(self, center: tuple[float, float] | None) -> float:
        if center is None or self._shooter_bbox is None or self._goal_bbox is None:
            return 0.0
        sx1, sy1, sx2, sy2 = self._shooter_bbox
        shooter_h = max(1, sy2 - sy1)
        origin = np.array([(sx1 + sx2) / 2.0, sy2 - shooter_h * 0.08], dtype=float)
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal = np.array([(gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0], dtype=float)
        point = np.array(center, dtype=float)
        segment = goal - origin
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1.0:
            return 0.0
        return max(0.0, float(np.dot(point - origin, segment) / seg_len_sq))

    def _jump_penalty(self, detection: BallDetection) -> float:
        if self.last_position is None:
            return 0.0

        jump = self._distance(detection.center, self.last_position)
        max_jump = (
            ModelConfig.BALL_MAX_JUMP_POST_SHOT
            if self._shot_detected
            else ModelConfig.BALL_MAX_JUMP_PRE_SHOT
        )
        if jump <= max_jump:
            return 0.0
        if jump > max_jump * 2.5:
            return 1.0
        return (jump - max_jump) / max(1.0, max_jump) * 0.45

    def _clearly_impossible_jump(self, detection: BallDetection) -> bool:
        if self.last_position is None:
            return False

        max_jump = (
            ModelConfig.BALL_MAX_JUMP_POST_SHOT
            if self._shot_detected
            else ModelConfig.BALL_MAX_JUMP_PRE_SHOT
        )
        return self._distance(detection.center, self.last_position) > max_jump * 3.0

    def _update_state(self, detection: BallDetection) -> None:
        if self.last_position is not None:
            self.velocity = (
                detection.center[0] - self.last_position[0],
                detection.center[1] - self.last_position[1],
            )

        self.last_detection = detection
        self.last_real_detection = detection
        self.last_position = detection.center
        self.last_track_id = detection.track_id
        self.missed_frames = 0
        if self._shot_detected:
            self._max_forward_progress = max(
                self._max_forward_progress,
                self._progress_for_center_without_frame(detection.center),
            )

    def _handle_no_detection(self) -> BallDetection | None:
        self.increment_miss()
        return self._make_prediction(include_next_step=False)

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return float(hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _frame_diag(frame_shape: tuple[int, ...]) -> float:
        h, w = frame_shape[:2]
        return float(hypot(w, h))

    def _speed(self) -> float:
        if self.velocity is None:
            return 0.0
        return float(hypot(self.velocity[0], self.velocity[1]))

    def _max_missed_frames(self) -> int:
        return (
            ModelConfig.BALL_MAX_MISSED_POST_SHOT
            if self._shot_detected
            else ModelConfig.BALL_MAX_MISSED_PRE_SHOT
        )
