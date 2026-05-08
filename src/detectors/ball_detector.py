"""Ball detection and tracking using YOLO with physics-aware re-acquisition."""

from __future__ import annotations

from dataclasses import dataclass

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
    """60fps-calibrated YOLO ball detector with physics gates."""

    def __init__(self, model_path: str | None = None, confidence: float | None = None):
        self.model_path = model_path or ModelConfig.get_ball_model_path()
        self.confidence = confidence if confidence is not None else ModelConfig.BALL_CONFIDENCE
        self.model = YOLO(self.model_path)

        self._goal_bbox: tuple[int, int, int, int] | None = None
        self._shooter_bbox: tuple[int, int, int, int] | None = None
        self._shot_detected = False
        self.last_reject_reason: str | None = None
        self.last_real_detection: BallDetection | None = None
        self.reset()

    def reset(self) -> None:
        self.last_position: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.last_detection: BallDetection | None = None
        self.last_real_detection: BallDetection | None = None
        self.last_track_id: int | None = None
        self.missed_frames = 0
        self._area_history: list[float] = []
        self._candidate_buffer: list[BallDetection] = []

    def set_goal_bbox(self, goal_bbox: tuple[int, int, int, int] | None) -> None:
        self._goal_bbox = goal_bbox

    def set_shooter_bbox(self, shooter_bbox: tuple[int, int, int, int] | None) -> None:
        self._shooter_bbox = shooter_bbox

    def set_shot_detected(self, detected: bool) -> None:
        self._shot_detected = detected

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        h = frame.shape[0]
        self.last_reject_reason = None
        results = self.model.track(
            frame,
            imgsz=ModelConfig.BALL_IMGSZ,
            persist=True,
            tracker=ModelConfig.BALL_TRACKER,
            conf=self.confidence,
            classes=[ModelConfig.BALL_CLASS_ID],
            verbose=False,
            **self._device_kwargs(),
        )

        if not results or len(results[0].boxes) == 0:
            return self._handle_no_detection()

        boxes = results[0].boxes
        candidates: list[tuple[BallDetection, int | None]] = []
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
            if self._is_valid(detection, frame.shape):
                candidates.append((detection, track_id))

        if not candidates:
            return self._handle_no_detection()

        if self.last_track_id is not None:
            for detection, track_id in candidates:
                if track_id == self.last_track_id and self._area_consistent(detection):
                    self._candidate_buffer.clear()
                    self._update_state(detection)
                    return detection

        pred = self._predict()
        ellipse = self._search_ellipse()
        if ellipse is not None:
            in_ellipse = [
                detection
                for detection, _ in candidates
                if self._in_ellipse(detection.center, ellipse)
            ]
            if in_ellipse:
                candidates = [
                    (detection, track_id)
                    for detection, track_id in candidates
                    if detection in in_ellipse
                ]

        max_dist = self._adaptive_max_dist()
        valid: list[BallDetection] = []
        for detection, _ in candidates:
            if pred is None:
                valid.append(detection)
            else:
                dist = self._distance(detection.center, pred)
                if dist < max_dist and self._area_consistent(detection):
                    valid.append(detection)

        if not valid and pred is not None:
            valid = [
                detection
                for detection, _ in candidates
                if self._distance(detection.center, pred) < max_dist * ModelConfig.BALL_RELAXED_DIST_MULTIPLIER
            ]

        if not valid:
            return self._handle_no_detection()

        best = min(valid, key=lambda detection: self._score(detection, pred, h))
        if self.missed_frames >= ModelConfig.BALL_REACQ_MIN_MISSED:
            confirmed = self._confirm_reacquisition(best)
            if confirmed is None:
                self.last_reject_reason = "reacq_confirm"
                return self._handle_no_detection()
            best = confirmed

        self._candidate_buffer.clear()
        self._update_state(best)
        return best

    def hold_last(self) -> BallDetection | None:
        return self._prediction_detection(increment_miss=False)

    def track_without_detection(self) -> BallDetection | None:
        return self.hold_last()

    def increment_miss(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > self._max_missed_frames():
            self.reset()

    def _is_valid(self, detection: BallDetection, shape: tuple[int, ...]) -> bool:
        hard_reject = self._hard_reject_reason(detection, shape)
        if hard_reject is not None:
            self.last_reject_reason = hard_reject
            return False
        return (
            self._size_ok(detection)
            and self._aspect_ratio_ok(detection, shape)
            and self._position_ok(detection.center, shape)
            and self._in_roi(detection.center)
        )

    def _size_ok(self, detection: BallDetection) -> bool:
        return ModelConfig.BALL_MIN_AREA < detection.area < ModelConfig.BALL_MAX_AREA

    def _aspect_ratio_ok(self, detection: BallDetection, shape: tuple[int, ...]) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        ratio = max(width, height) / min(width, height)
        if self._inside_goal_frame(detection):
            return ratio <= ModelConfig.BALL_MAX_ASPECT_RATIO
        if detection.center[1] < shape[0] * ModelConfig.BALL_NEAR_GOAL_Y_FRACTION:
            return ratio <= ModelConfig.BALL_NEAR_GOAL_MAX_ASPECT
        return ratio <= ModelConfig.BALL_MAX_ASPECT_RATIO

    @staticmethod
    def _position_ok(center: tuple[float, float], shape: tuple[int, ...]) -> bool:
        x, y = center
        h, w = shape[:2]
        return 0.02 * w < x < 0.98 * w and 0.05 * h < y < 0.98 * h

    def _in_roi(self, center: tuple[float, float]) -> bool:
        if self.last_position is None:
            return True
        if self._center_inside_goal_frame(center):
            return True
        dx = abs(center[0] - self.last_position[0])
        dy = abs(center[1] - self.last_position[1])
        radius = max(ModelConfig.BALL_ROI_RADIUS, self._adaptive_max_dist() * 1.15)
        return dx < radius and dy < radius

    def _hard_reject_reason(
        self,
        detection: BallDetection,
        shape: tuple[int, ...],
    ) -> str | None:
        if self._is_behind_goal_noise(detection):
            return "behind_goal"
        if self._shot_detected and self._near_penalty_spot_after_shot(detection, shape):
            return "penalty_spot"
        return None

    def _is_behind_goal_noise(self, detection: BallDetection) -> bool:
        if not ModelConfig.BALL_REJECT_BEHIND_GOAL or self._goal_bbox is None:
            return False
        if self._inside_goal_frame(detection):
            return False
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal_w = max(1, gx2 - gx1)
        goal_h = max(1, gy2 - gy1)
        x, y = detection.center
        top_margin = goal_h * ModelConfig.BALL_REJECT_ABOVE_GOAL_MARGIN_RATIO
        lateral_margin = goal_w * ModelConfig.BALL_GOAL_LATERAL_MARGIN_RATIO

        if y < gy1 - top_margin:
            return True
        if y < gy2 - top_margin and not (gx1 <= x <= gx2):
            return True
        if y < gy2 and not (gx1 - lateral_margin <= x <= gx2 + lateral_margin):
            return True
        return False

    def _inside_goal_frame(self, detection: BallDetection) -> bool:
        return self._center_inside_goal_frame(detection.center)

    def _center_inside_goal_frame(self, center: tuple[float, float]) -> bool:
        if self._goal_bbox is None:
            return False
        gx1, gy1, gx2, gy2 = self._goal_bbox
        goal_w = max(1, gx2 - gx1)
        goal_h = max(1, gy2 - gy1)
        x, y = center
        x_pad = goal_w * 0.08
        y_pad = goal_h * ModelConfig.BALL_REJECT_ABOVE_GOAL_MARGIN_RATIO
        return gx1 - x_pad <= x <= gx2 + x_pad and gy1 - y_pad <= y <= gy2 + y_pad

    def _near_penalty_spot_after_shot(
        self,
        detection: BallDetection,
        shape: tuple[int, ...],
    ) -> bool:
        if self.velocity is None:
            return False
        speed = self._speed()
        if speed < ModelConfig.BALL_MIN_POST_SHOT_SPEED:
            return False
        origin = self._penalty_origin(shape)
        radius = self._frame_diag(shape) * ModelConfig.BALL_POST_SHOT_REJECT_PENALTY_SPOT_RADIUS_RATIO
        return self._distance(detection.center, origin) <= radius

    def _penalty_origin(self, shape: tuple[int, ...]) -> tuple[float, float]:
        if self._shooter_bbox is not None:
            sx1, sy1, sx2, sy2 = self._shooter_bbox
            shooter_h = max(1, sy2 - sy1)
            return ((sx1 + sx2) / 2.0, sy2 - shooter_h * 0.08)
        if self.last_real_detection is not None:
            return self.last_real_detection.center
        h, w = shape[:2]
        return (w / 2.0, h * 0.78)

    def _search_ellipse(self) -> tuple[float, float, float, float, float] | None:
        if self.last_position is None or self.velocity is None:
            return None
        n = max(1, self.missed_frames)
        decay = ModelConfig.BALL_DECEL_PER_MISS ** n
        vx, vy = self.velocity
        cx = self.last_position[0] + vx * n * decay
        cy = self.last_position[1] + vy * n * decay
        speed = float(np.sqrt(vx**2 + vy**2))
        r_along = ModelConfig.BALL_ELLIPSE_R_ALONG_BASE + speed * 0.4 * n
        r_perp = ModelConfig.BALL_ELLIPSE_R_PERP_BASE + ModelConfig.BALL_ELLIPSE_R_PERP_GROW * n
        angle = float(np.arctan2(vy, vx))
        return (cx, cy, r_along, r_perp, angle)

    @staticmethod
    def _in_ellipse(
        center: tuple[float, float],
        ellipse: tuple[float, float, float, float, float],
    ) -> bool:
        cx, cy, r_along, r_perp, angle = ellipse
        dx = center[0] - cx
        dy = center[1] - cy
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx_r = dx * cos_a + dy * sin_a
        dy_r = -dx * sin_a + dy * cos_a
        val = (dx_r / max(1, r_along)) ** 2 + (dy_r / max(1, r_perp)) ** 2
        return val <= 1.0

    def _confirm_reacquisition(self, candidate: BallDetection) -> BallDetection | None:
        self._candidate_buffer.append(candidate)
        if len(self._candidate_buffer) < ModelConfig.BALL_REACQ_CONFIRM_FRAMES:
            return None

        positions = [candidate.center for candidate in self._candidate_buffer]
        max_drift = max(
            self._distance(p1, p2)
            for p1, p2 in zip(positions, positions[1:])
        )
        buffer = self._candidate_buffer.copy()
        self._candidate_buffer.clear()
        if max_drift < ModelConfig.BALL_REACQ_MAX_DRIFT:
            return buffer[-1]
        return None

    def _predict(self) -> tuple[float, float] | None:
        if self.last_position is None:
            return None
        if self.velocity is None:
            return self.last_position
        return (
            self.last_position[0] + self.velocity[0],
            self.last_position[1] + self.velocity[1],
        )

    def _update_state(self, detection: BallDetection) -> None:
        new_pos = detection.center
        if self.last_position is not None:
            dvx = new_pos[0] - self.last_position[0]
            dvy = new_pos[1] - self.last_position[1]
            if self.velocity is None:
                self.velocity = (dvx, dvy)
            else:
                alpha = ModelConfig.BALL_VELOCITY_EMA_ALPHA
                self.velocity = (
                    (1 - alpha) * self.velocity[0] + alpha * dvx,
                    (1 - alpha) * self.velocity[1] + alpha * dvy,
                )

        self.last_position = new_pos
        self.last_detection = detection
        self.last_real_detection = detection
        self.last_track_id = detection.track_id
        self.missed_frames = 0
        self._area_history.append(detection.area)
        if len(self._area_history) > 8:
            self._area_history.pop(0)

    def _handle_no_detection(self) -> BallDetection | None:
        return self._prediction_detection(increment_miss=True)

    def _prediction_detection(self, increment_miss: bool) -> BallDetection | None:
        if increment_miss:
            self.missed_frames += 1

        if self._shot_detected and self.missed_frames > ModelConfig.BALL_MAX_PREDICTED_HOLD_POST_SHOT:
            return None
        if (
            self.last_position is None
            or self.velocity is None
            or self.last_detection is None
            or self.missed_frames > self._max_missed_frames()
        ):
            if self.missed_frames > self._max_missed_frames():
                self.reset()
            return None
        if self._shot_detected and self._speed() < ModelConfig.BALL_MIN_POST_SHOT_SPEED:
            return None

        pred = self._predict()
        if pred is None:
            return None

        if increment_miss:
            self.last_position = pred
        x1, y1, x2, y2 = self.last_detection.bbox_xyxy
        width = max(2, x2 - x1)
        height = max(2, y2 - y1)
        px, py = float(pred[0]), float(pred[1])
        return BallDetection(
            bbox_xyxy=(
                int(px - width / 2),
                int(py - height / 2),
                int(px + width / 2),
                int(py + height / 2),
            ),
            confidence=max(0.05, self.last_detection.confidence * 0.70),
            center=(px, py),
            track_id=self.last_track_id,
            predicted=True,
        )

    def _score(self, detection: BallDetection, pred: tuple[float, float] | None, frame_h: int) -> float:
        dist = 0.0
        if pred is not None:
            dx = detection.center[0] - pred[0]
            dy = detection.center[1] - pred[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if self.velocity is not None and dx * self.velocity[0] + dy * self.velocity[1] < 0:
                dist *= 2.0

        area_penalty = 0.0
        if self.last_detection is not None:
            area_penalty = abs(
                float(np.log((detection.area + 1.0) / (self.last_detection.area + 1.0)))
            )

        return (
            dist
            + 45.0 * area_penalty
            - 20.0 * detection.confidence
            + self._direction_penalty(detection)
            + self._goal_zone_penalty(detection, frame_h)
            + self._continuity_penalty(detection)
        )

    def _continuity_penalty(self, detection: BallDetection) -> float:
        if self.velocity is None or self.last_position is None or self.missed_frames == 0:
            return 0.0
        n = self.missed_frames
        decay = ModelConfig.BALL_DECEL_PER_MISS ** n
        pred_x = self.last_position[0] + self.velocity[0] * n * decay
        pred_y = self.last_position[1] + self.velocity[1] * n * decay
        dev = self._distance(detection.center, (pred_x, pred_y))
        speed = self._speed()
        allowed = speed * 0.5 * n + 40
        if dev > allowed * 2:
            return 300.0
        if dev > allowed:
            return 80.0
        return 0.0

    def _direction_penalty(self, detection: BallDetection) -> float:
        if self.velocity is None or self.last_position is None:
            return 0.0
        speed = self._speed()
        if speed < ModelConfig.BALL_MIN_SPEED_FOR_DIR_FILTER:
            return 0.0
        dx = detection.center[0] - self.last_position[0]
        dy = detection.center[1] - self.last_position[1]
        candidate_speed = float(np.sqrt(dx * dx + dy * dy))
        if candidate_speed < 1e-3:
            return 0.0
        cos_theta = (self.velocity[0] * dx + self.velocity[1] * dy) / (speed * candidate_speed)
        if cos_theta < -0.5:
            return 120.0
        if cos_theta < 0.0:
            return 40.0
        return 0.0

    def _goal_zone_penalty(self, detection: BallDetection, frame_h: int) -> float:
        if detection.center[1] >= frame_h * ModelConfig.BALL_NEAR_GOAL_Y_FRACTION:
            return 0.0
        if self._inside_goal_frame(detection):
            return 0.0
        penalty = 0.0
        if detection.area > ModelConfig.BALL_NEAR_GOAL_MAX_AREA:
            penalty += 80.0
        if detection.confidence < ModelConfig.BALL_NEAR_GOAL_MIN_CONF:
            penalty += 60.0
        if self._goal_bbox is not None:
            gx1, _, gx2, _ = self._goal_bbox
            margin = (gx2 - gx1) * ModelConfig.BALL_GOAL_LATERAL_MARGIN_RATIO
            if detection.center[0] < gx1 - margin or detection.center[0] > gx2 + margin:
                penalty += 100.0
        return penalty

    def _area_consistent(self, detection: BallDetection) -> bool:
        reference = self.last_real_detection or self.last_detection
        if reference is None:
            return True
        last_area = max(1.0, float(reference.area))
        ratio = float(detection.area) / last_area
        max_growth = min(ModelConfig.BALL_AREA_RATIO_MAX + 0.10 * self.missed_frames, 2.2)
        min_shrink = max(ModelConfig.BALL_AREA_RATIO_MIN - 0.02 * self.missed_frames, 0.4)
        return min_shrink <= ratio <= max_growth

    def _adaptive_max_dist(self) -> float:
        speed = self._speed()
        dist = (
            ModelConfig.BALL_BASE_MAX_DIST
            + 0.9 * speed
            + self.missed_frames * ModelConfig.BALL_MAX_DIST_PER_MISS
        )
        return min(ModelConfig.BALL_MAX_ADAPTIVE_DIST, dist)

    def _max_missed_frames(self) -> int:
        return (
            ModelConfig.BALL_MAX_MISSED_POST_SHOT
            if self._shot_detected
            else ModelConfig.BALL_MAX_MISSED_PRE_SHOT
        )

    def _speed(self) -> float:
        if self.velocity is None:
            return 0.0
        return float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))

    @staticmethod
    def _frame_diag(shape: tuple[int, ...]) -> float:
        h, w = shape[:2]
        return float(np.sqrt(w * w + h * h))

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(np.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _device_kwargs() -> dict:
        kwargs = {}
        if ModelConfig.BALL_DEVICE is not None:
            kwargs["device"] = ModelConfig.BALL_DEVICE
        if ModelConfig.BALL_HALF:
            kwargs["half"] = True
        return kwargs
