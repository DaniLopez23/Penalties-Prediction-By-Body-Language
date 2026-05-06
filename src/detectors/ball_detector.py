"""Ball detection + tracking using YOLO with physics-aware re-acquisition."""

import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO
from ..models import ModelConfig


@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class BallDetector:
    """
    Ball detector optimised for 1080p / 60fps penalty footage.

    Changes vs original
    -------------------
    1. Parameters scaled for 60fps — BASE_MAX_DIST, ROI_RADIUS and speed
       thresholds are roughly 1.4× the 30fps values.
    2. Physics-constrained search ellipse: instead of a growing circle,
       re-acquisition searches an ellipse aligned to the last velocity vector,
       tight perpendicular to motion, wider along it.
    3. Trajectory continuity gate: candidates that imply an impossible
       acceleration after a miss are penalised heavily.
    4. N-frame confirmation buffer: after ≥3 missed frames a new candidate
       must appear consistently in 2 consecutive frames before being accepted.
    5. Tighter area relaxation: area ratio grows more slowly with missed frames
       and is hard-capped at 2.2× to block hands and boots.
    6. Split MAX_MISSED_FRAMES: generous pre-shot, strict post-shot.
    7. Near-goal filters tightened: ASPECT 1.30, CONF 0.35, AREA 4000.
    8. Velocity EMA alpha reduced 0.30 → 0.20 for smoother trajectory.
    """

    # ── Motion / tracking (60fps-calibrated) ─────────────────────────────────
    BASE_MAX_DIST           = 280   # was 200
    MAX_DIST_PER_MISS       = 30    # was 40
    ROI_RADIUS              = 320   # was 260
    MAX_MISSED_PRE_SHOT     = 12    # generous before kick
    MAX_MISSED_POST_SHOT    = 5     # strict after ball is moving fast
    MIN_SPEED_FOR_DIR_FILTER = 14.0 # was 8.0 — scaled for 60fps px/frame
    VELOCITY_EMA_ALPHA      = 0.20  # was 0.30

    # ── Ball size ─────────────────────────────────────────────────────────────
    MIN_AREA       = 20     # was 12
    MAX_AREA       = 8000   # was 9000
    AREA_RATIO_MIN = 0.55   # was 0.45
    AREA_RATIO_MAX = 1.90   # was 2.4

    # ── Shape ─────────────────────────────────────────────────────────────────
    MAX_ASPECT_RATIO = 1.6  # was 1.8

    # ── Near-goal zone (tightened) ────────────────────────────────────────────
    NEAR_GOAL_Y_FRACTION = 0.45
    NEAR_GOAL_MAX_AREA   = 4000    # was 5500
    NEAR_GOAL_MIN_CONF   = 0.35    # was 0.25
    NEAR_GOAL_MAX_ASPECT = 1.30    # was 1.45

    # ── Re-acquisition confirmation ───────────────────────────────────────────
    REACQ_MIN_MISSED       = 3     # gaps shorter than this: trust immediately
    REACQ_CONFIRM_FRAMES   = 2     # frames needed to confirm re-acquisition
    REACQ_MAX_DRIFT        = 60    # px — candidates must be spatially consistent

    # ── Physics projection ────────────────────────────────────────────────────
    DECEL_PER_MISS         = 0.92  # speed multiplier per missed frame (air drag)
    ELLIPSE_R_ALONG_BASE   = 60    # px base uncertainty along velocity axis
    ELLIPSE_R_PERP_BASE    = 40    # px base uncertainty perpendicular
    ELLIPSE_R_PERP_GROW    = 8     # px per missed frame (perpendicular uncertainty)

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ):
        if model_path is None:
            model_path = ModelConfig.get_ball_model_path()
        if confidence is None:
            confidence = ModelConfig.BALL_CONFIDENCE

        self.model      = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence

        self._goal_bbox: tuple[int, int, int, int] | None = None
        self._shot_detected: bool = False

        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.last_position:  tuple[float, float] | None = None
        self.velocity:       tuple[float, float] | None = None
        self.last_detection: BallDetection | None       = None
        self.missed_frames:  int                        = 0
        self.last_track_id:  int | None                 = None
        self._area_history:  list[float]                = []
        self._candidate_buffer: list[BallDetection]     = []

    def set_goal_bbox(self, goal_bbox: tuple | None) -> None:
        self._goal_bbox = goal_bbox

    def set_shot_detected(self, detected: bool) -> None:
        """Pipeline notifies us once the shot is confirmed."""
        self._shot_detected = detected

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        h, w = frame.shape[:2]

        results = self.model.track(
            frame,
            imgsz=1280,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=[32],
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return self._handle_no_detection()

        boxes = results[0].boxes
        candidates: list[tuple[BallDetection, int | None]] = []

        for i in range(len(boxes)):
            coords   = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf     = float(boxes.conf[i])
            center   = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None

            det = BallDetection((x1, y1, x2, y2), conf, center, track_id)
            if self._is_valid(det, frame.shape):
                candidates.append((det, track_id))

        if not candidates:
            return self._handle_no_detection()

        # ── 1. Same track_id as last frame → fast-path ─────────────────────
        if self.last_track_id is not None:
            for det, tid in candidates:
                if tid == self.last_track_id and self._area_consistent(det):
                    self._candidate_buffer.clear()
                    self._update_state(det)
                    return det

        pred = self._predict()

        # ── 2. Physics ellipse filter ──────────────────────────────────────
        ellipse = self._search_ellipse()
        if ellipse is not None:
            in_ellipse = [
                d for d, _ in candidates
                if self._in_ellipse(d.center, ellipse)
            ]
            if in_ellipse:
                candidates = [(d, t) for d, t in candidates if d in in_ellipse]

        # ── 3. Standard distance + area gate ──────────────────────────────
        max_dist = self._adaptive_max_dist()
        valid: list[BallDetection] = []
        for det, _ in candidates:
            if pred is None:
                valid.append(det)
            else:
                dist = self._distance(det.center, pred)
                if dist < max_dist and self._area_consistent(det):
                    valid.append(det)

        # Soft fallback
        if not valid:
            if pred is not None:
                relaxed = [
                    d for d, _ in candidates
                    if self._distance(d.center, pred) < max_dist * 1.6
                ]
                valid = relaxed if relaxed else []
            if not valid:
                return self._handle_no_detection()

        # ── 4. Score candidates ────────────────────────────────────────────
        best = min(valid, key=lambda d: self._score(d, pred, h))

        # ── 5. Re-acquisition confirmation buffer ──────────────────────────
        if self.missed_frames >= self.REACQ_MIN_MISSED:
            confirmed = self._confirm_reacquisition(best)
            if confirmed is None:
                return self._handle_no_detection()
            best = confirmed

        self._candidate_buffer.clear()
        self._update_state(best)
        return best

    # ── Validation ────────────────────────────────────────────────────────────

    def _is_valid(self, det: BallDetection, shape: tuple) -> bool:
        return (
            self._size_ok(det)
            and self._aspect_ratio_ok(det, shape)
            and self._position_ok(det.center, shape)
            and self._in_roi(det.center)
        )

    def _size_ok(self, det: BallDetection) -> bool:
        return self.MIN_AREA < det.area < self.MAX_AREA

    def _aspect_ratio_ok(self, det: BallDetection, shape: tuple) -> bool:
        x1, y1, x2, y2 = det.bbox_xyxy
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        ratio = max(bw, bh) / min(bw, bh)
        if det.center[1] < shape[0] * self.NEAR_GOAL_Y_FRACTION:
            return ratio <= self.NEAR_GOAL_MAX_ASPECT
        return ratio <= self.MAX_ASPECT_RATIO

    def _position_ok(self, center: tuple, shape: tuple) -> bool:
        x, y = center
        h, w = shape[:2]
        return 0.02 * w < x < 0.98 * w and 0.05 * h < y < 0.98 * h

    def _in_roi(self, center: tuple) -> bool:
        if self.last_position is None:
            return True
        dx = abs(center[0] - self.last_position[0])
        dy = abs(center[1] - self.last_position[1])
        r  = max(self.ROI_RADIUS, self._adaptive_max_dist() * 1.15)
        return dx < r and dy < r

    # ── Physics projection & ellipse ──────────────────────────────────────────

    def _search_ellipse(self) -> tuple | None:
        """Return (cx, cy, r_along, r_perp, angle_rad) or None."""
        if self.last_position is None or self.velocity is None:
            return None
        n = max(1, self.missed_frames)
        decay = self.DECEL_PER_MISS ** n
        vx, vy = self.velocity
        cx = self.last_position[0] + vx * n * decay
        cy = self.last_position[1] + vy * n * decay
        speed   = float(np.sqrt(vx ** 2 + vy ** 2))
        r_along = self.ELLIPSE_R_ALONG_BASE + speed * 0.4 * n
        r_perp  = self.ELLIPSE_R_PERP_BASE  + self.ELLIPSE_R_PERP_GROW * n
        angle   = float(np.arctan2(vy, vx))
        return (cx, cy, r_along, r_perp, angle)

    def _in_ellipse(self, center: tuple, ellipse: tuple) -> bool:
        cx, cy, r_along, r_perp, angle = ellipse
        dx = center[0] - cx
        dy = center[1] - cy
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotate to ellipse frame
        dx_r =  dx * cos_a + dy * sin_a
        dy_r = -dx * sin_a + dy * cos_a
        val = (dx_r / max(1, r_along)) ** 2 + (dy_r / max(1, r_perp)) ** 2
        return val <= 1.0

    # ── Re-acquisition confirmation buffer ────────────────────────────────────

    def _confirm_reacquisition(self, candidate: BallDetection) -> BallDetection | None:
        """Require REACQ_CONFIRM_FRAMES consistent detections before committing."""
        self._candidate_buffer.append(candidate)
        if len(self._candidate_buffer) < self.REACQ_CONFIRM_FRAMES:
            return None  # not enough evidence yet

        # Check spatial consistency between buffered candidates
        positions = [c.center for c in self._candidate_buffer]
        max_drift = max(
            self._distance(p1, p2)
            for p1, p2 in zip(positions, positions[1:])
        )
        buf = self._candidate_buffer.copy()
        self._candidate_buffer.clear()

        if max_drift < self.REACQ_MAX_DRIFT:
            return buf[-1]   # consistent — accept
        return None           # jumping around — reject

    # ── Tracking & prediction ─────────────────────────────────────────────────

    def _predict(self) -> tuple | None:
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
                a = self.VELOCITY_EMA_ALPHA
                self.velocity = (
                    (1 - a) * self.velocity[0] + a * dvx,
                    (1 - a) * self.velocity[1] + a * dvy,
                )

        self.last_position  = new_pos
        self.last_detection = detection
        self.last_track_id  = detection.track_id
        self.missed_frames  = 0

        self._area_history.append(detection.area)
        if len(self._area_history) > 8:
            self._area_history.pop(0)

    def _handle_no_detection(self) -> BallDetection | None:
        max_miss = (
            self.MAX_MISSED_POST_SHOT if self._shot_detected
            else self.MAX_MISSED_PRE_SHOT
        )
        if (
            self.last_position is not None
            and self.velocity is not None
            and self.missed_frames < max_miss
        ):
            self.missed_frames += 1
            pred = self._predict()
            self.last_position = pred
            if self.last_detection is not None:
                x1, y1, x2, y2 = self.last_detection.bbox_xyxy
                bw = max(2, x2 - x1)
                bh = max(2, y2 - y1)
                px, py = float(pred[0]), float(pred[1])
                return BallDetection(
                    bbox_xyxy=(
                        int(px - bw / 2), int(py - bh / 2),
                        int(px + bw / 2), int(py + bh / 2),
                    ),
                    confidence=max(0.05, self.last_detection.confidence * 0.7),
                    center=(px, py),
                )
            return BallDetection(bbox_xyxy=(0,0,0,0), confidence=0.0, center=pred)

        self.missed_frames += 1
        if self.missed_frames > max_miss:
            self.reset()
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, det: BallDetection, pred: tuple | None, frame_h: int) -> float:
        dist = 0.0
        if pred is not None:
            dx = det.center[0] - pred[0]
            dy = det.center[1] - pred[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if self.velocity is not None:
                if dx * self.velocity[0] + dy * self.velocity[1] < 0:
                    dist *= 2.0

        continuity = self._continuity_penalty(det)

        area_penalty = 0.0
        if self.last_detection is not None:
            area_penalty = abs(
                float(np.log((det.area + 1.0) / (self.last_detection.area + 1.0)))
            )

        goal_penalty = self._goal_zone_penalty(det, frame_h)

        return (
            dist
            + 45.0 * area_penalty
            - 20.0 * det.confidence
            + self._direction_penalty(det)
            + goal_penalty
            + continuity
        )

    def _continuity_penalty(self, det: BallDetection) -> float:
        """Heavy penalty for candidates that imply impossible jumps after a miss."""
        if self.velocity is None or self.last_position is None or self.missed_frames == 0:
            return 0.0
        n = self.missed_frames
        decay = self.DECEL_PER_MISS ** n
        pred_x = self.last_position[0] + self.velocity[0] * n * decay
        pred_y = self.last_position[1] + self.velocity[1] * n * decay
        dev = self._distance(det.center, (pred_x, pred_y))
        speed = float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))
        allowed = speed * 0.5 * n + 40
        if dev > allowed * 2:
            return 300.0
        if dev > allowed:
            return 80.0
        return 0.0

    def _direction_penalty(self, det: BallDetection) -> float:
        if self.velocity is None or self.last_position is None:
            return 0.0
        speed = float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))
        if speed < self.MIN_SPEED_FOR_DIR_FILTER:
            return 0.0
        dx = det.center[0] - self.last_position[0]
        dy = det.center[1] - self.last_position[1]
        cs = float(np.sqrt(dx * dx + dy * dy))
        if cs < 1e-3:
            return 0.0
        cos_t = (self.velocity[0] * dx + self.velocity[1] * dy) / (speed * cs)
        if cos_t < -0.5:
            return 120.0
        if cos_t < 0.0:
            return 40.0
        return 0.0

    def _goal_zone_penalty(self, det: BallDetection, frame_h: int) -> float:
        cy = det.center[1]
        if cy >= frame_h * self.NEAR_GOAL_Y_FRACTION:
            return 0.0
        penalty = 0.0
        if det.area > self.NEAR_GOAL_MAX_AREA:
            penalty += 80.0
        if det.confidence < self.NEAR_GOAL_MIN_CONF:
            penalty += 60.0
        if self._goal_bbox is not None:
            gx1, _, gx2, _ = self._goal_bbox
            margin = (gx2 - gx1) * 0.25
            if det.center[0] < gx1 - margin or det.center[0] > gx2 + margin:
                penalty += 100.0
        return penalty

    # ── Area consistency ──────────────────────────────────────────────────────

    def _area_consistent(self, det: BallDetection) -> bool:
        if self.last_detection is None:
            return True
        last_area = max(1.0, float(self.last_detection.area))
        ratio     = float(det.area) / last_area
        # Grow more slowly with missed frames; hard cap at 2.2
        max_growth = min(self.AREA_RATIO_MAX + 0.10 * self.missed_frames, 2.2)
        min_shrink = max(self.AREA_RATIO_MIN - 0.02 * self.missed_frames, 0.4)
        return min_shrink <= ratio <= max_growth

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _adaptive_max_dist(self) -> float:
        speed = 0.0
        if self.velocity is not None:
            speed = float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))
        dist = self.BASE_MAX_DIST + 0.9 * speed + self.missed_frames * self.MAX_DIST_PER_MISS
        return min(500.0, dist)

    def _distance(self, p1: tuple, p2: tuple) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(np.sqrt(dx * dx + dy * dy))