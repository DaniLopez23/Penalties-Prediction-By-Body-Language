"""Ball detection + tracking using YOLO with motion constraints."""

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class BallDetector:
    def __init__(self, model_path="yolov8s.pt", confidence=0.2):
        self.model = YOLO(model_path)
        self.confidence = confidence

        # tracking state
        self.last_position = None
        self.velocity = None
        self.last_detection = None
        self.missed_frames = 0

        # Tuned for penalty shots: allow fast motion near goal while keeping continuity.
        # Increased max dist and missed frames to tolerate occlusions near goal.
        self.BASE_MAX_DIST = 200
        self.MAX_DIST_PER_MISS = 40
        self.MAX_MISSED_FRAMES = 8
        self.ROI_RADIUS = 260
        self.MIN_AREA = 12
        self.MAX_AREA = 8000
        self.AREA_RATIO_MIN = 0.45
        self.AREA_RATIO_MAX = 2.4

    def detect(self, frame):
        results = self.model.predict(
            frame,
            imgsz=1280,
            conf=self.confidence,
            classes=[32],  # sports ball
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return self._handle_no_detection()

        boxes = results[0].boxes
        candidates = []

        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            det = BallDetection((x1, y1, x2, y2), conf, center)

            if self._is_valid(det, frame.shape):
                candidates.append(det)

        if not candidates:
            return self._handle_no_detection()

        # predicción
        pred = self._predict()

        max_dist = self._adaptive_max_dist()

        # filtrar por distancia + consistencia de area
        valid_candidates = []
        for det in candidates:
            if pred is None:
                valid_candidates.append(det)
            else:
                dist = self._distance(det.center, pred)
                if dist < max_dist and self._area_consistent(det):
                    valid_candidates.append(det)

        if not valid_candidates:
            # fallback suave para evitar perder el balon por cambios bruscos puntuales
            if pred is not None:
                relaxed = [d for d in candidates if self._distance(d.center, pred) < (max_dist * 1.8)]
                if relaxed:
                    valid_candidates = relaxed
                else:
                    return self._handle_no_detection()
            else:
                return self._handle_no_detection()

        # elegir mejor candidato (cercano a predicción + coherente con movimiento)
        best = min(valid_candidates, key=lambda d: self._score(d, pred))

        # actualizar tracking
        self._update_state(best)

        return best

    # -------------------------
    # VALIDACIONES
    # -------------------------

    def _is_valid(self, det, shape):
        return self._size_ok(det, shape) and self._position_ok(det.center, shape) and self._in_roi(det.center)

    def _size_ok(self, det, shape):
        area = det.area
        return self.MIN_AREA < area < self.MAX_AREA

    def _position_ok(self, center, shape):
        x, y = center
        h, w = shape[:2]
        # Permissive limits: only reject detections too close to borders.
        return 0.02 * w < x < 0.98 * w and 0.05 * h < y < 0.98 * h

    def _in_roi(self, center):
        if self.last_position is None:
            return True

        dx = abs(center[0] - self.last_position[0])
        dy = abs(center[1] - self.last_position[1])

        roi_radius = max(self.ROI_RADIUS, self._adaptive_max_dist() * 1.15)

        return dx < roi_radius and dy < roi_radius

    # -------------------------
    # TRACKING
    # -------------------------

    def _predict(self):
        if self.last_position is None:
            return None

        if self.velocity is None:
            return self.last_position

        px = self.last_position[0] + self.velocity[0]
        py = self.last_position[1] + self.velocity[1]

        return (px, py)

    def _update_state(self, detection):
        new_pos = detection.center
        if self.last_position is not None:
            new_vx = new_pos[0] - self.last_position[0]
            new_vy = new_pos[1] - self.last_position[1]
            if self.velocity is None:
                self.velocity = (new_vx, new_vy)
            else:
                # Smooth velocity to reduce abrupt direction changes from noisy detections.
                self.velocity = (
                    0.65 * self.velocity[0] + 0.35 * new_vx,
                    0.65 * self.velocity[1] + 0.35 * new_vy,
                )

        self.last_position = new_pos
        self.last_detection = detection
        self.missed_frames = 0

    def _handle_no_detection(self):
        # Si hay una perdida corta, mantiene trayectoria estimada para evitar saltos.
        if self.last_position and self.velocity and self.missed_frames < self.MAX_MISSED_FRAMES:
            self.missed_frames += 1
            pred = self._predict()
            self.last_position = pred

            if self.last_detection is not None:
                x1, y1, x2, y2 = self.last_detection.bbox_xyxy
                bw = max(2, x2 - x1)
                bh = max(2, y2 - y1)
                px, py = pred
                px = float(px)
                py = float(py)

                return BallDetection(
                    bbox_xyxy=(
                        int(px - bw / 2),
                        int(py - bh / 2),
                        int(px + bw / 2),
                        int(py + bh / 2),
                    ),
                    confidence=max(0.05, self.last_detection.confidence * 0.7),
                    center=(px, py),
                )

            return BallDetection(
                bbox_xyxy=(0, 0, 0, 0),
                confidence=0.0,
                center=pred
            )

        self.missed_frames += 1
        if self.missed_frames > self.MAX_MISSED_FRAMES:
            self.last_position = None
            self.velocity = None
            self.last_detection = None
            self.missed_frames = 0

        return None

    def _adaptive_max_dist(self):
        speed = 0.0
        if self.velocity is not None:
            speed = float(np.sqrt(self.velocity[0] * self.velocity[0] + self.velocity[1] * self.velocity[1]))

        adaptive = self.BASE_MAX_DIST + 0.9 * speed + self.missed_frames * self.MAX_DIST_PER_MISS
        return min(520.0, adaptive)

    def _area_consistent(self, det):
        if self.last_detection is None:
            return True

        last_area = max(1.0, float(self.last_detection.area))
        ratio = float(det.area) / last_area
        # Tight gate to reject background objects; slightly relax when short misses happen.
        ratio_min = max(0.3, self.AREA_RATIO_MIN - 0.03 * self.missed_frames)
        ratio_max = min(3.2, self.AREA_RATIO_MAX + 0.15 * self.missed_frames)
        return ratio_min <= ratio <= ratio_max

    # -------------------------
    # MÉTRICAS
    # -------------------------

    def _distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx * dx + dy * dy)

    def _score(self, det, pred):
        if pred is None:
            return -det.confidence  # fallback

        dx = det.center[0] - pred[0]
        dy = det.center[1] - pred[1]
        dist = np.sqrt(dx * dx + dy * dy)

        # penalizar cambio brusco de dirección
        if self.velocity:
            dot = dx * self.velocity[0] + dy * self.velocity[1]
            if dot < 0:
                dist *= 2

        area_penalty = 0.0
        if self.last_detection is not None:
            area_penalty = abs(float(np.log((det.area + 1.0) / (self.last_detection.area + 1.0))))

        return dist + 45.0 * area_penalty - 20.0 * det.confidence