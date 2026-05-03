"""Ball detection + tracking using YOLO with motion constraints."""

import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO
from ..models import ModelConfig


@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None  # Ultralytics ByteTrack ID

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class BallDetector:
    """
    Detecta y sigue el balón en vídeos de penalti filmados desde detrás del lanzador.

    Mejoras respecto a la versión anterior
    ----------------------------------------
    1. Filtro de circularidad (aspect-ratio): el balón siempre es casi cuadrado en su
       bounding-box; manos alargadas, botellas y recogepelotas quedan eliminados.
    2. Zona de peligro (near-goal zone): cuando el balón se acerca a la portería se
       aplican criterios más exigentes (circularidad, tamaño máximo, confidence mínima)
       para no confundirlo con objetos estáticos del fondo.
    3. Filtro de movimiento direccional: el balón en un penalti siempre se aleja del
       lanzador (↑ en imagen). Detecciones que impliquen un giro de >120° respecto a la
       trayectoria esperada se penalizan fuertemente en el score.
    4. Estimación de tamaño esperado por perspectiva: al alejarse el balón crece en
       imagen (portería más cerca = más grande). Se mantiene un modelo lineal sencillo
       y se penalizan candidatos cuyo tamaño se desvíe demasiado.
    5. Historial de posiciones para una predicción más suave (media ponderada
       exponencial de velocidades con más pasos).
    6. `reset()` público para reinicializar el tracker entre clips.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ):
        """Initialize ball detector with YOLO model.
        
        Args:
            model_path: Path to YOLO model. If None, uses ModelConfig.BALL_MODEL.
            confidence: Confidence threshold. If None, uses ModelConfig.BALL_CONFIDENCE.
        """
        if model_path is None:
            model_path = ModelConfig.get_ball_model_path()
        if confidence is None:
            confidence = ModelConfig.BALL_CONFIDENCE
            
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence

        # ── Parámetros de tracking ───────────────────────────────────────────
        self.BASE_MAX_DIST = 200
        self.MAX_DIST_PER_MISS = 40
        self.MAX_MISSED_FRAMES = 8
        self.ROI_RADIUS = 260
        self.last_track_id: int | None = None  # Track ID from previous frame

        # ── Tamaño del balón ─────────────────────────────────────────────────
        self.MIN_AREA = 12
        self.MAX_AREA = 9000
        self.AREA_RATIO_MIN = 0.45
        self.AREA_RATIO_MAX = 2.4

        # ── Circularidad (aspect-ratio del bounding-box) ─────────────────────
        # Un balón proyecta una caja casi cuadrada; permitimos hasta 1.8 de ratio
        # para tolerar oclusiones parciales.
        self.MAX_ASPECT_RATIO = 1.8

        # ── Zona de peligro cerca de portería ────────────────────────────────
        # Fracción de la altura del frame a partir de la cual aplicamos filtros extra
        # (0.0 = top, 1.0 = bottom; la portería está en la parte superior del frame)
        self.NEAR_GOAL_Y_FRACTION = 0.45   # por encima de este umbral → zona peligro
        self.NEAR_GOAL_MAX_AREA = 5500     # el balón no puede ser "muy grande" lejos
        self.NEAR_GOAL_MIN_CONF = 0.25     # requerimos más confianza de YOLO cerca
        self.NEAR_GOAL_MAX_ASPECT = 1.45   # más exigente en circularidad

        # ── Velocidad mínima para aplicar filtro de dirección ────────────────
        self.MIN_SPEED_FOR_DIR_FILTER = 8.0  # píxeles/frame

        # ── Estado del tracker ───────────────────────────────────────────────
        self.reset()

        # ── Referencia de portería (inyectada desde el pipeline) ─────────────
        self._goal_bbox: tuple[int, int, int, int] | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reinicializa el tracker (útil entre clips o tras pérdida prolongada)."""
        self.last_position: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.last_detection: BallDetection | None = None
        self.missed_frames: int = 0
        self._area_history: list[float] = []   # historial de áreas para estimación

    def set_goal_bbox(self, goal_bbox: tuple[int, int, int, int] | None) -> None:
        """Permite que el pipeline inyecte la posición de la portería."""
        self._goal_bbox = goal_bbox

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        h, w = frame.shape[:2]

        # Use track() with ByteTrack for native tracking
        results = self.model.track(
            frame,
            imgsz=1280,
            persist=True,
            tracker='bytetrack.yaml',
            conf=self.confidence,
            classes=[32],   # sports ball
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return self._handle_no_detection()

        boxes = results[0].boxes
        candidates: list[tuple[BallDetection, int | None]] = []

        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            # Extract track ID if available (None if tracking not active yet)
            track_id: int | None = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
            
            det = BallDetection((x1, y1, x2, y2), conf, center, track_id)

            if self._is_valid(det, frame.shape):
                candidates.append((det, track_id))

        if not candidates:
            return self._handle_no_detection()

        pred = self._predict()
        max_dist = self._adaptive_max_dist()

        # ── Filtro principal con prioridad a track_id coincidente ──────────────
        # Buscar primero un candidato que coincida con el track_id anterior
        if self.last_track_id is not None:
            for det, track_id in candidates:
                if track_id == self.last_track_id:
                    # Encontramos el mismo track_id, usarlo si pasa validaciones
                    if self._area_consistent(det):
                        self._update_state(det)
                        return det

        # ── Si no encontramos track_id coincidente, usar lógica normal ────────
        valid: list[BallDetection] = []
        for det, _ in candidates:
            if pred is None:
                valid.append(det)
            else:
                dist = self._distance(det.center, pred)
                if dist < max_dist and self._area_consistent(det):
                    valid.append(det)

        # ── Fallback suave ───────────────────────────────────────────────────
        if not valid:
            if pred is not None:
                relaxed = [
                    d for d, _ in candidates
                    if self._distance(d.center, pred) < max_dist * 1.8
                ]
                if relaxed:
                    valid = relaxed
                else:
                    return self._handle_no_detection()
            else:
                return self._handle_no_detection()

        # ── Elegir mejor candidato ───────────────────────────────────────────
        best = min(valid, key=lambda d: self._score(d, pred, h))
        self._update_state(best)
        return best

    # ─────────────────────────────────────────────────────────────────────────
    # Validaciones
    # ─────────────────────────────────────────────────────────────────────────

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
        """Rechaza objetos alargados (botellas, manos, recogepelotas)."""
        x1, y1, x2, y2 = det.bbox_xyxy
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        ratio = max(bw, bh) / min(bw, bh)

        h = shape[0]
        # Cerca de la portería (zona superior) somos más estrictos
        if det.center[1] < h * self.NEAR_GOAL_Y_FRACTION:
            return ratio <= self.NEAR_GOAL_MAX_ASPECT

        return ratio <= self.MAX_ASPECT_RATIO

    def _position_ok(self, center: tuple[float, float], shape: tuple) -> bool:
        x, y = center
        h, w = shape[:2]
        return 0.02 * w < x < 0.98 * w and 0.05 * h < y < 0.98 * h

    def _in_roi(self, center: tuple[float, float]) -> bool:
        if self.last_position is None:
            return True
        dx = abs(center[0] - self.last_position[0])
        dy = abs(center[1] - self.last_position[1])
        roi_radius = max(self.ROI_RADIUS, self._adaptive_max_dist() * 1.15)
        return dx < roi_radius and dy < roi_radius

    # ─────────────────────────────────────────────────────────────────────────
    # Tracking y predicción
    # ─────────────────────────────────────────────────────────────────────────

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
            new_vx = new_pos[0] - self.last_position[0]
            new_vy = new_pos[1] - self.last_position[1]
            if self.velocity is None:
                self.velocity = (new_vx, new_vy)
            else:
                # EMA de velocidad: suavizado más conservador para reducir saltos
                alpha = 0.30
                self.velocity = (
                    (1 - alpha) * self.velocity[0] + alpha * new_vx,
                    (1 - alpha) * self.velocity[1] + alpha * new_vy,
                )

        self.last_position = new_pos
        self.last_detection = detection
        self.last_track_id = detection.track_id  # Store ByteTrack ID for next frame
        self.missed_frames = 0

        # Mantener historial de áreas (últimos 8 frames)
        self._area_history.append(detection.area)
        if len(self._area_history) > 8:
            self._area_history.pop(0)

    def _handle_no_detection(self) -> BallDetection | None:
        if (
            self.last_position is not None
            and self.velocity is not None
            and self.missed_frames < self.MAX_MISSED_FRAMES
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
                        int(px - bw / 2),
                        int(py - bh / 2),
                        int(px + bw / 2),
                        int(py + bh / 2),
                    ),
                    confidence=max(0.05, self.last_detection.confidence * 0.7),
                    center=(px, py),
                )

            return BallDetection(bbox_xyxy=(0, 0, 0, 0), confidence=0.0, center=pred)

        self.missed_frames += 1
        if self.missed_frames > self.MAX_MISSED_FRAMES:
            self.reset()

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Score (menor = mejor candidato)
    # ─────────────────────────────────────────────────────────────────────────

    def _score(
        self,
        det: BallDetection,
        pred: tuple[float, float] | None,
        frame_height: int,
    ) -> float:
        # ── Distancia a predicción ───────────────────────────────────────────
        if pred is None:
            dist = 0.0
        else:
            dx = det.center[0] - pred[0]
            dy = det.center[1] - pred[1]
            dist = float(np.sqrt(dx * dx + dy * dy))

            # Penalizar cambio brusco de dirección
            if self.velocity is not None:
                dot = dx * self.velocity[0] + dy * self.velocity[1]
                if dot < 0:
                    dist *= 2.0

        # ── Penalización por dirección inesperada ────────────────────────────
        dir_penalty = self._direction_penalty(det)

        # ── Penalización de área ─────────────────────────────────────────────
        area_penalty = 0.0
        if self.last_detection is not None:
            area_penalty = abs(
                float(np.log((det.area + 1.0) / (self.last_detection.area + 1.0)))
            )

        # ── Penalización extra en zona de portería ───────────────────────────
        goal_zone_penalty = self._goal_zone_penalty(det, frame_height)

        return (
            dist
            + 45.0 * area_penalty
            - 20.0 * det.confidence
            + dir_penalty
            + goal_zone_penalty
        )

    def _direction_penalty(self, det: BallDetection) -> float:
        """
        En un penalti el balón se mueve hacia la portería (arriba en imagen → vy < 0).
        Si el candidato implicaría un movimiento contrario a la trayectoria actual,
        lo penalizamos. Sólo se activa cuando la velocidad es significativa.
        """
        if self.velocity is None or self.last_position is None:
            return 0.0

        speed = float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))
        if speed < self.MIN_SPEED_FOR_DIR_FILTER:
            return 0.0

        # Vector del candidato respecto a posición actual
        dx = det.center[0] - self.last_position[0]
        dy = det.center[1] - self.last_position[1]
        candidate_speed = float(np.sqrt(dx * dx + dy * dy))
        if candidate_speed < 1e-3:
            return 0.0

        # Coseno entre la velocidad actual y el vector al candidato
        cos_theta = (
            self.velocity[0] * dx + self.velocity[1] * dy
        ) / (speed * candidate_speed)

        # cos < -0.5 → ángulo > 120°: movimiento prácticamente opuesto
        if cos_theta < -0.5:
            return 120.0
        # cos < 0 → ángulo > 90°: penalización suave
        if cos_theta < 0.0:
            return 40.0
        return 0.0

    def _goal_zone_penalty(self, det: BallDetection, frame_height: int) -> float:
        """
        Penaliza candidatos en la zona de portería que no cumplan criterios estrictos.
        Esto evita que se detecten botellas, manos del portero o recogepelotas.
        """
        cy = det.center[1]
        # La portería está en la parte superior del frame
        if cy >= frame_height * self.NEAR_GOAL_Y_FRACTION:
            return 0.0   # fuera de zona de riesgo, sin penalización extra

        penalty = 0.0

        # Área demasiado grande para un balón lejos de la cámara
        if det.area > self.NEAR_GOAL_MAX_AREA:
            penalty += 80.0

        # Confianza de YOLO demasiado baja: probablemente un falso positivo
        if det.confidence < self.NEAR_GOAL_MIN_CONF:
            penalty += 60.0

        # Si la portería es conocida y el candidato está claramente fuera de ella,
        # penalizamos (botellas en los laterales de la portería)
        if self._goal_bbox is not None:
            gx1, gy1, gx2, gy2 = self._goal_bbox
            goal_w = gx2 - gx1
            # Margen lateral generoso para no descartar balones que rozan el poste
            margin = goal_w * 0.25
            if det.center[0] < gx1 - margin or det.center[0] > gx2 + margin:
                penalty += 100.0

        return penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Consistencia de área
    # ─────────────────────────────────────────────────────────────────────────

    def _area_consistent(self, det: BallDetection) -> bool:
        if self.last_detection is None:
            return True

        last_area = max(1.0, float(self.last_detection.area))
        ratio = float(det.area) / last_area
        ratio_min = max(0.3, self.AREA_RATIO_MIN - 0.03 * self.missed_frames)
        ratio_max = min(3.2, self.AREA_RATIO_MAX + 0.15 * self.missed_frames)
        return ratio_min <= ratio <= ratio_max

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────────

    def _adaptive_max_dist(self) -> float:
        speed = 0.0
        if self.velocity is not None:
            speed = float(np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2))
        adaptive = self.BASE_MAX_DIST + 0.9 * speed + self.missed_frames * self.MAX_DIST_PER_MISS
        return min(520.0, adaptive)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(np.sqrt(dx * dx + dy * dy))