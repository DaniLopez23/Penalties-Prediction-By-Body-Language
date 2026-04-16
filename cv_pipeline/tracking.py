from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from .detection import Detection, FrameDetections


@dataclass
class TrackedObject:
    role: str
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    interpolated: bool = False


@dataclass
class TrackedFrame:
    shooter: TrackedObject | None
    goalkeeper: TrackedObject | None
    ball: TrackedObject | None
    goal: Detection | None
    goal_zones: dict[str, tuple[int, int, int, int]] | None
    ball_trajectory: list[tuple[int, int]]
    ball_trajectory_smooth: list[tuple[int, int]]


class MultiObjectTracker:
    """ByteTrack wrapper with role-aware mapping and optional ball interpolation."""

    ROLE_TO_CLASS_ID = {"shooter": 0, "goalkeeper": 1, "ball": 2}
    CLASS_ID_TO_ROLE = {0: "shooter", 1: "goalkeeper", 2: "ball"}

    def __init__(
        self,
        frame_rate: float,
        trajectory_size: int = 50,
        interpolate_ball: bool = True,
    ) -> None:
        self.tracker = sv.ByteTrack(frame_rate=max(1, int(round(frame_rate))))
        self.interpolate_ball = interpolate_ball
        self.track_history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trajectory_size))
        self.last_objects_by_role: dict[str, TrackedObject] = {}
        self.missed_frames_by_role: dict[str, int] = {"shooter": 0, "goalkeeper": 0, "ball": 0}
        self.max_role_gap = {"shooter": 8, "goalkeeper": 14, "ball": 10}
        self._ball_kalman = self._create_ball_kalman()
        self._ball_kalman_initialized = False

    def update(self, frame_detections: FrameDetections) -> TrackedFrame:
        detections_input: list[Detection] = []
        for role_name in ("shooter", "goalkeeper", "ball"):
            det = getattr(frame_detections, role_name)
            if det is not None:
                detections_input.append(Detection(role=role_name, bbox_xyxy=det.bbox_xyxy, confidence=det.confidence))

        shooter: TrackedObject | None = None
        goalkeeper: TrackedObject | None = None
        ball: TrackedObject | None = None

        if detections_input:
            xyxy = np.array([det.bbox_xyxy for det in detections_input], dtype=np.float32)
            confidence = np.array([det.confidence for det in detections_input], dtype=np.float32)
            class_id = np.array([self.ROLE_TO_CLASS_ID[det.role] for det in detections_input], dtype=np.int32)

            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = self.tracker.update_with_detections(sv_detections)

            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=np.int32)
            seen_roles: set[str] = set()
            for idx, track_id in enumerate(tracker_ids):
                cls = int(tracked.class_id[idx]) if tracked.class_id is not None else -1
                role = self.CLASS_ID_TO_ROLE.get(cls)
                if role is None:
                    continue

                x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[idx]]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                obj = TrackedObject(
                    role=role,
                    track_id=int(track_id),
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0,
                    center=(float(cx), float(cy)),
                )

                self.track_history[obj.track_id].append((cx, cy))
                self.last_objects_by_role[role] = obj
                self.missed_frames_by_role[role] = 0
                seen_roles.add(role)

                if role == "ball":
                    self._update_ball_kalman(cx, cy)

                if role == "shooter":
                    shooter = obj
                elif role == "goalkeeper":
                    goalkeeper = obj
                elif role == "ball":
                    ball = obj

            for role in ("shooter", "goalkeeper", "ball"):
                if role not in seen_roles:
                    self.missed_frames_by_role[role] += 1
        else:
            for role in ("shooter", "goalkeeper", "ball"):
                self.missed_frames_by_role[role] += 1

        if shooter is None and self.missed_frames_by_role["shooter"] <= self.max_role_gap["shooter"]:
            shooter = self.last_objects_by_role.get("shooter")
        if goalkeeper is None and self.missed_frames_by_role["goalkeeper"] <= self.max_role_gap["goalkeeper"]:
            goalkeeper = self._predict_goalkeeper()
        if ball is None and self.missed_frames_by_role["ball"] <= self.max_role_gap["ball"]:
            ball = self._interpolate_ball()

        ball_trajectory: list[tuple[int, int]] = []
        if ball is not None and ball.track_id in self.track_history:
            ball_trajectory = list(self.track_history[ball.track_id])
        ball_trajectory_smooth = self._smooth_points(ball_trajectory)

        goal_zones = frame_detections.goal_zones.zones if frame_detections.goal_zones is not None else None

        return TrackedFrame(
            shooter=shooter,
            goalkeeper=goalkeeper,
            ball=ball,
            goal=frame_detections.goal,
            goal_zones=goal_zones,
            ball_trajectory=ball_trajectory,
            ball_trajectory_smooth=ball_trajectory_smooth,
        )

    def _interpolate_ball(self) -> TrackedObject | None:
        if not self.interpolate_ball:
            return self.last_objects_by_role.get("ball")

        last_ball = self.last_objects_by_role.get("ball")
        if last_ball is None:
            return None

        pred_x, pred_y = self._predict_ball_center(last_ball)

        bx1, by1, bx2, by2 = last_ball.bbox_xyxy
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        interpolated = TrackedObject(
            role="ball",
            track_id=last_ball.track_id,
            bbox_xyxy=(pred_x - bw // 2, pred_y - bh // 2, pred_x + bw // 2, pred_y + bh // 2),
            confidence=last_ball.confidence * 0.9,
            center=(float(pred_x), float(pred_y)),
            interpolated=True,
        )
        self.track_history[last_ball.track_id].append((pred_x, pred_y))
        self.last_objects_by_role["ball"] = interpolated
        return interpolated

    def _predict_goalkeeper(self) -> TrackedObject | None:
        last_goalkeeper = self.last_objects_by_role.get("goalkeeper")
        if last_goalkeeper is None:
            return None

        history = self.track_history.get(last_goalkeeper.track_id)
        if history is None or len(history) < 2:
            return last_goalkeeper

        (x1, y1), (x2, y2) = history[-2], history[-1]
        pred_x, pred_y = int(x2 + (x2 - x1)), int(y2 + (y2 - y1))

        gx1, gy1, gx2, gy2 = last_goalkeeper.bbox_xyxy
        gw, gh = max(1, gx2 - gx1), max(1, gy2 - gy1)
        predicted = TrackedObject(
            role="goalkeeper",
            track_id=last_goalkeeper.track_id,
            bbox_xyxy=(pred_x - gw // 2, pred_y - gh // 2, pred_x + gw // 2, pred_y + gh // 2),
            confidence=last_goalkeeper.confidence * 0.92,
            center=(float(pred_x), float(pred_y)),
            interpolated=True,
        )
        self.track_history[last_goalkeeper.track_id].append((pred_x, pred_y))
        self.last_objects_by_role["goalkeeper"] = predicted
        return predicted

    @staticmethod
    def _smooth_points(points: list[tuple[int, int]], window: int = 5) -> list[tuple[int, int]]:
        if len(points) < 3:
            return points

        smoothed: list[tuple[int, int]] = []
        for idx in range(len(points)):
            left = max(0, idx - window // 2)
            right = min(len(points), idx + window // 2 + 1)
            chunk = points[left:right]
            avg_x = int(sum(p[0] for p in chunk) / len(chunk))
            avg_y = int(sum(p[1] for p in chunk) / len(chunk))
            smoothed.append((avg_x, avg_y))
        return smoothed

    @staticmethod
    def _create_ball_kalman() -> cv2.KalmanFilter:
        # State: [x, y, vx, vy], Measurement: [x, y]
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman

    def _update_ball_kalman(self, x: int, y: int) -> None:
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._ball_kalman_initialized:
            self._ball_kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._ball_kalman_initialized = True
        self._ball_kalman.correct(measurement)

    def _predict_ball_center(self, last_ball: TrackedObject) -> tuple[int, int]:
        if self._ball_kalman_initialized:
            pred = self._ball_kalman.predict()
            return int(pred[0][0]), int(pred[1][0])

        history = self.track_history.get(last_ball.track_id)
        if history is None or len(history) < 2:
            cx, cy = last_ball.center
            return int(cx), int(cy)

        (x1, y1), (x2, y2) = history[-2], history[-1]
        return int(x2 + (x2 - x1)), int(y2 + (y2 - y1))
