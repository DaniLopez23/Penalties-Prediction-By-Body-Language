from __future__ import annotations

import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import BallState, Detection
from src.utils.geometry import center_distance, clamp


class BallTracker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.kalman: Optional[cv2.KalmanFilter] = None
        self.previous_gray: Optional[np.ndarray] = None
        self.missing_frames = 0
        self.last_state: Optional[BallState] = None
        self.trail: deque[tuple[int, int]] = deque(maxlen=config.ball.trail_length)

    def update(
        self,
        frame: np.ndarray,
        yolo_balls: list[Detection],
        player_boxes: list[Detection],
        valid_mask: Optional[np.ndarray] = None,
    ) -> Optional[BallState]:
        prediction = self._predict()
        candidates = self._yolo_candidates(yolo_balls, prediction, frame.shape, valid_mask)
        candidates.extend(self._motion_candidates(frame, prediction, player_boxes, valid_mask))

        observed_state: Optional[BallState] = None
        if candidates:
            observed_state = max(candidates, key=lambda item: item.confidence)
            self._correct(observed_state.center)
            self.missing_frames = 0
            self.last_state = observed_state
        elif prediction is not None and self.missing_frames < self.config.ball.max_missing_frames:
            self.missing_frames += 1
            radius = self.last_state.radius if self.last_state is not None else self.config.ball.min_ball_radius
            observed_state = BallState(prediction, radius, 0.15, "kalman", observed=False)
            self.last_state = observed_state
        else:
            self.missing_frames += 1
            if self.missing_frames > self.config.ball.max_missing_frames:
                self.kalman = None
                self.trail.clear()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = max(3, self.config.ball.motion_blur_kernel | 1)
        self.previous_gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if observed_state is not None:
            x, y = observed_state.center
            self.trail.append((int(round(x)), int(round(y))))
        return observed_state

    def _predict(self) -> Optional[tuple[float, float]]:
        if self.kalman is None:
            return None
        prediction = self.kalman.predict()
        return float(prediction[0, 0]), float(prediction[1, 0])

    def _correct(self, center: tuple[float, float]) -> None:
        if self.kalman is None:
            self._init_kalman(center)
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
        self.kalman.correct(measurement)

    def _init_kalman(self, center: tuple[float, float]) -> None:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.75
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
        self.kalman = kalman

    def _yolo_candidates(
        self,
        balls: list[Detection],
        prediction: Optional[tuple[float, float]],
        frame_shape: tuple[int, int, int],
        valid_mask: Optional[np.ndarray],
    ) -> list[BallState]:
        height, width = frame_shape[:2]
        max_radius = max(self.config.ball.min_ball_radius, int(min(width, height) * self.config.ball.max_ball_radius_ratio))
        candidates: list[BallState] = []
        for ball in balls:
            if self.config.ball.reject_outside_roi and not self._inside_mask(ball.center, valid_mask):
                continue
            if not self._candidate_jump_is_valid(ball.center, frame_shape):
                continue
            radius = int(clamp(max(ball.width, ball.height) * 0.5, self.config.ball.min_ball_radius, max_radius))
            score = ball.confidence * self.config.ball.yolo_candidate_weight
            if prediction is not None:
                distance = center_distance(ball.center, prediction)
                max_distance = max(1.0, math.hypot(width, height) * self.config.ball.max_prediction_distance_ratio)
                score += self.config.ball.prediction_distance_weight * (1.0 - clamp(distance / max_distance, 0.0, 1.0))
            if self.last_state is not None:
                distance = center_distance(ball.center, self.last_state.center)
                max_distance = max(1.0, math.hypot(width, height) * self.config.ball.max_candidate_jump_ratio)
                score += self.config.ball.last_position_weight * (1.0 - clamp(distance / max_distance, 0.0, 1.0))
            candidates.append(BallState(ball.center, radius, float(score), "yolo", observed=True, track_id=ball.track_id))
        return candidates

    def _motion_candidates(
        self,
        frame: np.ndarray,
        prediction: Optional[tuple[float, float]],
        player_boxes: list[Detection],
        valid_mask: Optional[np.ndarray],
    ) -> list[BallState]:
        if self.previous_gray is None:
            return []

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = max(3, self.config.ball.motion_blur_kernel | 1)
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        diff = cv2.absdiff(self.previous_gray, gray)
        _, threshold = cv2.threshold(diff, self.config.ball.motion_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
        if valid_mask is not None:
            threshold = cv2.bitwise_and(threshold, valid_mask)

        min_area = width * height * self.config.ball.min_motion_area_ratio
        max_area = width * height * self.config.ball.max_motion_area_ratio
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[BallState] = []
        diag = math.hypot(width, height)
        max_prediction_distance = max(1.0, diag * self.config.ball.max_prediction_distance_ratio)
        max_radius = max(self.config.ball.min_ball_radius, int(min(width, height) * self.config.ball.max_ball_radius_ratio))
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue
            aspect_ratio = max(w, h) / max(1.0, min(w, h))
            if aspect_ratio > self.config.ball.motion_candidate_max_aspect_ratio:
                continue
            cx, cy = x + w * 0.5, y + h * 0.5
            if self.config.ball.reject_outside_roi and not self._inside_mask((cx, cy), valid_mask):
                continue
            if not self._candidate_jump_is_valid((cx, cy), frame.shape):
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else clamp((4.0 * math.pi * area) / (perimeter * perimeter), 0.0, 1.0)
            size_score = 1.0 - clamp(area / max_area, 0.0, 1.0)
            score = 0.35 + 0.25 * size_score + self.config.ball.contour_circularity_weight * circularity
            if prediction is not None:
                distance = center_distance((cx, cy), prediction)
                score += self.config.ball.prediction_distance_weight * (1.0 - clamp(distance / max_prediction_distance, 0.0, 1.0))
            if self.last_state is not None:
                distance = center_distance((cx, cy), self.last_state.center)
                score += self.config.ball.last_position_weight * (1.0 - clamp(distance / max_prediction_distance, 0.0, 1.0))
            for player in player_boxes:
                px1, py1, px2, py2 = player.xyxy
                margin = max(8.0, 0.04 * max(player.width, player.height))
                if px1 - margin <= cx <= px2 + margin and py1 - margin <= cy <= py2 + margin:
                    score -= self.config.ball.player_overlap_penalty
                    break
            if self._inside_player_foot_zone((cx, cy), player_boxes):
                score -= self.config.ball.player_foot_overlap_penalty
            if score <= 0.05:
                continue
            radius = int(clamp(max(w, h) * 0.5, self.config.ball.min_ball_radius, max_radius))
            candidates.append(BallState((cx, cy), radius, float(score), "motion", observed=True))
        return candidates

    def _candidate_jump_is_valid(self, center: tuple[float, float], frame_shape: tuple[int, int, int]) -> bool:
        if self.last_state is None:
            return True
        height, width = frame_shape[:2]
        recovery_scale = 1.0 + min(2.0, self.missing_frames * 0.25)
        max_distance = max(1.0, math.hypot(width, height) * self.config.ball.max_candidate_jump_ratio * recovery_scale)
        return center_distance(center, self.last_state.center) <= max_distance

    @staticmethod
    def _inside_mask(center: tuple[float, float], mask: Optional[np.ndarray]) -> bool:
        if mask is None:
            return True
        height, width = mask.shape[:2]
        x = int(round(center[0]))
        y = int(round(center[1]))
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return mask[y, x] > 0

    def _inside_player_foot_zone(self, center: tuple[float, float], player_boxes: list[Detection]) -> bool:
        cx, cy = center
        for player in player_boxes:
            foot_zone_top = player.y1 + player.height * 0.72
            horizontal_margin = max(10.0, player.width * 0.18)
            vertical_margin = max(8.0, player.height * 0.06)
            if (
                player.x1 - horizontal_margin <= cx <= player.x2 + horizontal_margin
                and foot_zone_top <= cy <= player.y2 + vertical_margin
            ):
                return True
        return False
