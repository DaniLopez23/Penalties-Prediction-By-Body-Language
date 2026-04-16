from dataclasses import dataclass

from ultralytics import YOLO


SPORTS_BALL_CLASS_ID = 32


@dataclass
class BallDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]


class BallDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame) -> BallDetection | None:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes if results else None

        detections: list[BallDetection] = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                if class_id != SPORTS_BALL_CLASS_ID:
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf.item())
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                detections.append(
                    BallDetection(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=confidence,
                        center=(center_x, center_y),
                    )
                )

        if not detections:
            return None

        return max(detections, key=lambda item: item.confidence)
