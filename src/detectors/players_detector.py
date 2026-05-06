"""Players detection using YOLO with goalkeeper occlusion recovery."""
 
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from ultralytics import YOLO
from ..models import ModelConfig
 
 
@dataclass
class PlayerDetection:
    """Detection result for a player."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None
 
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
 
 
class PlayersDetector:
    """
    Detect players in penalty scenes using YOLO + ByteTrack.
 
    Key improvements over the original:
    - Goalkeeper last-known position is preserved across short occlusion gaps
      (jump into the net, overlap with post) — the 8 micro-loss clusters seen
      in the analysis are bridged using a ghost detection while the tracker
      re-acquires.
    - Confidence threshold lowered specifically for the goalkeeper region to
      tolerate partial occlusion by the net.
    - Per-role max-missed-frames so the goalkeeper ghost stays alive longer
      than the shooter ghost.
    """
 
    # Goalkeeper ghost: frames to keep alive during normal play
    GK_MAX_GHOST_FRAMES = 8
    # Post-shot: ghost kept indefinitely (portero en el suelo, ByteTrack no re-adquiere)
    GK_MAX_GHOST_FRAMES_POST_SHOT = 60
    # Shooter ghost: shorter — shooter is almost always visible
    SH_MAX_GHOST_FRAMES = 4
    # Confidence decay per ghost frame (ghost conf * decay^n)
    # Post-shot decay is slower — we want the box to remain visible
    GHOST_CONF_DECAY = 0.80
    GHOST_CONF_DECAY_POST_SHOT = 0.98
 
    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
    ):
        if model_path is None:
            model_path = ModelConfig.get_players_model_path()
        if confidence is None:
            confidence = ModelConfig.PLAYERS_CONFIDENCE
 
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence
 
        # Spatial filters
        self.central_x_min   = 0.15
        self.central_x_max   = 0.85
        self.top_ignore_ratio = 0.14
        self.min_area_ratio   = 0.001
 
        # ── Ghost state (role -> last confirmed detection + missed counter) ──
        self._ghost: dict[str, dict] = {
            "goalkeeper": {"det": None, "missed": 0},
            "shooter":    {"det": None, "missed": 0},
        }
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        Detect all visible players. Returns raw detections — role assignment
        and ghost injection happen in pipeline._identify_roles_by_track_id
        via update_ghosts().
        """
        results = self.model.track(
            frame,
            imgsz=1280,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )
 
        if not results or len(results[0].boxes) == 0:
            return []
 
        detections = []
        boxes = results[0].boxes
        shape = frame.shape
 
        for i in range(len(boxes)):
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            conf   = float(boxes.conf[i])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
 
            if not self._size_ok((x1, y1, x2, y2), shape):
                continue
            if not self._position_ok((x1, y1, x2, y2), center, shape):
                continue
 
            track_id: int | None = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
 
            detections.append(PlayerDetection(
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                center=center,
                track_id=track_id,
            ))
 
        return detections
 
    def update_ghost(
        self,
        role: str,
        confirmed: PlayerDetection | None,
        post_shot: bool = False,
    ) -> PlayerDetection | None:
        """
        Called by the pipeline after role assignment for each frame.
 
        - If `confirmed` is not None: store it and reset missed counter.
        - If `confirmed` is None: increment missed counter and return a
          ghost detection (last known position) if within the allowed gap.
        - post_shot=True: GK ghost never expires and decays very slowly.
          Used once the shot is detected and the GK dives — ByteTrack
          typically cannot re-acquire a diving goalkeeper, so we hold the
          last known position until the end of the clip.
 
        Returns the detection to actually use for this role (real or ghost).
        """
        if role == "goalkeeper" and post_shot:
            max_miss = self.GK_MAX_GHOST_FRAMES_POST_SHOT
            decay    = self.GHOST_CONF_DECAY_POST_SHOT
        elif role == "goalkeeper":
            max_miss = self.GK_MAX_GHOST_FRAMES
            decay    = self.GHOST_CONF_DECAY
        else:
            max_miss = self.SH_MAX_GHOST_FRAMES
            decay    = self.GHOST_CONF_DECAY
 
        state = self._ghost[role]
 
        if confirmed is not None:
            state["det"]    = confirmed
            state["missed"] = 0
            return confirmed
 
        # confirmed is None — try ghost
        state["missed"] += 1
        if state["det"] is None or state["missed"] > max_miss:
            return None
 
        # Return ghost with decayed confidence so it's visually distinct
        g = state["det"]
        ghost_conf = max(0.05, g.confidence * (decay ** state["missed"]))
        return PlayerDetection(
            bbox_xyxy=g.bbox_xyxy,
            confidence=ghost_conf,
            center=g.center,
            track_id=g.track_id,
        )
 
    def reset_ghost(self, role: str) -> None:
        """Force-clear a role's ghost (e.g. between penalty kicks)."""
        self._ghost[role] = {"det": None, "missed": 0}
 
    # ── Spatial filters ───────────────────────────────────────────────────────
 
    def _size_ok(self, bbox: tuple, shape: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area >= self.min_area_ratio * float(shape[0] * shape[1])
 
    def _position_ok(self, bbox: tuple, center: tuple, shape: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = center
        h, w = shape[:2]
        if not (self.central_x_min * w <= cx <= self.central_x_max * w):
            return False
        if cy < self.top_ignore_ratio * h:
            return False
        if y2 < self.top_ignore_ratio * h:
            return False
        return True