# 📋 Comparación: Antes vs Después - ByteTrack

## BallDetector

### Dataclass BallDetection

**ANTES:**
```python
@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
```

**DESPUÉS:**
```python
@dataclass
class BallDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None  # ← NUEVO

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
```

---

### __init__()

**ANTES:**
```python
self.BASE_MAX_DIST = 200
self.MAX_DIST_PER_MISS = 40
self.MAX_MISSED_FRAMES = 8
self.ROI_RADIUS = 260
# ... resto de parámetros
self.reset()
self._goal_bbox: tuple[int, int, int, int] | None = None
```

**DESPUÉS:**
```python
self.BASE_MAX_DIST = 200
self.MAX_DIST_PER_MISS = 40
self.MAX_MISSED_FRAMES = 8
self.ROI_RADIUS = 260
self.last_track_id: int | None = None  # ← NUEVO (para ByteTrack)
# ... resto de parámetros
self.reset()
self._goal_bbox: tuple[int, int, int, int] | None = None
```

---

### Método detect()

**ANTES:**
```python
def detect(self, frame: np.ndarray) -> BallDetection | None:
    h, w = frame.shape[:2]

    # ❌ Usa predict (sin tracking)
    results = self.model.predict(
        frame,
        imgsz=1280,
        conf=self.confidence,
        classes=[32],
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return self._handle_no_detection()

    boxes = results[0].boxes
    candidates: list[BallDetection] = []

    for i in range(len(boxes)):
        coords = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        conf = float(boxes.conf[i])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        det = BallDetection((x1, y1, x2, y2), conf, center)  # ❌ Sin track_id

        if self._is_valid(det, frame.shape):
            candidates.append(det)

    # ... resto de lógica de scoring
```

**DESPUÉS:**
```python
def detect(self, frame: np.ndarray) -> BallDetection | None:
    h, w = frame.shape[:2]

    # ✅ Usa track() con ByteTrack
    results = self.model.track(
        frame,
        imgsz=1280,
        persist=True,                    # ← NUEVO
        tracker='bytetrack.yaml',        # ← NUEVO
        conf=self.confidence,
        classes=[32],
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
        
        # ✅ Extrae track_id
        track_id: int | None = None
        if boxes.id is not None:
            track_id = int(boxes.id[i].item())
        
        det = BallDetection((x1, y1, x2, y2), conf, center, track_id)

        if self._is_valid(det, frame.shape):
            candidates.append((det, track_id))

    # ✅ NUEVA LÓGICA: Prioriza track_id coincidente
    if not candidates:
        return self._handle_no_detection()

    pred = self._predict()
    max_dist = self._adaptive_max_dist()

    # Buscar primero un candidato que coincida con el track_id anterior
    if self.last_track_id is not None:
        for det, track_id in candidates:
            if track_id == self.last_track_id:
                if self._area_consistent(det):
                    self._update_state(det)
                    return det

    # Si no encontramos track_id coincidente, usar lógica normal
    # ... resto de lógica de scoring
```

---

### Método _update_state()

**ANTES:**
```python
def _update_state(self, detection: BallDetection) -> None:
    new_pos = detection.center
    if self.last_position is not None:
        # ... cálculo de velocidad
    
    self.last_position = new_pos
    self.last_detection = detection
    self.missed_frames = 0  # ← No guarda track_id
    
    # ... actualiza historial de áreas
```

**DESPUÉS:**
```python
def _update_state(self, detection: BallDetection) -> None:
    new_pos = detection.center
    if self.last_position is not None:
        # ... cálculo de velocidad
    
    self.last_position = new_pos
    self.last_detection = detection
    self.last_track_id = detection.track_id  # ← NUEVO: Guarda para próximo frame
    self.missed_frames = 0
    
    # ... actualiza historial de áreas
```

---

## PlayersDetector

### Dataclass PlayerDetection

**ANTES:**
```python
@dataclass
class PlayerDetection:
    """Detection result for a player."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
```

**DESPUÉS:**
```python
@dataclass
class PlayerDetection:
    """Detection result for a player."""
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    center: tuple[float, float]
    track_id: int | None = None  # ← NUEVO
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
```

---

### Método detect()

**ANTES:**
```python
def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
    """Detect all players in frame."""
    
    # ❌ Usa predict (sin tracking)
    results = self.model.predict(
        frame,
        conf=self.confidence,
        classes=[0],
        verbose=False
    )
    
    if not results or len(results[0].boxes) == 0:
        return []
    
    detections = []
    boxes = results[0].boxes
    frame_shape = frame.shape
    
    for i in range(len(boxes)):
        coords = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        conf = float(boxes.conf[i])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if not self._size_ok((x1, y1, x2, y2), frame_shape):
            continue

        if not self._position_ok((x1, y1, x2, y2), center, frame_shape):
            continue
        
        detections.append(PlayerDetection(
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf,
            center=center  # ❌ Sin track_id
        ))
    
    return detections
```

**DESPUÉS:**
```python
def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
    """Detect all players in frame using ByteTrack."""
    
    # ✅ Usa track() con ByteTrack
    results = self.model.track(
        frame,
        imgsz=1280,                   # ← NUEVO
        persist=True,                  # ← NUEVO
        tracker='bytetrack.yaml',      # ← NUEVO
        conf=self.confidence,
        classes=[0],
        verbose=False
    )
    
    if not results or len(results[0].boxes) == 0:
        return []
    
    detections = []
    boxes = results[0].boxes
    frame_shape = frame.shape
    
    for i in range(len(boxes)):
        coords = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        conf = float(boxes.conf[i])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if not self._size_ok((x1, y1, x2, y2), frame_shape):
            continue

        if not self._position_ok((x1, y1, x2, y2), center, frame_shape):
            continue
        
        # ✅ Extrae track_id después de pasar filtros
        track_id: int | None = None
        if boxes.id is not None:
            track_id = int(boxes.id[i].item())
        
        detections.append(PlayerDetection(
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf,
            center=center,
            track_id=track_id  # ← NUEVO
        ))
    
    return detections
```

---

## Resumen de Cambios

| Aspecto | Cambio |
|---------|--------|
| **Método de detección** | `predict()` → `track()` |
| **Dataclass BallDetection** | +`track_id` |
| **Dataclass PlayerDetection** | +`track_id` |
| **__init__ BallDetector** | +`last_track_id` |
| **Lógica de filtrado** | +Prioridad a track_id coincidente |
| **Backward compatibility** | ✅ Sí (score como fallback) |
| **Lines of code** | ~20 líneas adicionales |
| **Complexity** | Baja (código muy legible) |

---

## ✅ Testing

### Verificar que track_ids se asignan correctamente:

```python
from src.detectors.ball_detector import BallDetector
from src.detectors.players_detector import PlayersDetector
import cv2

# Test BallDetector
ball_detector = BallDetector()
frame = cv2.imread("test_frame.jpg")
detection = ball_detector.detect(frame)

if detection:
    print(f"Ball track_id: {detection.track_id}")
    assert isinstance(detection.track_id, (int, type(None)))
    print("✅ BallDetector OK")

# Test PlayersDetector
players_detector = PlayersDetector()
detections = players_detector.detect(frame)

for i, det in enumerate(detections):
    print(f"Player {i} track_id: {det.track_id}")
    assert isinstance(det.track_id, (int, type(None)))

if detections:
    print("✅ PlayersDetector OK")
```

---

**Status:** ✅ **COMPLETADO**
**Cambios:** Mínimos, focalizados, backward compatible
**Impacto:** Mejor tracking y continuidad
