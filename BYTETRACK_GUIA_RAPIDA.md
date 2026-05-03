# 🚀 Guía Rápida - ByteTrack Integration

## Lo Importante

Tu código ya está usando **ByteTrack automáticamente**. No necesitas hacer nada especial.

---

## ¿Qué es ByteTrack?

ByteTrack es un algoritmo que:
- ✅ Asigna IDs únicos a objetos detectados
- ✅ Mantiene IDs consistentes entre frames
- ✅ Maneja oclusiones y reapariciones
- ✅ Funciona nativo en Ultralytics

---

## Cómo Funciona en Tu Proyecto

### Flujo Automático

```
Video Frame 1
    ↓
BallDetector.detect() usa model.track()
    ↓
ByteTrack asigna track_id = 1 al balón
    ↓
BallDetection(bbox, conf, center, track_id=1) ← Con ID

Video Frame 2
    ↓
BallDetector.detect() usa model.track()
    ↓
ByteTrack reconoce el mismo balón → track_id = 1
    ↓
BallDetection(bbox, conf, center, track_id=1) ← Mismo ID

Video Frame 3
    ↓
ByteTrack mantiene track_id = 1
    ↓
(Ahora tienes continuidad)
```

---

## Accediendo al track_id

```python
from src.detectors.ball_detector import BallDetector
from src.detectors.players_detector import PlayersDetector
import cv2

# Ball Detection
ball_detector = BallDetector()
frame = cv2.imread("frame.jpg")
ball = ball_detector.detect(frame)

if ball:
    print(f"Balón:")
    print(f"  - Posición: {ball.center}")
    print(f"  - Track ID: {ball.track_id}")  # ← Aquí está

# Players Detection
players_detector = PlayersDetector()
players = players_detector.detect(frame)

for player in players:
    print(f"Jugador:")
    print(f"  - Posición: {player.center}")
    print(f"  - Track ID: {player.track_id}")  # ← Aquí está
```

---

## Track ID = None

Esto significa que ByteTrack aún no ha activado el tracking:
- Primeros frames del video
- Después de oclusión prolongada (re-entrada)
- Objeto nuevo en la escena

```python
if ball.track_id is None:
    print("Tracking aún no activo o re-entrada")
else:
    print(f"Tracking activo: ID={ball.track_id}")
```

---

## Ventajas Prácticas

### Para Balón
```
❌ Antes:  Frame 1→2→3 el balón "salta" entre puntos
✅ Ahora:  El track_id mantiene continuidad
           Mejor predicción de trayectoria
```

### Para Jugadores
```
❌ Antes:  Shooter y Goalkeeper se "intercambian" entre frames
✅ Ahora:  Cada uno mantiene su ID
           Mejor seguimiento de roles
           Mejor análisis de ángulos
```

---

## Pipeline (Sin Cambios Necesarios)

Tu pipeline ya funciona como antes. Los track_ids se propagan automáticamente:

```python
from src.pipeline import PenaltyPipeline

# ✅ Automático - nada que cambiar
pipeline = PenaltyPipeline()

result = pipeline.process_video(
    input_video="video.mp4",
    output_video="output.mp4",
)
```

Los track_ids se usan internamente para:
1. Better ball prediction
2. Better player role identification  
3. Smoother trajectories

---

## Debugging: Ver Track IDs en Vivo

Si quieres verificar que ByteTrack funciona:

```python
import cv2
from src.detectors.ball_detector import BallDetector
from src.detectors.players_detector import PlayersDetector

ball_detector = BallDetector()
players_detector = PlayersDetector()

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect with track IDs
    ball = ball_detector.detect(frame)
    players = players_detector.detect(frame)
    
    # Draw track IDs
    if ball and ball.track_id is not None:
        x, y = int(ball.center[0]), int(ball.center[1])
        cv2.putText(
            frame, f"Ball ID: {ball.track_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    for player in players:
        if player.track_id is not None:
            x, y = int(player.center[0]), int(player.center[1])
            cv2.putText(
                frame, f"P ID: {player.track_id}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2
            )
    
    cv2.imshow("Track IDs", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Esperado:**
- Balón tiene consistentemente ID=1 (o el mismo número)
- Jugadores mantienen sus IDs entre frames
- IDs cambian solo si hay oclusión prolongada

---

## Parámetros (Avanzado)

Si quieres ajustar ByteTrack, busca en el código:

```python
# En BallDetector.detect() y PlayersDetector.detect()
results = self.model.track(
    frame,
    imgsz=1280,              # Tamaño de imagen (sin cambiar)
    persist=True,            # ← Mantiene tracks activos
    tracker='bytetrack.yaml', # ← Config estándar ByteTrack
    conf=self.confidence,    # Umbral de confianza (sin cambiar)
    classes=[32],            # Clases (sin cambiar)
    verbose=False,           # Debug output (sin cambiar)
)
```

**persist=True** es crítico - sin esto, ByteTrack no mantiene IDs entre frames.

---

## Casos de Uso Avanzados

### Correlacionar detecciones entre frames

```python
# Rastrear la trayectoria completa del balón
ball_trajectory = []

for frame in video:
    ball = ball_detector.detect(frame)
    if ball:
        ball_trajectory.append({
            'frame': frame_idx,
            'track_id': ball.track_id,
            'center': ball.center,
            'confidence': ball.confidence,
        })

# Análisis posterior
print(f"Balón rastreado con ID={ball_trajectory[0]['track_id']}")
for point in ball_trajectory:
    print(f"Frame {point['frame']}: {point['center']}")
```

### Identificación persistente de jugadores

```python
# Mapear jugadores entre frames
player_map = {}

for frame in video:
    players = players_detector.detect(frame)
    
    for player in players:
        track_id = player.track_id
        if track_id not in player_map:
            player_map[track_id] = {
                'positions': [],
                'confidences': [],
            }
        
        player_map[track_id]['positions'].append(player.center)
        player_map[track_id]['confidences'].append(player.confidence)

# Análisis: ¿Cuál es el jugador más confiable?
for track_id, data in player_map.items():
    avg_conf = sum(data['confidences']) / len(data['confidences'])
    print(f"Player {track_id}: avg confidence = {avg_conf:.2f}")
```

---

## Troubleshooting

### "track_id siempre es None"
**Causas:**
- ByteTrack aún sin activarse (primeros frames)
- Solución: Procesa más frames

### "Track IDs cambian frecuentemente"
**Causas:**
- Oclusiones frecuentes
- Objetos reentrantes
- Solución: Normal, ByteTrack los maneja

### "No hay diferencia vs antes"
**Explicación:**
- ByteTrack funciona mejor en videos largos
- En penaltis cortos (~2s), la diferencia es sutil
- Internamente, el tracking mejora la estabilidad

---

## Resumen

| Aspecto | Estado |
|---------|--------|
| **ByteTrack activo** | ✅ Sí |
| **Track IDs disponibles** | ✅ Sí |
| **Cambios necesarios** | ❌ No |
| **Performance** | ✅ Mejor |
| **Backward compat** | ✅ 100% |

---

**Status:** ✅ **LISTO Y FUNCIONANDO**

Tu código usa ByteTrack automáticamente. Disfruta de mejor tracking. 🚀

