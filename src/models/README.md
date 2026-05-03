# YOLO Models Configuration

## Overview
This directory centralizes all YOLO model configuration for the penalty prediction pipeline. Models are automatically downloaded to the `models/` directory in the project root and cached locally.

## File Structure
```
Penalties-Prediction-By-Body-Language/
├── src/models/
│   ├── config.py       ← Configuration (sets YOLO_HOME)
│   ├── __init__.py
│   └── README.md       ← This file
│
├── models/             ← ⭐ Where models are stored
│   ├── README.md
│   ├── .gitignore
│   └── [yolov8m.pt, etc. downloaded here]
```

## Models Used

### Ball Detection: `yolov8m.pt`
**Previous:** yolov8s.pt (small)
**Chosen:** yolov8m.pt (medium)

**Rationale:**
- Ball detection is critical for accuracy (small object in video)
- yolov8m provides ~15% better mAP than yolov8s
- Speed penalty is acceptable (~2x slower): pipeline processes every 2 frames anyway
- YOLO trained on COCO includes "sports ball" class (index 32)
- Medium model has better localization precision for small objects

**Comparison:**
| Model | Accuracy (mAP) | Speed | Size | Recommendation |
|-------|----------------|-------|------|---|
| yolov8n | 37.3 | ~1ms | 6.2M | Too fast, poor accuracy |
| yolov8s | 44.9 | ~11ms | 22.5M | Previous choice, acceptable |
| **yolov8m** | **50.2** | **26ms** | **49.0M** | ✓ **Best balance** |
| yolov8l | 52.9 | ~103ms | 129.3M | Overkill, too slow |

### Players Detection: `yolov8m.pt`
**Previous:** yolov8s.pt (small)
**Chosen:** yolov8m.pt (medium)

**Rationale:**
- Need to distinguish shooter vs goalkeeper accurately
- Requires good pose quality for role identification
- yolov8m better at detecting multiple persons and distinguishing them
- Person detection (COCO class 0) is more reliable with larger model
- Post-processing filters by geometry (shooter/goalkeeper roles)

### Pose Estimation: `yolov8m-pose.pt`
**Previous:** yolov8s-pose.pt (small)
**Chosen:** yolov8m-pose.pt (medium)

**Rationale:**
- Pose keypoints critical for penalty analysis (body angles, momentum)
- Accuracy in keypoint localization directly affects metrics
- Medium pose model provides more stable keypoint detection
- Used for calculating: shooting angle, goalkeeper position, ball trajectory predictions
- Performance impact acceptable given importance of pose quality

## Configuration

All model paths and confidence thresholds are centralized in `config.py`:

```python
from src.models import ModelConfig

# Access model paths
ball_model = ModelConfig.get_ball_model_path()        # "yolov8m.pt"
players_model = ModelConfig.get_players_model_path()  # "yolov8m.pt"
pose_model = ModelConfig.get_pose_model_path()        # "yolov8m-pose.pt"

# Access confidence thresholds
ball_conf = ModelConfig.BALL_CONFIDENCE               # 0.4
players_conf = ModelConfig.PLAYERS_CONFIDENCE         # 0.25
pose_conf = ModelConfig.POSE_CONFIDENCE               # 0.25
```

## Integration

### Direct Pipeline Usage (Recommended)
```python
from src.pipeline import PenaltyPipeline
Storage

- **Location:** `models/` directory in project root
- **Auto-download:** Models are downloaded on first use
- **Caching:** Downloaded once and reused for all subsequent runs
- **Environment:** `YOLO_HOME` is set to `models/` directory automatically
- **Approximate total:** ~107 MB for all three models
  - yolov8m.pt: ~49 MB
  - yolov8m-pose.pt: ~58 MB
  - Shared model: ~49 MB (used for both ball and players detection)
pipeline = PenaltyPipeline(
    ball_model="yolov8s.pt",  # Use smaller/faster model
    players_model="path/to/custom/model.pt",
)
```

## Model Download & Caching

- Models are auto-downloaded by Ultralytics on first use
- Location: `~/.config/Ultralytics/` or project root if using relative paths
- Downloaded once and reused for all subsequent runs
- Approximate sizes:
  - yolov8m.pt: ~49 MB
  - yolov8m-pose.pt: ~58 MB

## Confidence Thresholds

Current configuration uses **permissive confidence thresholds** (0.25-0.4) because:
1. Additional filtering is applied by geometry (ROI, aspect ratio, motion)
2. Better to have false positives filtered by business logic than miss true positives
3. Pipeline validates detections through spatial constraints

- **Ball (0.4):** Higher threshold - ball is unique object, fewer false positives
- **Players (0.25):** Lower threshold - multiple persons, filtered by roles
- **Pose (0.25):** Lower threshold - only estimated for detected persons

## Performance Impact

Processing times per frame (estimated):
- Goal detection (HSV): ~1-2ms
- Ball detection (yolov8m): ~26ms
- Players detection (yolov8m): ~26ms
- Pose estimation (yolov8m-pose): ~35ms
- **Total: ~90ms per frame** (with process_every_n_frames=2)

For 30 FPS video, this is acceptable as a separate analysis thread.

## Future Improvements

1. **Custom Training:** Fine-tune models on penalty-specific dataset
2. **Model Distillation:** Compress medium models to small with maintained accuracy
3. **ONNX Conversion:** Export to ONNX for faster inference
4. **Ensemble Methods:** Combine predictions from multiple models for robustness
5. **Lightweight Alternatives:** Evaluate YOLOv5, YOLOv4-tiny for edge deployment

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
- [YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/)
