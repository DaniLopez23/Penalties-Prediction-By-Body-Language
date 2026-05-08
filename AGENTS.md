# Penalties Prediction By Body Language - Project Architecture & Pipeline

## Project Overview

**Goal:** Analyze penalty kick videos to extract and track multiple entities (goalkeeper, shooter, ball) with precise role assignment and state tracking.

**Technologies:**
- YOLOv11 (Object Detection & Pose Estimation)
- ByteTrack (Multi-object Tracking)
- OpenCV (Video I/O & Visualization)
- MediaPipe + Supervision (Pose analysis & utilities)
- yt-dlp (Video downloading)

---

## Architecture Overview

```
Input Video
    ↓
VideoReader (CV Handler)
    ↓
FRAME PROCESSING LOOP (every N frames):
    ├─ Goal Detection → Stable Goal Estimation
    ├─ Ball Detection → Physics-aware tracking
    ├─ Players Detection → Shooter/Goalkeeper classification
    └─ Role Assignment + Ghost Detection (occlusion handling)
    ↓
Frame Annotation (bboxes, tracks, labels)
    ↓
VideoWriter (CV Handler)
    ↓
Output Video + Stats
```

---

## Core Components

### 1. **PenaltyPipeline** (`src/pipeline.py`)
**Main orchestrator** for the entire analysis workflow.

**Key Responsibilities:**
- Initialize all detectors (ball, players, goal)
- Process video frames sequentially
- Own the cadence for each expensive inference step; do not force every entity through YOLO on every frame
- Manage frame skipping / refresh cadence (`process_every_n_frames`, goal refresh cadence, ball refresh cadence, player refresh cadence)
- Coordinate role assignment with occlusion recovery and reuse the last confirmed state when detections are temporarily missing
- Annotate frames with bounding boxes and labels
- Export annotated video

**Key State Management:**
- `_stable_goal_bbox`: Exponential moving average (EMA) of detected goal position (stabilized after 4 detections)
- `_shot_detected`: Boolean flag that triggers once ball velocity exceeds 18px threshold
- `_frozen_role_map`: Locks shooter/goalkeeper role assignment post-shot (prevents label flip when shooter enters goal area)
- `_ball_trajectory`: Last 4 positions used to compute shot detection velocity

**Shot Detection Logic:**
```python
velocity = distance(pos[-4], pos[-1])  # 4-frame window
if velocity > 18.0 and not _shot_detected:
    _shot_detected = True
    # Enable stricter ball re-acquisition (MAX_MISSED_POST_SHOT = 5)
    # Freeze role assignments
```

**Pipeline Optimization Rules:**
- Prefer detect -> track/propagate -> reacquire, not detect-every-frame.
- Only refresh a detector when the cadence says so, when confidence collapses, or when the tracked state has been lost.
- Keep a narrow reacquisition region around the last known state instead of reopening the full frame.
- Treat role freezing, ghost tracking and Kalman/physics propagation as first-class parts of the pipeline, not as visualization-only helpers.
- The pipeline must remain the place where refresh frequency, loss handling and fallback behavior are coordinated.

---

### 2. **BallDetector** (`src/detectors/ball_detector.py`)
**Specialized ball tracking** with physics-aware re-acquisition.

**Key Features:**
- YOLO detection (yolov8m.pt, confidence ≥ 0.5)
- **Physics-constrained re-acquisition ellipse:** Searches aligned to velocity vector
  - Tight perpendicular to motion
  - Wider along motion direction
- **Trajectory continuity gate:** Rejects candidates implying impossible acceleration
- **N-frame confirmation buffer:** After ≥3 missed frames, new candidate must appear in 2 consecutive frames
- **Split missed-frame strategy:**
  - Pre-shot: MAX_MISSED_PRE_SHOT = 12 (generous)
  - Post-shot: MAX_MISSED_POST_SHOT = 5 (strict)
- **Velocity EMA:** α = 0.20 (smoother trajectory)

**Operating Rule:**
- Use the ball detector as a refresh/reacquisition step. Between refreshes, propagate the last state with tracking logic instead of calling full inference again.

**Parameters (60fps-calibrated for 1080p):**
- BASE_MAX_DIST = 280px
- ROI_RADIUS = 320px
- MIN_SPEED_FOR_DIR_FILTER = 14.0px/frame

**State:**
- `last_valid_detection`: Bounding box + center
- `velocity_ema`: Exponentially smoothed velocity vector
- `missed_count`: Frames without detection
- `shot_detected`: Flag from pipeline to enable strict mode

---

### 3. **PlayersDetector** (`src/detectors/players_detector.py`)
**Detects and tracks players** (shooter + goalkeeper) with occlusion recovery.

**Key Features:**
- YOLO detection (yolov8m.pt, confidence ≥ 0.3)
- **ByteTrack integration:** Multi-object tracking with track ID assignment
- **Ghost detection:** Preserves last-known position during short occlusions
  - Pre-shot GK ghost: 8 frames
  - Post-shot GK ghost: 60 frames (goalkeeper on ground)
  - Shooter ghost: 4 frames
- **Per-role confidence decay:**
  - Normal: α_decay = 0.80
  - Post-shot: α_decay = 0.98 (slower decay to keep visible)

**Operating Rule:**
- The player pass should stay track-based and cadence-controlled. Re-run expensive pose/person inference only when a refresh is needed; otherwise keep the last confirmed roles alive through the ghost tracker and role assigner.

**Occlusion Scenarios Handled:**
- Goalkeeper jumping/diving (net occlusion)
- Goalkeeper overlap with goal post
- Shooter entering goal area

---

### 4. **GoalDetector** (`src/detectors/goal_detector.py`)
**Detects goal location** in frame and provides region masking.

**Key Features:**
- Goal bounding box estimation
- Masking outside goal area for analysis frame
- Crossbar region filtering (rejects detections in crowd area)

**Masking Rule:**
- Any blur or mask that should influence detection or tracking must be applied before the detector sees the frame. Visualization blur alone is not enough.
- Keep a reusable preprocessing mask for detector inputs, separate from the final annotated output frame.

---

## Pipeline Processing Flow

### Frame Loop (in `process_video`)

**For each frame:**

1. **Frame Skip Check:**
   ```
   if (frame_idx % process_every_n_frames) != 0:
       skip_detection
   ```

2. **Goal Detection:**
   - Detect goal region
   - Update stable goal via EMA: `_update_stable_goal()`
   - Pass to ball detector for ROI constraint

3. **Ball Detection:**
   - Run YOLO on masked analysis frame
   - Extract physics-aware trajectory
   - Compute velocity magnitude
   - Check if velocity > 18.0 → set `_shot_detected = True`

4. **Players Detection:**
   - Run YOLO on masked analysis frame
   - Recover missing players via ghost detection
   - Update ByteTrack with new detections

5. **Role Assignment:**
   ```python
   shooter, goalkeeper = _identify_roles_by_track_id(
       players, effective_goal, stable_goal_bbox
   )
   ```
   - **Pre-shot:** Spatial heuristic
     - Player left of goal center → shooter
     - Player right of goal center → goalkeeper
   - **Post-shot:** Frozen role map (no reassignment)
   - **Crossbar filter:** Reject detections in crowd region

6. **Frame Annotation:**
   - Draw bounding boxes (green=shooter, blue=goalkeeper, cyan=ball)
   - Render track IDs
   - Add frame counter + metrics
   - Handle preview window (if enabled)

7. **Output:**
   - Write annotated frame to video
   - Optionally display in preview window

---

## Model Configuration (`src/models/config.py`)

### Active Models
| Purpose | Model | Confidence |
|---------|-------|-----------|
| Ball Detection | yolov8m.pt | 0.5 |
| Players Detection | yolov8m.pt | 0.3 |
| Pose Estimation | yolov8m-pose.pt | 0.6 (optional) |

### Why yolov8m (Medium)?
- **Speed:** ~2x slower than yolov8s but crucial for accuracy
- **Accuracy:** Good balance for penalty analysis
- **Trade-off:** Better than yolov8n (nano) or yolov8s (small) for this task

### Processing Control
- `PROCESS_EVERY_N_FRAMES`: Skip frames for performance (e.g., 1 = every frame, 2 = every other)
- Models auto-downloaded on first use to `YOLO_HOME/weights/`

---

## Entry Points

### **Main Entry:** `run_penalty_analizer.py`

```python
DEFAULT_INPUT = Path("data/penalties_mbappe_1.mp4")
DEFAULT_OUTPUT_DIR = Path("data/cv_output")

pipeline = PenaltyPipeline()
result = pipeline.process_video(
    input_video=input_video,
    output_video=output_path,
    show_preview=SHOW_WINDOW,  # Display live
    max_frames=MAX_FRAMES       # Limit for testing
)
```

**Configurable Variables:**
- `INPUT_VIDEO`: Input video path
- `OUTPUT_VIDEO`: Output path (auto-generated if None)
- `SHOW_WINDOW`: Show preview during processing
- `MAX_FRAMES`: Limit frame count for testing

### **Video Download:** `extract_videos/download_videos.py`
- Downloads penalty kick videos using yt-dlp
- Source URLs from `extract_videos/videos.json`

---

## Project Structure

```
Penalties-Prediction-By-Body-Language/
├── run_penalty_analizer.py          # Main entry point
├── requirements.txt                  # Dependencies
├── yolov8m-pose.pt                   # Pose model (optional)
├── yolov8m.pt                        # Detection model
├── AGENTS.md                         # This file
│
├── data/
│   ├── cv_output/                    # Output annotated videos
│   └── penalties_mbappe_1.mp4        # Input video
│
├── extract_videos/
│   ├── download_videos.py            # Video downloader
│   └── videos.json                   # Video URLs
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                   # Main pipeline orchestrator
│   ├── penalty_metrics.py            # Metrics extraction (optional)
│   ├── video_io.py                   # VideoReader/VideoWriter
│   │
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── ball_detector.py          # Ball tracking
│   │   ├── goal_detector.py          # Goal detection
│   │   └── players_detector.py       # Players + roles
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py                 # Centralized model config
│   │   └── README.md
│   │
│   └── pose/
│       ├── __init__.py
│       ├── angles.py                 # Angle calculations
│       └── pose_estimator.py         # Pose extraction (optional)
│
└── utils/
    └── drawings.py                   # Annotation utilities
```

---

## Key Algorithms & Heuristics

### Role Assignment (Shooter vs Goalkeeper)
**Pre-shot:**
```python
if player_center_x < goal_center_x:
    role = "shooter"
else:
    role = "goalkeeper"
```

**Post-shot:** Frozen map prevents reassignment even if positions cross.

### Goal Stabilization (EMA)
```python
alpha = 0.2  # 20% weight to new detection
stable_bbox = (1-alpha) * stable_bbox + alpha * new_detection_bbox
```
After 4 stable detections, enables goal-based ROI constraints.

### Shot Detection
```python
# Velocity threshold from 4-frame window
if distance(pos[-4], pos[-1]) > 18.0 pixels:
    shot_detected = True
    # Trigger strict tracking mode
```

### Ghost Detection (Occlusion Recovery)
```python
if track_missed_for > N_frames:
    ghost = {
        bbox: last_known_position,
        confidence: conf * (decay_factor ** n_frames),
        track_id: original_id
    }
    # Render with reduced opacity
    # Re-acquisition compatible with ByteTrack
```

---

## Configuration & Customization

### Adjust Detection Sensitivity
**File:** `src/models/config.py`
```python
ModelConfig.BALL_CONFIDENCE = 0.5      # Lower = more detections
ModelConfig.PLAYERS_CONFIDENCE = 0.3   # Already permissive
ModelConfig.PROCESS_EVERY_N_FRAMES = 2 # Skip 50% of frames
```

### Adjust Role Assignment Thresholds
**File:** `src/pipeline.py` → `_identify_roles_by_track_id()`
- Modify spatial heuristic (left/right of goal center)
- Adjust crossbar margin (crowd filter)
- Modify role lock timing

### Adjust Tracking Parameters
**Files:**
- `src/detectors/ball_detector.py`: BASE_MAX_DIST, ROI_RADIUS, MAX_MISSED_*
- `src/detectors/players_detector.py`: GK_MAX_GHOST_FRAMES, SH_MAX_GHOST_FRAMES

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 models |
| `opencv-python` | Video I/O + visualization |
| `mediapipe` | Optional pose support |
| `supervision` | Annotation utilities |
| `yt-dlp` | Video downloading |
| `imageio-ffmpeg` | FFmpeg backend |
| `numpy` | Array operations |

---

## Common Workflows

### Run Full Analysis
```bash
python run_penalty_analizer.py
```

### Process Subset of Frames (Testing)
```python
# Edit run_penalty_analizer.py
MAX_FRAMES = 300  # Process only 300 frames
SHOW_WINDOW = True
```

### Download New Videos
```bash
# Edit extract_videos/videos.json with new URLs, then:
python extract_videos/download_videos.py
```

### Extract Metrics from Output
```python
from src.penalty_metrics import extract_metrics
metrics = extract_metrics("data/cv_output/annotated_penalties_mbappe_1.mp4")
```

---

## Debugging & Troubleshooting

### Ball Detector Issues
- **Missing detections:** Increase `BALL_CONFIDENCE` tolerance or reduce `MAX_MISSED_PRE_SHOT`
- **Ghost ball:** Reduce `ROI_RADIUS` or lower `MIN_SPEED_FOR_DIR_FILTER`

### Role Assignment Issues
- **Goalkeeper labeled as shooter:** Check crossbar margin in `_identify_roles_by_track_id()`
- **Post-shot role flip:** Role should be frozen; verify `_shot_detected` flag is set

### Video Output Issues
- **Corrupted output:** Check FFmpeg installation (`imageio-ffmpeg`)
- **Missing frames:** Verify input video codec compatibility

---

## Future Enhancements

- [ ] Pose estimation integration (body angle, dive direction)
- [ ] Ball trajectory prediction (ML model)
- [ ] Goalkeeper reaction time metrics
- [ ] Multi-camera support
- [ ] Real-time streaming mode
- [ ] Database export (SQLite/PostgreSQL)

---

**Last Updated:** May 2026  
**Project Status:** Active - Penalty kick analysis pipeline with robust occlusion handling
