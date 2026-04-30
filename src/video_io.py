"""Video input/output utilities for reading and writing penalty analysis videos."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


class VideoReader:
    """Simple iterator for reading video frames."""
    
    def __init__(self, video_path: str | Path):
        """Initialize video reader.
        
        Args:
            video_path: Path to input video file.
            
        Raises:
            RuntimeError: If video cannot be opened.
        """
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over frames with frame index.
        
        Yields:
            Tuple of (frame_index, frame) where frame is BGR numpy array.
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    
    def release(self):
        """Release video reader resources."""
        self.cap.release()
    
    def __del__(self):
        self.release()


class VideoWriter:
    """Write frames to MP4 video file."""
    
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        """Initialize video writer.
        
        Args:
            output_path: Output video file path.
            fps: Frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
            
        Raises:
            RuntimeError: If video writer cannot be initialized.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video.
        
        Args:
            frame: Frame as BGR numpy array.
        """
        self.writer.write(frame)
    
    def release(self):
        """Release video writer resources and finalize file."""
        self.writer.release()
    
    def __del__(self):
        self.release()