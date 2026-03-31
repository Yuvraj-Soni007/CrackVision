"""
Image Loader Module
====================
Handles loading, resizing, colour-space conversion, and basic validation
of input images.  Supports single images, directories, and video frames.

CV Syllabus: Digital Image Formation & Low-Level Processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

from ..utils import load_config, logger


class ImageLoader:
    """Load and prepare images for the crack-detection pipeline."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        pre = cfg["preprocessing"]
        self.target_size: Tuple[int, int] = tuple(pre["target_size"])
        self.color_space: str = pre["color_space"]

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------
    def load(self, path: str) -> np.ndarray:
        """Load an image from disk, resize, and convert colour space."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return self._convert_color(img)

    def load_pair(self, left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a stereo image pair (for depth estimation)."""
        return self.load(left_path), self.load(right_path)

    # ------------------------------------------------------------------
    # Batch / directory
    # ------------------------------------------------------------------
    def load_directory(self, directory: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp")) -> List[Tuple[str, np.ndarray]]:
        """Load all images from a directory.  Returns list of (filename, image)."""
        results = []
        for p in sorted(Path(directory).iterdir()):
            if p.suffix.lower() in extensions:
                try:
                    img = self.load(str(p))
                    results.append((p.name, img))
                except Exception as e:
                    logger.warning(f"Skipping {p.name}: {e}")
        logger.info(f"Loaded {len(results)} images from {directory}")
        return results

    # ------------------------------------------------------------------
    # Video frames
    # ------------------------------------------------------------------
    def load_video_frames(self, video_path: str, max_frames: int = 100, skip: int = 1) -> List[np.ndarray]:
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % (skip + 1) == 0:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                frames.append(self._convert_color(frame))
            idx += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _convert_color(self, img: np.ndarray) -> np.ndarray:
        if self.color_space == "gray":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_space == "hsv":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.color_space == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # keep BGR

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        """Convert any image to grayscale (safe if already gray)."""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

    @staticmethod
    def to_bgr(img: np.ndarray) -> np.ndarray:
        """Ensure image is BGR (3-channel)."""
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
