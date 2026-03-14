"""Face detection and ROI extraction using MediaPipe."""

import numpy as np
import cv2
from typing import Tuple, Optional


class FaceDetector:
    """Detect face and extract ROI using MediaPipe."""

    def __init__(self, roi_size: int = 128, padding: float = 0.1):
        self.roi_size = roi_size
        self.padding = padding
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        except ImportError:
            raise ImportError("mediapipe required. Install with: pip install mediapipe")

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.

        Args:
            frame: (H, W, 3) RGB image

        Returns:
            bbox: (x, y, w, h) bounding box or None if no face detected
        """
        h, w = frame.shape[:2]
        results = self.detector.process(frame)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # Add padding
        pad_x = int(box_w * self.padding)
        pad_y = int(box_h * self.padding)

        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        box_w = min(w - x, box_w + 2 * pad_x)
        box_h = min(h - y, box_h + 2 * pad_y)

        return (x, y, box_w, box_h)

    def extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and resize face ROI.

        Args:
            frame: (H, W, 3) input frame
            bbox: (x, y, w, h) bounding box

        Returns:
            roi: (roi_size, roi_size, 3) resized face region
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (self.roi_size, self.roi_size))
        return roi_resized

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and extract ROI in one call."""
        bbox = self.detect_face(frame)
        if bbox is None:
            return None
        return self.extract_roi(frame, bbox)
