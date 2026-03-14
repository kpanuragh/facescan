"""Tests for preprocessing module."""

import numpy as np
import pytest
from src.preprocessing.face_detector import FaceDetector


def test_face_detector_initialization():
    """Test FaceDetector can be initialized."""
    detector = FaceDetector(roi_size=128, padding=0.1)
    assert detector.roi_size == 128
    assert detector.padding == 0.1


def test_face_detection_with_synthetic_image():
    """Test face detection on synthetic image."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detector = FaceDetector()

    # Should not crash (may or may not detect face)
    bbox = detector.detect_face(frame)
    if bbox is not None:
        x, y, w, h = bbox
        assert x >= 0 and y >= 0 and w > 0 and h > 0


def test_roi_extraction():
    """Test ROI extraction and resizing."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    detector = FaceDetector(roi_size=128)

    bbox = (100, 100, 200, 200)
    roi = detector.extract_roi(frame, bbox)

    assert roi.shape == (128, 128, 3)
    assert roi.dtype == np.uint8


def test_process_frame():
    """Test complete frame processing (detect + extract)."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    detector = FaceDetector()

    # May return None if no face detected, which is valid
    roi = detector.process_frame(frame)
    if roi is not None:
        assert roi.shape == (128, 128, 3)

# Tests for model architecture
import torch
from src.models.architecture import SharedFeatureExtractor, TaskHead, rPPGModel


def test_shared_extractor():
    """Test SharedFeatureExtractor."""
    extractor = SharedFeatureExtractor(output_dim=256)
    x = torch.randn(4, 30, 3, 128, 128)  # (B, T, C, H, W)
    features = extractor(x)
    assert features.shape == (4, 256)


def test_task_head():
    """Test TaskHead."""
    head = TaskHead(input_dim=256, hidden_dim=128)
    x = torch.randn(4, 256)
    value, confidence = head(x)
    assert value.shape == (4,)
    assert confidence.shape == (4,)
    assert (confidence >= 0).all() and (confidence <= 1).all()


def test_rppg_model():
    """Test complete rPPGModel."""
    model = rPPGModel(feature_dim=256, hidden_dim=128)
    x = torch.randn(4, 30, 3, 128, 128)
    hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf = model(x)

    assert hr_val.shape == (4,)
    assert hr_conf.shape == (4,)
    assert rr_val.shape == (4,)
    assert rr_conf.shape == (4,)
    assert spo2_val.shape == (4,)
    assert spo2_conf.shape == (4,)

    # Check confidence scores are in [0, 1]
    for conf in [hr_conf, rr_conf, spo2_conf]:
        assert (conf >= 0).all() and (conf <= 1).all()
