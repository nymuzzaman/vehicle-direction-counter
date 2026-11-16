"""Detection backends.

A detector is any callable ``frame -> sv.Detections``. Two backends are
provided: a local Ultralytics YOLO model and a hosted Roboflow model. Either can
be wrapped in an ``sv.InferenceSlicer`` (SAHI-style tiling), which helps a lot
with small vehicles in high-resolution aerial footage.
"""

import os
from typing import Callable, Optional

import numpy as np
import supervision as sv

Detector = Callable[[np.ndarray], sv.Detections]


def yolo_detector(weights_path: str, confidence: float, iou: float) -> Detector:
    from ultralytics import YOLO

    model = YOLO(weights_path)

    def detect(frame: np.ndarray) -> sv.Detections:
        result = model(frame, verbose=False, conf=confidence, iou=iou)[0]
        return sv.Detections.from_ultralytics(result)

    return detect


def roboflow_detector(
    model_id: str, api_key: Optional[str], confidence: float, iou: float
) -> Detector:
    # Imported lazily so the YOLO backend works without the `inference` package.
    from inference.models.utils import get_roboflow_model

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Roboflow API key is missing. Pass --roboflow_api_key or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    model = get_roboflow_model(model_id=model_id, api_key=api_key)

    def detect(frame: np.ndarray) -> sv.Detections:
        result = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
        return sv.Detections.from_inference(result)

    return detect


def with_slicer(detector: Detector, slice_wh: int = 640) -> Detector:
    """Run ``detector`` on overlapping tiles and merge the results."""
    slicer = sv.InferenceSlicer(callback=detector, slice_wh=(slice_wh, slice_wh))
    return slicer


def build_detector(
    backend: str,
    *,
    weights_path: Optional[str] = None,
    model_id: str = "skyview-vehicle/4",
    roboflow_api_key: Optional[str] = None,
    confidence: float = 0.3,
    iou: float = 0.7,
    slicer: bool = False,
    slice_wh: int = 640,
) -> Detector:
    if backend == "yolo":
        if not weights_path:
            raise ValueError("--weights is required for the yolo backend.")
        detector = yolo_detector(weights_path, confidence, iou)
    elif backend == "roboflow":
        detector = roboflow_detector(model_id, roboflow_api_key, confidence, iou)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return with_slicer(detector, slice_wh) if slicer else detector
