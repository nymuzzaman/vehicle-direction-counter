"""Run the detector on a single image and save the annotated result.

Useful for checking a model or a confidence threshold before processing a
whole video.

    python scripts/detect_image.py --image frame.jpg --backend yolo --weights data/vehiclesTraffic.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vehicle_tracking.config import COLORS  # noqa: E402
from vehicle_tracking.detectors import build_detector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--output", default="processed_frame.png", help="Output image path."
    )
    parser.add_argument("--backend", choices=["yolo", "roboflow"], default="yolo")
    parser.add_argument("--weights", default=None, help="YOLO weights (.pt).")
    parser.add_argument("--model_id", default="skyview-vehicle/4")
    parser.add_argument("--roboflow_api_key", default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--slicer", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    detector = build_detector(
        args.backend,
        weights_path=args.weights,
        model_id=args.model_id,
        roboflow_api_key=args.roboflow_api_key,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        slicer=args.slicer,
    )
    detections = detector(image)

    labels = [
        f"{name} {conf:.2f}"
        for name, conf in zip(
            detections.data.get("class_name", detections.class_id), detections.confidence
        )
    ]
    annotated = sv.BoxAnnotator(color=COLORS).annotate(image.copy(), detections)
    annotated = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK).annotate(
        annotated, detections, labels
    )
    cv2.imwrite(args.output, annotated)
    print(f"{len(detections)} vehicles detected -> {args.output}")


if __name__ == "__main__":
    main()
