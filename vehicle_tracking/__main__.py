"""Command-line entry point: ``python -m vehicle_tracking --help``."""

import argparse

from vehicle_tracking.detectors import build_detector
from vehicle_tracking.processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vehicle_tracking",
        description="Vehicle detection, tracking, counting and direction analysis.",
    )
    parser.add_argument(
        "--source_video_path", required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Where to save the annotated video. If omitted, frames are shown "
        "in a window instead (press q to quit).",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--backend",
        choices=["yolo", "roboflow"],
        default="yolo",
        help="Detection backend (default: yolo).",
    )
    model.add_argument(
        "--weights",
        "--source_weights_path",
        dest="weights",
        default=None,
        help="Path to YOLO weights (.pt). Required for --backend yolo.",
    )
    model.add_argument(
        "--model_id",
        default="skyview-vehicle/4",
        help="Roboflow model ID (default: skyview-vehicle/4).",
    )
    model.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API key. Falls back to the ROBOFLOW_API_KEY env variable.",
    )
    model.add_argument(
        "--confidence_threshold", type=float, default=0.3, help="Default: 0.3."
    )
    model.add_argument("--iou_threshold", type=float, default=0.7, help="Default: 0.7.")
    model.add_argument(
        "--slicer",
        action="store_true",
        help="Run detection on 640x640 tiles (better for small, distant vehicles; slower).",
    )
    model.add_argument(
        "--slice_size", type=int, default=640, help="Tile size for --slicer."
    )

    out = parser.add_argument_group("analysis")
    out.add_argument(
        "--mode",
        choices=["flow", "occupancy"],
        default="flow",
        help="flow: count vehicles per entry->exit road (default). "
        "occupancy: count vehicles inside one central zone.",
    )
    out.add_argument(
        "--dashboard",
        action="store_true",
        help="In flow mode, also draw a North/East/South/West count box in each corner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(
        args.backend,
        weights_path=args.weights,
        model_id=args.model_id,
        roboflow_api_key=args.roboflow_api_key,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        slicer=args.slicer,
        slice_wh=args.slice_size,
    )
    VideoProcessor(
        detector=detector,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        mode=args.mode,
        dashboard=args.dashboard,
    ).process_video()


if __name__ == "__main__":
    main()
