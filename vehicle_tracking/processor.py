"""Video pipeline: detect -> track -> zone logic -> annotate -> write/show."""

from typing import Iterable, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from vehicle_tracking.annotation import FrameAnnotator
from vehicle_tracking.config import (
    FLOW_REFERENCE_WH,
    OCCUPANCY_POLYGON,
    OCCUPANCY_REFERENCE_WH,
    ZONE_IN_POLYGONS,
    ZONE_OUT_POLYGONS,
    scale_polygons,
)
from vehicle_tracking.counting import FlowCounter, OccupancyCounter
from vehicle_tracking.detectors import Detector


def make_zones(
    polygons: List[np.ndarray],
    anchors: Iterable[sv.Position] = (sv.Position.CENTER,),
) -> List[sv.PolygonZone]:
    return [sv.PolygonZone(polygon=p, triggering_anchors=anchors) for p in polygons]


class VideoProcessor:
    """Runs the full pipeline over a video.

    ``mode="flow"`` counts vehicles per entry/exit road pair (direction
    analysis). ``mode="occupancy"`` counts vehicles inside one central zone.
    """

    def __init__(
        self,
        detector: Detector,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        mode: str = "flow",
        dashboard: bool = False,
    ) -> None:
        if mode not in ("flow", "occupancy"):
            raise ValueError(f"Unknown mode: {mode!r}")

        self.detector = detector
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.mode = mode
        self.dashboard = dashboard

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.tracker = sv.ByteTrack(frame_rate=self.video_info.fps)
        self.annotator = FrameAnnotator()

        wh = self.video_info.resolution_wh
        if mode == "flow":
            self.zones_in = make_zones(
                scale_polygons(ZONE_IN_POLYGONS, FLOW_REFERENCE_WH, wh)
            )
            self.zones_out = make_zones(
                scale_polygons(ZONE_OUT_POLYGONS, FLOW_REFERENCE_WH, wh)
            )
            self.flow_counter = FlowCounter()
        else:
            (polygon,) = scale_polygons(
                [OCCUPANCY_POLYGON], OCCUPANCY_REFERENCE_WH, wh
            )
            (self.zone,) = make_zones([polygon])
            self.occupancy_counter = OccupancyCounter()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detector(frame)
        # Treat every detection as a generic "vehicle" so the tracker does not
        # split one car into two tracks when its predicted class flickers.
        detections.class_id = np.zeros(len(detections), dtype=int)
        detections = self.tracker.update_with_detections(detections)

        if self.mode == "occupancy":
            inside = detections[self.zone.trigger(detections=detections)]
            inside = self.occupancy_counter.update(inside)
            return self.annotator.draw_occupancy(
                frame, inside, self.zone, self.occupancy_counter
            )

        in_zones = [detections[z.trigger(detections=detections)] for z in self.zones_in]
        out_zones = [
            detections[z.trigger(detections=detections)] for z in self.zones_out
        ]
        detections = self.flow_counter.update(detections, in_zones, out_zones)
        return self.annotator.draw_flow(
            frame,
            detections,
            self.zones_in,
            self.zones_out,
            self.flow_counter,
            dashboard=self.dashboard,
        )

    def process_video(self) -> None:
        frames = sv.get_video_frames_generator(source_path=self.source_video_path)
        total = self.video_info.total_frames

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frames, total=total):
                    sink.write_frame(self.process_frame(frame))
            return

        for frame in tqdm(frames, total=total):
            cv2.imshow("Vehicle tracking (press q to quit)", self.process_frame(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
