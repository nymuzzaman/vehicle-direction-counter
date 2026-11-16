"""Drawing helpers for the annotated output frames."""

from typing import List

import cv2
import numpy as np
import supervision as sv

from vehicle_tracking.config import COLORS, DIRECTIONS
from vehicle_tracking.counting import FlowCounter, OccupancyCounter


class FrameAnnotator:
    def __init__(self) -> None:
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )

    def draw_vehicles(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        frame = self.trace_annotator.annotate(frame, detections)
        frame = self.box_annotator.annotate(frame, detections)
        return self.label_annotator.annotate(frame, detections, labels)

    def draw_flow(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        zones_in: List[sv.PolygonZone],
        zones_out: List[sv.PolygonZone],
        counter: FlowCounter,
        dashboard: bool = False,
    ) -> np.ndarray:
        frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(zones_in, zones_out)):
            frame = sv.draw_polygon(frame, zone_in.polygon, COLORS.colors[i])
            frame = sv.draw_polygon(frame, zone_out.polygon, COLORS.colors[i])

        frame = self.draw_vehicles(frame, detections)

        # Next to every exit zone, one coloured counter per entry zone the
        # vehicles came from (colour = origin road).
        for zone_out_id, zone_out in enumerate(zones_out):
            center = sv.get_polygon_center(polygon=zone_out.polygon)
            for i, zone_in_id in enumerate(counter.counts.get(zone_out_id, {})):
                frame = sv.draw_text(
                    scene=frame,
                    text=str(counter.count(zone_out_id, zone_in_id)),
                    text_anchor=sv.Point(x=center.x, y=center.y + 40 * i),
                    background_color=COLORS.colors[zone_in_id],
                )

        if dashboard:
            frame = draw_direction_dashboard(frame, counter, len(zones_out))
        return frame

    def draw_occupancy(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        zone: sv.PolygonZone,
        counter: OccupancyCounter,
    ) -> np.ndarray:
        frame = frame.copy()
        detections.class_id = np.zeros(len(detections), dtype=int)
        frame = self.draw_vehicles(frame, detections)
        frame = sv.draw_polygon(frame, zone.polygon, COLORS.colors[0])

        center = sv.get_polygon_center(polygon=zone.polygon)
        return sv.draw_text(
            scene=frame,
            text=f"Inside: {counter.current}  Total: {counter.total}",
            text_anchor=sv.Point(x=center.x, y=center.y + 40),
            background_color=COLORS.colors[0],
        )


def draw_direction_dashboard(
    frame: np.ndarray, counter: FlowCounter, num_zones: int, box_size: int = 150
) -> np.ndarray:
    """Draw a white box per exit direction in the frame corners.

    Each box shows the direction name and, along the bottom, one count per
    origin road coloured like that road's zones.
    """
    height, width = frame.shape[:2]
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    corners = [
        (10, 10),  # North: top-left
        (width - box_size - 10, 10),  # East: top-right
        (width - box_size - 10, height - box_size - 10),  # South: bottom-right
        (10, height - box_size - 10),  # West: bottom-left
    ]

    for zone_out_id, (direction, (x, y)) in enumerate(
        zip(DIRECTIONS[:num_zones], corners)
    ):
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 255, 255), -1)
        (tw, th), _ = cv2.getTextSize(direction, font, font_scale, thickness)
        cv2.putText(
            frame,
            direction,
            (x + (box_size - tw) // 2, y + th + 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        for i, zone_in_id in enumerate(counter.counts.get(zone_out_id, {})):
            text = str(counter.count(zone_out_id, zone_in_id))
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            color = COLORS.colors[zone_in_id]
            cv2.putText(
                frame,
                text,
                (x + 10 + i * (tw + 10), y + box_size - th - 5),
                font,
                font_scale,
                color.as_bgr(),
                thickness,
            )
    return frame
