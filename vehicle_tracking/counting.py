"""Zone bookkeeping: which vehicle came from where and where it went."""

from typing import Dict, List, Set

import numpy as np
import supervision as sv


class FlowCounter:
    """Counts vehicles per (entry zone -> exit zone) pair.

    A tracked vehicle is assigned to the first entry zone it is seen in. When it
    later shows up in an exit zone, its tracker id is added to
    ``counts[exit_zone][entry_zone]``. Using sets means each vehicle is counted
    once, no matter how many frames it spends inside the exit zone.
    """

    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, in_zone in enumerate(detections_in_zones):
            for tracker_id in in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, out_zone in enumerate(detections_out_zones):
            for tracker_id in out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        # Re-use class_id as "entry zone id" so annotators colour each vehicle
        # by the road it came from. Vehicles not yet seen in an entry zone are
        # dropped from the output.
        if len(detections) > 0:
            detections.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections.tracker_id)
        else:
            detections.class_id = np.array([], dtype=int)
        return detections[detections.class_id != -1]

    def count(self, zone_out_id: int, zone_in_id: int) -> int:
        return len(self.counts.get(zone_out_id, {}).get(zone_in_id, ()))


class OccupancyCounter:
    """Tracks how many vehicles are inside a single zone.

    ``current`` is the number of vehicles inside the zone in the latest frame;
    ``total`` is the number of distinct vehicles that have ever entered it.
    """

    def __init__(self) -> None:
        self.current: int = 0
        self.seen: Set[int] = set()

    @property
    def total(self) -> int:
        return len(self.seen)

    def update(self, detections_in_zone: sv.Detections) -> sv.Detections:
        self.current = len(detections_in_zone)
        self.seen.update(detections_in_zone.tracker_id.tolist())
        return detections_in_zone
