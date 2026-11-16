"""Zone definitions and drawing constants.

Polygons are written in the pixel coordinates of a *reference* resolution and
are rescaled to the actual video resolution at runtime, so the same zones
work for the 1080p and the 4K version of the intersection footage.
"""

from typing import List, Tuple

import numpy as np
import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

# Direction names for the four exit zones, in the same order as ZONE_OUT_POLYGONS.
DIRECTIONS = ["North", "East", "South", "West"]

# --- Flow mode: 4 entry zones + 4 exit zones (one pair per road) -------------
FLOW_REFERENCE_WH: Tuple[int, int] = (1920, 1080)

ZONE_IN_POLYGONS: List[np.ndarray] = [
    np.array([[652, 214], [795, 62], [947, 205], [804, 357]]),
    np.array([[1275, 384], [1418, 232], [1570, 375], [1427, 527]]),
    np.array([[1127, 933], [1271, 781], [1423, 925], [1279, 1077]]),
    np.array([[478, 753], [621, 601], [773, 744], [630, 896]]),
]

ZONE_OUT_POLYGONS: List[np.ndarray] = [
    np.array([[622, 555], [766, 403], [614, 260], [470, 412]]),  # red
    np.array([[1052, 164], [1195, 12], [1347, 155], [1204, 307]]),  # green
    np.array([[1302, 754], [1445, 602], [1597, 745], [1454, 897]]),  # yellow
    np.array([[654, 931], [798, 779], [950, 922], [806, 1074]]),  # blue
]

# --- Occupancy mode: a single zone in the middle of the intersection ---------
OCCUPANCY_REFERENCE_WH: Tuple[int, int] = (3840, 2160)

OCCUPANCY_POLYGON: np.ndarray = np.array(
    [[2038, 444], [2775, 1108], [2062, 1963], [1332, 1211]]
)


def scale_polygons(
    polygons: List[np.ndarray],
    reference_wh: Tuple[int, int],
    target_wh: Tuple[int, int],
) -> List[np.ndarray]:
    """Rescale polygons from ``reference_wh`` to ``target_wh`` pixel space."""
    sx = target_wh[0] / reference_wh[0]
    sy = target_wh[1] / reference_wh[1]
    return [
        np.round(polygon * np.array([sx, sy])).astype(int) for polygon in polygons
    ]
