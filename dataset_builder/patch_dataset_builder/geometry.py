"""Geometric utilities for forward projection in geographic space."""

from __future__ import annotations

import math
from typing import Tuple

from .constants import EARTH_RADIUS_M, KNOT_TO_MPS


def project_point_by_sog_cog(
    lon_deg: float,
    lat_deg: float,
    sog_knots: float,
    cog_deg: float,
    delta_seconds: float,
) -> Tuple[float, float]:
    """Project a trajectory point forward using SOG and COG.

    The projection is spherical and intentionally lightweight. It is suitable for
    fixed-step future point generation in preprocessing, where the goal is to create
    deterministic intermediate supervision targets rather than high-fidelity marine
    navigation simulation.

    Parameters
    ----------
    lon_deg, lat_deg:
        Current longitude and latitude in degrees.
    sog_knots:
        Speed over ground in knots.
    cog_deg:
        Course over ground in degrees.
    delta_seconds:
        Forward projection interval in seconds.

    Returns
    -------
    tuple of float
        Projected longitude and latitude in degrees.
    """
    if delta_seconds <= 0:
        return float(lon_deg), float(lat_deg)

    speed_mps = max(float(sog_knots), 0.0) * KNOT_TO_MPS
    distance_m = speed_mps * float(delta_seconds)

    brng = math.radians(cog_deg % 360.0)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    ang = distance_m / EARTH_RADIUS_M

    sin_lat2 = math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    sin_lat2 = min(1.0, max(-1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(lon2)), float(math.degrees(lat2))
