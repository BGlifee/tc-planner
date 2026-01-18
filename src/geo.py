# src/geo.py
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


# Earth's mean radius in kilometers (WGS84-ish)
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance between two points on Earth (lat/lng in degrees).
    Returns distance in kilometers.
    """
    # convert degrees -> radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def within_radius_km(
    center_lat: float,
    center_lng: float,
    points: Iterable[Tuple[float, float]],
    radius_km: float,
) -> List[bool]:
    """
    Given a center and an iterable of (lat, lng) points,
    return a list of booleans: True if within radius_km.
    """
    out: List[bool] = []
    for lat, lng in points:
        d = haversine_km(center_lat, center_lng, lat, lng)
        out.append(d <= radius_km)
    return out


def distance_km_list(
    center_lat: float,
    center_lng: float,
    points: Iterable[Tuple[float, float]],
) -> List[float]:
    """
    Given a center and an iterable of (lat, lng) points,
    return distances (km) in the same order.
    """
    dists: List[float] = []
    for lat, lng in points:
        dists.append(haversine_km(center_lat, center_lng, lat, lng))
    return dists


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))
