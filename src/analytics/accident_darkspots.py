"""
accident_darkspots.py — BONUS ANALYTICS for Category 4 (Dark / Black Spots)

A "dark spot" (black spot) is a road segment/intersection where accidents happen
repeatedly. So the analytic is: detect accident events, attach each to a location,
then RANK locations by accident frequency.

Required outputs:
  1. Detect accidents on roads      -> handled by the trained classifier/model
  2. Identify dark spots by frequency -> aggregate_darkspots() below

Location can come from camera_id, GPS, or a grid cell. We keep it generic: any
hashable 'location' key works.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class DarkSpot:
    location: str
    accident_count: int
    risk_level: str  # Low / Medium / High


# Frequency thresholds for risk banding (tune to your data volume).
HIGH_MIN = 5
MEDIUM_MIN = 2


def _risk(count: int) -> str:
    if count >= HIGH_MIN:
        return "High"
    if count >= MEDIUM_MIN:
        return "Medium"
    return "Low"


def aggregate_darkspots(accident_events: list[dict]) -> list[DarkSpot]:
    """
    accident_events: list of {"location": <key>, ...} — one entry per detected
    accident. Returns dark spots sorted by accident frequency (highest first).
    """
    counts = Counter(e["location"] for e in accident_events)
    spots = [DarkSpot(loc, n, _risk(n)) for loc, n in counts.most_common()]
    return spots


def darkspots_to_dataframe(spots: list[DarkSpot]):
    import pandas as pd
    return pd.DataFrame([s.__dict__ for s in spots])


if __name__ == "__main__":
    events = (
        [{"location": "MG_Road_Junction"}] * 6
        + [{"location": "Ring_Road_KM12"}] * 3
        + [{"location": "Sector7_Cross"}] * 1
    )
    for s in aggregate_darkspots(events):
        print(s)
