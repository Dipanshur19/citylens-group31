"""
fire_severity.py — BONUS ANALYTICS for Category 1 (Fire & Smoke)

Three feature outputs required by the rubric:
  1. Severity classification    -> Mild / Moderate / Severe
  2. Vulnerability classification -> Mild / Moderate / Severe
     (based on how crowded/dense the locality is: people + building count)
  3. Burning / waste area estimation in square metres

Design choice: these are rule-based on top of YOLO detections so they are
transparent and explainable to judges (no extra training needed). All thresholds
are tunable constants at the top of the file.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Tunable thresholds ----------------------------------------------------
# Fraction of the frame area covered by fire/smoke boxes.
SEVERITY_MILD_MAX = 0.05      # < 5% of frame  -> Mild
SEVERITY_MODERATE_MAX = 0.20  # 5-20% of frame -> Moderate, else Severe

# Vulnerability = how many people/buildings are exposed.
VULN_MILD_MAX = 2             # <= 2 exposed objects -> Mild
VULN_MODERATE_MAX = 8         # 3-8 -> Moderate, else Severe

# Real-world scale: metres represented by one pixel along one axis.
# Calibrate per camera if known; default assumes ~0.02 m/px (tune this!).
DEFAULT_METRES_PER_PIXEL = 0.02

FIRE_CLASSES = {"fire", "flame", "burning", "waste_fire"}
SMOKE_CLASSES = {"smoke"}
PEOPLE_CLASSES = {"person", "people"}
BUILDING_CLASSES = {"building", "house", "structure"}


@dataclass
class FireReport:
    severity: str
    vulnerability: str
    burning_area_sqm: float
    fire_coverage_pct: float
    exposed_objects: int


def _level(value: float, mild_max: float, moderate_max: float) -> str:
    if value <= mild_max:
        return "Mild"
    if value <= moderate_max:
        return "Moderate"
    return "Severe"


def analyse_frame(
    detections: list[dict],
    frame_w: int,
    frame_h: int,
    metres_per_pixel: float = DEFAULT_METRES_PER_PIXEL,
) -> FireReport:
    """
    detections: list of dicts with keys: class, x, y, width, height (pixels).
    Returns a FireReport with severity, vulnerability and burning area.
    """
    frame_area = max(frame_w * frame_h, 1)

    fire_px_area = 0.0
    exposed = 0
    for d in detections:
        cls = str(d["class"]).lower()
        box_area = float(d["width"]) * float(d["height"])
        if cls in FIRE_CLASSES or cls in SMOKE_CLASSES:
            fire_px_area += box_area
        if cls in PEOPLE_CLASSES or cls in BUILDING_CLASSES:
            exposed += 1

    coverage = fire_px_area / frame_area
    severity = _level(coverage, SEVERITY_MILD_MAX, SEVERITY_MODERATE_MAX)
    vulnerability = _level(exposed, VULN_MILD_MAX, VULN_MODERATE_MAX)

    # Area in real-world square metres = pixel_area * (m/px)^2
    burning_area_sqm = fire_px_area * (metres_per_pixel ** 2)

    return FireReport(
        severity=severity,
        vulnerability=vulnerability,
        burning_area_sqm=round(burning_area_sqm, 2),
        fire_coverage_pct=round(coverage * 100, 2),
        exposed_objects=exposed,
    )


if __name__ == "__main__":
    # tiny self-test
    demo = [
        {"class": "fire", "x": 100, "y": 100, "width": 300, "height": 250},
        {"class": "smoke", "x": 80, "y": 60, "width": 200, "height": 180},
        {"class": "person", "x": 500, "y": 400, "width": 40, "height": 90},
        {"class": "building", "x": 0, "y": 0, "width": 200, "height": 300},
    ]
    print(analyse_frame(demo, 1280, 720))
