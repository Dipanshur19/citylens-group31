"""
streetlight_state.py — BONUS ANALYTICS for Category 3 (Damaged Street Lights)

Two required feature outputs:
  1. OFF-state detection      -> is a given light ON or OFF right now?
  2. Flickering detection     -> does a light switch ON/OFF repeatedly across
                                 consecutive video frames?

Approach:
- The YOLO model locates each street-light bounding box.
- ON vs OFF is decided by the brightness of the lamp region (a lit lamp is a
  bright blob; an OFF/damaged lamp is dark). This is robust and needs no extra
  labels. If your dataset already has explicit on/off/damaged classes, just read
  the predicted class instead and skip the brightness step.
- Flickering is detected by tracking each light across frames (by position) and
  counting ON<->OFF transitions over a sliding window.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

# --- Tunables --------------------------------------------------------------
BRIGHTNESS_ON_THRESHOLD = 130   # mean grayscale (0-255) above this => ON
FLICKER_MIN_TRANSITIONS = 3     # ON<->OFF switches within the window => flicker
FLICKER_WINDOW = 12             # number of recent frames considered
MATCH_DIST_PX = 40              # max centre distance to treat as the same light


def lamp_is_on(frame_bgr, box_xyxy) -> bool:
    """Decide ON/OFF from the brightness of the lamp region."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop
    # use the brightest 10% of pixels — a lit bulb is a small bright spot
    flat = np.sort(gray.ravel())
    top = flat[int(len(flat) * 0.9):]
    return float(top.mean()) >= BRIGHTNESS_ON_THRESHOLD


class FlickerTracker:
    """Tracks lights across video frames and flags flickering ones."""

    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}  # id -> {pos, history(deque)}

    def _match(self, cx, cy):
        for tid, t in self._tracks.items():
            px, py = t["pos"]
            if (px - cx) ** 2 + (py - cy) ** 2 <= MATCH_DIST_PX ** 2:
                return tid
        return None

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: [{cx, cy, on(bool)}...] for the current frame.
        Returns the same list annotated with light_id, state, flickering.
        """
        out = []
        for d in detections:
            tid = self._match(d["cx"], d["cy"])
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"pos": (d["cx"], d["cy"]),
                                     "history": deque(maxlen=FLICKER_WINDOW)}
            t = self._tracks[tid]
            t["pos"] = (d["cx"], d["cy"])
            t["history"].append(1 if d["on"] else 0)

            transitions = sum(
                1 for a, b in zip(t["history"], list(t["history"])[1:]) if a != b
            )
            out.append({
                "light_id": tid,
                "state": "ON" if d["on"] else "OFF",
                "flickering": transitions >= FLICKER_MIN_TRANSITIONS,
                "transitions": transitions,
            })
        return out
