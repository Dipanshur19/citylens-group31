"""
animal_dwell.py — BONUS ANALYTICS for Category 5 (Dead / Stray Animals)

Required feature outputs:
  1. Animal type categorization     -> from the YOLO predicted class
  2. Dead or Alive                  -> motion-based: an animal that never moves
                                       across many frames is flagged "dead"
  3. Count & dwell time             -> how many distinct animals + how long each
                                       stays in view (seconds)

We use YOLO's built-in tracker (ByteTrack) so each animal keeps a stable ID
across frames. Dwell time = (last_seen - first_seen) frames / fps. Movement is
the total centre displacement; near-zero movement over a long dwell => "Dead".
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- Tunables --------------------------------------------------------------
# If an animal is present this many seconds but moves less than MOVE_PX total,
# we classify it as "Dead" (lying still on the road).
DEAD_MIN_DWELL_SEC = 4.0
DEAD_MAX_MOVE_PX = 25.0


@dataclass
class Track:
    track_id: int
    animal_type: str
    first_frame: int
    last_frame: int
    last_pos: tuple[float, float]
    total_movement: float = 0.0
    positions: int = 0


@dataclass
class AnimalSummary:
    track_id: int
    animal_type: str
    dwell_time_sec: float
    status: str  # Dead / Alive
    total_movement_px: float


class AnimalDwellTracker:
    def __init__(self, fps: float = 25.0):
        self.fps = max(fps, 1.0)
        self.tracks: dict[int, Track] = {}

    def update(self, frame_idx: int, detections: list[dict]):
        """detections: [{track_id, animal_type, cx, cy}...] for current frame."""
        for d in detections:
            tid = d["track_id"]
            cx, cy = d["cx"], d["cy"]
            if tid not in self.tracks:
                self.tracks[tid] = Track(tid, d["animal_type"], frame_idx,
                                         frame_idx, (cx, cy))
            else:
                t = self.tracks[tid]
                dx = cx - t.last_pos[0]
                dy = cy - t.last_pos[1]
                t.total_movement += (dx * dx + dy * dy) ** 0.5
                t.last_pos = (cx, cy)
                t.last_frame = frame_idx
                t.positions += 1

    def summarise(self) -> list[AnimalSummary]:
        out = []
        for t in self.tracks.values():
            dwell = (t.last_frame - t.first_frame + 1) / self.fps
            is_dead = (dwell >= DEAD_MIN_DWELL_SEC
                       and t.total_movement <= DEAD_MAX_MOVE_PX)
            out.append(AnimalSummary(
                track_id=t.track_id,
                animal_type=t.animal_type,
                dwell_time_sec=round(dwell, 2),
                status="Dead" if is_dead else "Alive",
                total_movement_px=round(t.total_movement, 1),
            ))
        return out

    def total_count(self) -> int:
        return len(self.tracks)
