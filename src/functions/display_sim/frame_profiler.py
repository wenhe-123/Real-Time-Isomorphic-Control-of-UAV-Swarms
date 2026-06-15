"""Per-section wall-time profiler for the online control loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

@dataclass
class FrameSectionProfiler:
    """Accumulate per-section wall time in the integrated control loop (perf_counter)."""

    enabled: bool = False
    report_every: int = 60
    _active: bool = False
    _frame_t0: float = 0.0
    _last: float = 0.0
    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _frames: int = 0
    _skipped: int = 0

    def cancel(self) -> None:
        if self.enabled and self._active:
            self._skipped += 1
        self._active = False

    def frame_start(self) -> None:
        if not self.enabled:
            return
        self._active = True
        self._frame_t0 = self._last = time.perf_counter()

    def section(self, name: str) -> None:
        if not self.enabled or not self._active:
            return
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self._totals[name] = self._totals.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def frame_end(self, frame_idx: int) -> None:
        if not self.enabled or not self._active:
            return
        self.section("ui_wait")
        total = time.perf_counter() - self._frame_t0
        self._totals["frame_total"] = self._totals.get("frame_total", 0.0) + total
        self._counts["frame_total"] = self._counts.get("frame_total", 0) + 1
        self._active = False
        self._frames += 1
        every = max(1, int(self.report_every))
        if (frame_idx % every) != 0 and frame_idx > 0:
            return
        n = max(1, self._frames)
        parts: list[tuple[str, float]] = []
        for key, sec in self._totals.items():
            if key == "frame_total":
                continue
            cnt = max(1, self._counts.get(key, 0))
            parts.append((key, (sec / cnt) * 1000.0))
        parts.sort(key=lambda x: x[1], reverse=True)
        avg_frame_ms = (self._totals.get("frame_total", 0.0) / n) * 1000.0
        fps_est = 1000.0 / max(1e-3, avg_frame_ms)
        detail = " ".join(f"{k}={v:.1f}ms" for k, v in parts[:12])
        skip_txt = f" skip={self._skipped}" if self._skipped else ""
        print(
            f"[profile f={frame_idx} windows={n}{skip_txt}] "
            f"frame≈{avg_frame_ms:.1f}ms (~{fps_est:.1f}fps) | {detail}"
        )
        self._totals.clear()
        self._counts.clear()
        self._frames = 0
        self._skipped = 0
