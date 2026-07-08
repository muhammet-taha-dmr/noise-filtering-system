import sys
import numpy as np


BAR_CHAR = "█"
EMPTY_CHAR = "░"
WIDTH = 60
HEIGHT = 8
FREQ_BANDS = 20


def _build_frame(freqs: np.ndarray, magnitudes: np.ndarray, cutoff_hz: float | None) -> str:
  
    if cutoff_hz is None:
        cutoff_hz = float("inf")
    max_freq = freqs[-1] if len(freqs) > 0 else 1
    band_width = max_freq / FREQ_BANDS
    band_mags = []
    for i in range(FREQ_BANDS):
        lo = i * band_width
        hi = (i + 1) * band_width
        mask = (freqs >= lo) & (freqs < hi)
        band_mags.append(float(np.mean(magnitudes[mask])) if mask.any() else 0.0)

    peak = max(band_mags) if max(band_mags) > 0 else 1.0
    normalized = [min(v / peak, 1.0) for v in band_mags]

    lines = []
    for row in range(HEIGHT, 0, -1):
        threshold = row / HEIGHT
        line = ""
        for col, val in enumerate(normalized):
            freq_center = (col + 0.5) * band_width
            is_noise = freq_center > cutoff_hz
            if val >= threshold:
                line += "\033[91m█\033[0m" if is_noise else "\033[92m█\033[0m"
            else:
                line += "░"
            line += " "
        lines.append(line)

    freq_labels = f"  0Hz{' ' * (WIDTH - 12)}{int(max_freq / 1000)}kHz"
    lines.append(freq_labels)
    if np.isfinite(cutoff_hz) and cutoff_hz < max_freq:
        cutoff_pos = int((cutoff_hz / max_freq) * (FREQ_BANDS * 2))
        marker = " " * cutoff_pos + f"↑{int(cutoff_hz)}Hz cutoff"
        lines.append(marker)
    else:
        lines.append("  green=kept  red=removed")
    return "\n".join(lines)


class SpectrumVisualizer:
    def __init__(self) -> None:
        self._first = True
        self._frame_lines = HEIGHT + 2

    def update(self, freqs: np.ndarray, magnitudes: np.ndarray, cutoff_hz: float) -> None:
        frame = _build_frame(freqs, magnitudes, cutoff_hz)
        if not self._first:
            sys.stdout.write(f"\033[{self._frame_lines}A\033[J")
        sys.stdout.write(frame + "\n")
        sys.stdout.flush()
        self._first = False
