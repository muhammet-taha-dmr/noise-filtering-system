import numpy as np


def apply_lowpass_filter(data: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)
    spectrum[freqs > cutoff_hz] = 0
    filtered = np.fft.irfft(spectrum, n=len(data))
    return filtered.astype(np.float32)


def get_spectrum_magnitudes(data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)
    magnitudes = np.abs(spectrum) / len(data)
    return freqs, magnitudes
