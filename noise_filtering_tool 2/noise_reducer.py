# Spectral subtraction based noise reducer using STFT overlap-add.

import numpy as np


class SpectralNoiseReducer:
    def __init__(
        self,
        frame_size: int,
        hop: int,
        sample_rate: int,
        calib_frames: int,
        over_subtraction: float = 2.0,
        spectral_floor: float = 0.05,
        cutoff_hz: float | None = None,
        noise_update: float = 0.05,
    ) -> None:
        self.frame_size = frame_size
        self.hop = hop
        self.sample_rate = sample_rate
        self.calib_frames = calib_frames
        self.alpha = over_subtraction      
        self.beta = spectral_floor          
        self.noise_update = noise_update     

        
        self.window = np.sqrt(np.hanning(frame_size)).astype(np.float32)
        self._norm = self._cola_norm()

        self.in_buf = np.zeros(frame_size, dtype=np.float32)
        self.out_buf = np.zeros(frame_size, dtype=np.float32)

        n_bins = frame_size // 2 + 1
        self.noise_mag = np.zeros(n_bins, dtype=np.float64)
        self._frames_seen = 0

        
        if cutoff_hz is not None:
            freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
            self.lp_mask = (freqs <= cutoff_hz).astype(np.float32)
        else:
            self.lp_mask = None

        
        self._energy_avg = 0.0

    def _cola_norm(self) -> float:
        """Steady-state sum of squared windows across overlapping hops."""
        n = self.frame_size
        length = n * 4
        acc = np.zeros(length)
        w2 = self.window ** 2
        for start in range(0, length - n, self.hop):
            acc[start : start + n] += w2
        return float(acc[length // 2])

    @property
    def calibrating(self) -> bool:
        return self._frames_seen < self.calib_frames

    def calibration_progress(self) -> float:
        if self.calib_frames == 0:
            return 1.0
        return min(self._frames_seen / self.calib_frames, 1.0)

    def process(self, hop_samples: np.ndarray) -> np.ndarray:

        self.in_buf = np.roll(self.in_buf, -self.hop)
        self.in_buf[-self.hop:] = hop_samples

        windowed = self.in_buf * self.window
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        if self.calibrating:
    
            self.noise_mag += mag
            self._frames_seen += 1
            if self._frames_seen == self.calib_frames:
                self.noise_mag /= max(self.calib_frames, 1)
            clean_mag = mag  
        else:
            self._frames_seen += 1
            clean_mag = mag - self.alpha * self.noise_mag
            floor = self.beta * mag
            clean_mag = np.maximum(clean_mag, floor)

            frame_energy = float(np.mean(mag))
            self._energy_avg = 0.95 * self._energy_avg + 0.05 * frame_energy
            if frame_energy < 1.5 * self._energy_avg:
                self.noise_mag = (
                    (1 - self.noise_update) * self.noise_mag
                    + self.noise_update * mag
                )

        if self.lp_mask is not None:
            clean_mag = clean_mag * self.lp_mask

        clean_spectrum = clean_mag * np.exp(1j * phase)
        frame = np.fft.irfft(clean_spectrum, n=self.frame_size).astype(np.float32)
        frame *= self.window  

    
        self.out_buf += frame / self._norm
        out = self.out_buf[: self.hop].copy()
        self.out_buf = np.roll(self.out_buf, -self.hop)
        self.out_buf[-self.hop:] = 0.0
        return out

    def spectrum_for_display(self) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.fft.rfftfreq(self.frame_size, d=1.0 / self.sample_rate)
        return freqs, self.noise_mag.astype(np.float32)
