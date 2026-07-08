import threading
import numpy as np
import pyaudio
from filters import apply_lowpass_filter, get_spectrum_magnitudes
from noise_reducer import SpectralNoiseReducer
from visualizer import SpectrumVisualizer


FORMAT = pyaudio.paFloat32


class AudioProcessor:
    def __init__(
        self,
        mode: str = "spectral",
        cutoff_hz: float | None = 3000.0,
        sample_rate: int = 44100,
        chunk: int = 1024,
        channels: int = 1,
        input_device: int | None = None,
        output_device: int | None = None,
        visualize: bool = False,
        calibrate_sec: float = 1.0,
        aggressiveness: float = 2.0,
        floor: float = 0.05,
    ) -> None:
        self.mode = mode
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_device = input_device
        self.output_device = output_device
        self.visualize = visualize

        
        if mode == "spectral":
            self.frame_size = chunk
            self.hop = chunk // 2
            calib_frames = max(int(calibrate_sec * sample_rate / self.hop), 1)
            self.reducer = SpectralNoiseReducer(
                frame_size=self.frame_size,
                hop=self.hop,
                sample_rate=sample_rate,
                calib_frames=calib_frames,
                over_subtraction=aggressiveness,
                spectral_floor=floor,
                cutoff_hz=cutoff_hz,
            )
            self.read_size = self.hop
        else:
            self.frame_size = chunk
            self.hop = chunk
            self.reducer = None
            self.read_size = chunk

        self._pa = pyaudio.PyAudio()
        self._running = False
        self._thread: threading.Thread | None = None
        self._visualizer = SpectrumVisualizer() if visualize else None
        self._viz_counter = 0
        self._viz_every = 8
        self._announced_ready = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._pa.terminate()

    def _open_streams(self):
        in_kwargs: dict = dict(
            format=FORMAT, channels=self.channels, rate=self.sample_rate,
            input=True, frames_per_buffer=self.read_size,
        )
        out_kwargs: dict = dict(
            format=FORMAT, channels=self.channels, rate=self.sample_rate,
            output=True, frames_per_buffer=self.read_size,
        )
        if self.input_device is not None:
            in_kwargs["input_device_index"] = self.input_device
        if self.output_device is not None:
            out_kwargs["output_device_index"] = self.output_device
        return self._pa.open(**in_kwargs), self._pa.open(**out_kwargs)

    def _run(self) -> None:
        in_stream, out_stream = self._open_streams()
        try:
            while self._running:
                raw = in_stream.read(self.read_size, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).copy()

                if self.mode == "spectral":
                    out_data = self.reducer.process(data)
                    self._maybe_status()
                else:
                    out_data = apply_lowpass_filter(data, self.cutoff_hz, self.sample_rate)

                self._maybe_visualize(data)
                out_stream.write(np.ascontiguousarray(out_data).tobytes())
        finally:
            for s in (in_stream, out_stream):
                s.stop_stream()
                s.close()

    def _maybe_status(self) -> None:
        if self.reducer is None:
            return
        if self.reducer.calibrating:
            pct = self.reducer.calibration_progress() * 100
            print(f"\r  [Calibrating background noise... {pct:3.0f}%]   ", end="", flush=True)
        elif not self._announced_ready:
            print("\r  [Noise profile learned — filtering active]          ")
            self._announced_ready = True

    def _maybe_visualize(self, data: np.ndarray) -> None:
        if not self._visualizer:
            return
        if self._viz_counter % self._viz_every == 0:
            freqs, mags = get_spectrum_magnitudes(data, self.sample_rate)
            self._visualizer.update(freqs, mags, self.cutoff_hz)
        self._viz_counter += 1


def list_devices(pa: pyaudio.PyAudio) -> None:
    print("\nAvailable audio devices:")
    print(f"  {'Index':<6} {'Name':<40} {'In':<5} {'Out':<5} {'Rate'}")
    print("  " + "-" * 65)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(
            f"  {i:<6} {info['name'][:38]:<40} "
            f"{int(info['maxInputChannels']):<5} "
            f"{int(info['maxOutputChannels']):<5} "
            f"{int(info['defaultSampleRate'])}"
        )
    print()
