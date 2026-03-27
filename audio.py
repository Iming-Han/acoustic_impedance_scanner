import signal

import numpy as np
import sounddevice as sd

class SignalGenerator:
    def __init__(self, fs: int = 48000, amplitude_db: float = -10):
        self.fs = fs
        self.amplitude_db = amplitude_db
    def generate_sweep(self, duration: float = 2, f_start: float = 20, f_end: float = 20000) -> np.ndarray:
        num_sample = int(duration * self.fs)
        t = np.linspace(0, duration, num_sample, endpoint=False)
        # log sweep: f(t) = f_start * (f_end/f_start)^(t/duration)
        k = np.log(f_end / f_start) / duration
        phase = 2 * np.pi * f_start * (np.exp(k * t) - 1) / k
        sweep = np.sin(phase)
        # apply amplitude
        amplitude = 10 ** (self.amplitude_db / 20)
        sweep *= amplitude
        return sweep.astype(np.float32)

class AudioEngine:
    def __init__(self, fs: int = 48000):
        self.fs = fs
        self.input_device_id = None
        self.output_device_id = None
        self.output_channels = 1
        self.input_channels = 1
    
    def list_devices(self):
        print(f"{'ID':<3} {'Name':<40} {'API':<10} {'In':<4} {'Out':<4}")
        print("-" * 70)
        
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        for idx, device in enumerate(devices):
            # get Host API name (e.g. MME, Windows WASAPI, ASIO)
            api_name = hostapis[device['hostapi']]['name']
            
            # highlight (optional): if device supports both input and output, it is usually the sound card we want
            mark = ""
            if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
                mark = "*" # mark as full-duplex device
            
            print(f"{idx:<3} {device['name'][:38]:<40} {api_name:<10} "
                  f"{device['max_input_channels']:<4} {device['max_output_channels']:<4} {mark}")
        
        print("-" * 70)
        print(f"Default Input: {sd.default.device[0]}, Default Output: {sd.default.device[1]}")
    def select_device(self, input_id: int = None, output_id: int = None):
        """
        Specify the device ID to use.
        If not specified(None), then use the system default device.
        """
        # simple check: check if ID exists
        devices = sd.query_devices()
        max_id = len(devices) - 1

        if input_id is not None:
            if 0 <= input_id <= max_id:
                if devices[input_id]['max_input_channels'] > 0:
                    self.input_device_id = input_id
                    print(f"-> Selected input device [{input_id}]: {devices[input_id]['name']}")
                else:
                    print(f"Warning: Device [{input_id}] does not support input!")
            else:
                raise ValueError(f"Input device ID {input_id} is invalid")

        if output_id is not None:
            if 0 <= output_id <= max_id:
                if devices[output_id]['max_output_channels'] > 0:
                    self.output_device_id = output_id
                    print(f"-> Selected output device [{output_id}]: {devices[output_id]['name']}")
                else:
                    print(f"Warning: Device [{output_id}] does not support output!")
            else:
                raise ValueError(f"Output device ID {output_id} is invalid")

        # update sounddevice default device (optional, for convenience)
        sd.default.device = (self.input_device_id, self.output_device_id)
    def play_record(self, signal: np.ndarray, input_channel_idx: int = 1, output_channel_idx: int = 1) -> np.ndarray:
        """
        Play and record (full-duplex).
        """
        if self.input_device_id is None or self.output_device_id is None:
            print("-> Using system default device for measurement.")

        # check signal dimension, if single channel signal (N,), to prevent error, may need to reshape
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        print(f"-> Measuring... (fs={self.fs}, Input Device={self.input_device_id}, Output Device={self.output_device_id})")
        extra_sec = 1.0
        extra_frames = int(extra_sec * self.fs)

        pad = np.zeros((extra_frames, signal.shape[1]), dtype=signal.dtype)
        signal_padded = np.concatenate([signal, pad], axis=0)
        # core function: playrec
        # blocking=True will block until playback and recording are completed, suitable for measurement program
        recording = sd.playrec(
            signal_padded, 
            samplerate=self.fs, 
            channels=self.input_channels, 
            dtype='float32',
            device=(self.input_device_id, self.output_device_id), # (Input, Output)
            input_mapping=input_channel_idx,
            output_mapping=output_channel_idx, 
            blocking=True
        )
        recording = np.asarray(recording)
        return recording
if __name__ == "__main__":
    # 1. instantiate engine
    engine = AudioEngine(fs=48000)
    # 2. list all devices
    # user first look at this list, decide which ID to use
    print("Scanning audio devices...")
    engine.list_devices()
    input_device_id = 5
    output_device_id = 5
    engine.select_device(input_id=input_device_id, output_id=output_device_id)
    gen = SignalGenerator(fs=48000, amplitude_db=-10)
    sweep_signal = gen.generate_sweep(duration=2.0) # 生成2秒扫频

    # 5. 执行测量
    print("\n准备开始播放录音...")
    recorded_data = engine.play_record(sweep_signal, input_channel_idx=1, output_channel_idx=10)