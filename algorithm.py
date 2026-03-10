import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from functools import cached_property
class ImpedanceAnalyzer:
    # physical constants
    AIR_DENSITY = 1.225
    AIR_SPEED_OF_SOUND_0C = 331.3

    def __init__(self, 
                 mic1_time_data: np.ndarray, 
                 mic2_time_data: np.ndarray, 
                 mic1_pos: float, 
                 mic2_pos: float, 
                 temp_c: float = 20, 
                 fs: int = 48000):
        # basic acoustic parameters
        self.c = self.AIR_SPEED_OF_SOUND_0C * np.sqrt(1 + temp_c / 273.15)  # speed of sound, m/s
        self.temp_c = temp_c
        self.rho = self.AIR_DENSITY  # air density, kg/m³
        self.fs = fs
        # load time-domain data and mic positions
        self.mic1_time_data = mic1_time_data
        self.mic2_time_data = mic2_time_data
        self.mic1_pos = mic1_pos
        self.mic2_pos = mic2_pos
        self.s = mic1_pos - mic2_pos
        # init public attributes
        self.freqs = np.fft.rfftfreq(len(mic1_time_data), 1/fs)
        self.k = 2 * np.pi * self.freqs / self.c  # wavenumber

    @cached_property
    def H12(self)->np.ndarray:
        """compute frequency-domain transfer function H12 = X2 / X1"""
        N = len(self.mic1_time_data)
        X1 = np.fft.rfft(self.mic1_time_data, n=N)
        X2 = np.fft.rfft(self.mic2_time_data, n=N)
        H12 = X2 / (X1 + 1e-12)  # add small constant to prevent division by zero
        return H12
    @cached_property
    def reflection_factor(self)->np.ndarray:
        """compute reflection factor"""
        reflection_factor = (self.H12 - np.exp(1j * self.k * self.s)) / (-self.H12 + np.exp(-1j * self.k * self.s))*np.exp(-2j * self.k * self.mic1_pos)
        return reflection_factor
    @cached_property
    def impedance(self)->np.ndarray:
        return self.rho * self.c * (1 + self.reflection_factor) / (1 - self.reflection_factor)
    @cached_property
    def alpha(self)->np.ndarray:
        return 1 - np.abs(self.reflection_factor)**2

    def plot_results(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        freqs = self.k * self.c / (2 * np.pi)
        ax.plot(freqs, self.alpha, label='Absorption Coefficient')
        ax.set_title('Absorption Coefficient vs Frequency')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.legend()
        plt.show()
 
if __name__ == "__main__":
    fs = 48000
    t = np.linspace(0, 0.5, int(0.5 * fs))
    # 使用扫频信号，信噪比更好
    data1 = signal.chirp(t, f0=100, f1=2000, t1=0.5, method='logarithmic')

    # 2. 模拟 Mic2 数据 (只有行波)
    # 理论延时 time_delay = s / c
    s = 0.10 - 0.05  # 间距 0.05m
    c = 343.0
    delay_samples = int((s / c) * fs)

    # data2 就是 data1 向右平移 (在数组左边补0)
    data2 = np.pad(data1, (delay_samples, 0), 'constant')[:len(data1)]
    # 注意：行波传播会有幅度衰减 (1/r)，但短距离平面波可忽略，或者设为 1.0

    # 3. 运行分析
    analyzer = ImpedanceAnalyzer(data1, data2, mic1_pos=0.10, mic2_pos=0.05, fs=fs)
    analyzer.plot_results()