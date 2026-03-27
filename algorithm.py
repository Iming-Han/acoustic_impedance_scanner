import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from functools import cached_property
import warnings
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
        if mic1_time_data.ndim != 1 or mic2_time_data.ndim != 1:
            warnings.warn(f"Input time data is {mic1_time_data.ndim}D. The data will be flattened automatically.")
            self.mic1_time_data = mic1_time_data.ravel()
            self.mic2_time_data = mic2_time_data.ravel()
        else:
            self.mic1_time_data = mic1_time_data
            self.mic2_time_data = mic2_time_data
        
        self.mic1_pos = mic1_pos
        self.mic2_pos = mic2_pos
        self.s = mic1_pos - mic2_pos
        # init public attributes
        self.freqs = np.fft.rfftfreq(len(mic1_time_data), 1/fs)
        self.k = 2 * np.pi * self.freqs / self.c  # wavenumber
        print(f"Initialized ImpedanceAnalyzer with mic1_pos={mic1_pos}m, mic2_pos={mic2_pos}m, temp={temp_c}°C, fs={fs}Hz")

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
        reflection_factor = (self.H12 - np.exp(-1j * self.k * self.s)) / (-self.H12 + np.exp(1j * self.k * self.s))*np.exp(2j * self.k * self.mic1_pos)
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
    # --- 1. 设置系统参数 ---
    fs = 48000
    duration = 0.5
    N = int(duration * fs)
    mic1_pos = 0.10  # 麦克风1距离样品表面 0.10m
    mic2_pos = 0.05  # 麦克风2距离样品表面 0.05m
    temp_c = 20
    
    # 获取精确的物理常数以匹配类内部的计算
    c = ImpedanceAnalyzer.AIR_SPEED_OF_SOUND_0C * np.sqrt(1 + temp_c / 273.15)
    rho = ImpedanceAnalyzer.AIR_DENSITY

    # --- 2. 设定目标声学特性 (正向设计的基准) ---
    # 设定一个理论声阻抗，例如：空气特征阻抗的 1 倍
    Z_target = 3.0 * rho * c 
    
    # 根据阻抗计算理论反射系数 R
    # R = (Z - rho*c) / (Z + rho*c)
    R_target = (Z_target - rho * c) / (Z_target + rho * c) 
    
    # 根据反射系数计算理论吸声系数 α
    alpha_target = 1 - np.abs(R_target)**2 

    print("=== 设定的理论目标值 ===")
    print(f"目标声阻抗 Re(Z): {Z_target:.2f} Pa·s/m")
    print(f"目标反射系数 R:   {R_target:.4f}")
    print(f"目标吸声系数 α:   {alpha_target:.4f}")
    print("========================\n")

    # --- 3. 生成基础入射波信号 (频域) ---
    t = np.linspace(0, duration, N, endpoint=False)
    # 使用 Chirp 信号覆盖宽频带 (50Hz - 5000Hz)
    src_time = signal.chirp(t, f0=50, f1=5000, t1=duration, method='linear')
    # 加一个 Tukey 窗减小截断效应引起的频谱泄露
    window = signal.windows.tukey(N, alpha=0.1)
    src_time = src_time * window

    # 转换到频域，代表在 x=0 (样品表面) 处的入射波频谱 P_I
    P_I = np.fft.rfft(src_time)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    
    # 避免 0Hz 除零警告，轻微偏移
    freqs[0] = 1e-12 
    k = 2 * np.pi * freqs / c

    # --- 4. 生成 Mic1 和 Mic2 的物理观测信号 ---
    # 反射波频谱 P_R (在 x=0 处)
    P_R = R_target * P_I

    # 在位置 x 处的总声压: P(x) = 入射波分量 + 反射波分量
    # P(x) = P_I * e^(j*k*x) + P_R * e^(-j*k*x)
    # 依据物理约定：x 轴正向为远离样品方向，入射波向 -x 传播 (相位超前)，反射波向 +x 传播 (相位滞后)
    X1 = P_I * np.exp(1j * k * mic1_pos) + P_R * np.exp(-1j * k * mic1_pos)
    X2 = P_I * np.exp(1j * k * mic2_pos) + P_R * np.exp(-1j * k * mic2_pos)

    # 逆傅里叶变换回到时域，得到真实的麦克风采集数据
    mic1_time_data = np.fft.irfft(X1, n=N)
    mic2_time_data = np.fft.irfft(X2, n=N)

    # --- 5. 代入算法进行计算 (反向验证) ---
    analyzer = ImpedanceAnalyzer(
        mic1_time_data=mic1_time_data,
        mic2_time_data=mic2_time_data,
        mic1_pos=mic1_pos,
        mic2_pos=mic2_pos,
        temp_c=temp_c,
        fs=fs
    )

    # --- 6. 绘图对比验证 ---
    # 筛选有效频率范围 (忽略 Chirp 没有能量的极低频和高频)
    valid_idx = (analyzer.freqs >= 100) & (analyzer.freqs <= 4000)
    plot_freqs = analyzer.freqs[valid_idx]
    
    # 提取算法算出的结果
    calc_alpha = analyzer.alpha[valid_idx]
    calc_z_real = np.real(analyzer.impedance[valid_idx])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 子图1：吸声系数对比
    axs[0].plot(plot_freqs, calc_alpha, label='Calculated $\\alpha$', color='#1f77b4', linewidth=2)
    axs[0].axhline(y=alpha_target, color='#d62728', linestyle='--', label='Target $\\alpha$', linewidth=2)
    axs[0].set_title('Absorption Coefficient: Algorithm vs Target')
    axs[0].set_ylabel('Absorption Coefficient $\\alpha$')
    axs[0].grid(True, alpha=0.5)
    axs[0].legend()
    # 限制 y 轴范围以便观察误差
    axs[0].set_ylim(alpha_target - 0.1, alpha_target + 0.1) 

    # 子图2：声阻抗(实部)对比
    axs[1].plot(plot_freqs, calc_z_real, label='Calculated Re(Z)', color='#2ca02c', linewidth=2)
    axs[1].axhline(y=Z_target, color='#d62728', linestyle='--', label='Target Re(Z)', linewidth=2)
    axs[1].set_title('Acoustic Impedance (Real Part): Algorithm vs Target')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Impedance (Pa·s/m)')
    axs[1].grid(True, alpha=0.5)
    axs[1].legend()
    # 限制 y 轴范围以便观察误差
    axs[1].set_ylim(Z_target * 0.9, Z_target * 1.1)

    plt.tight_layout()
    plt.show()