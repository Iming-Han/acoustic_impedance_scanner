from algorithm import ImpedanceAnalyzer
from audio import AudioEngine, SignalGenerator
from stepper import StepperController
import serial.tools.list_ports
import time
def main():
    # basic parameters
    TEMPERATURE = 25.0  # celsuis
    SAMPLE_RATE = 48000  # hz
    EXCITATION_DURATION = 1.0  # excitation signal duration (s)
    AMPLITUDE_DB = -6  # excitation signal amplitude (dBFS)
    F_START, F_END = (20, 20000)  # excitation signal frequency range (Hz)
    
    MIC1_POS = -2
    MIC2_POS = -2.55
    MIC3_POS = -2.95
    # recording setting
    #  1.信号生成
    signal_gen = SignalGenerator(fs=SAMPLE_RATE, amplitude_db=AMPLITUDE_DB)
    signal = signal_gen.generate_sweep(duration=EXCITATION_DURATION, f_start=F_START, f_end=F_END)
    #  2.音频设备选择
    audio_engine = AudioEngine(fs=SAMPLE_RATE)
    audio_engine.list_devices()
    input_id = 4   
    output_id = 4  
    audio_engine.select_device(input_id=input_id, output_id=output_id)
    # 初始化步进电机控制器，列出可用串口并设置
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for port in ports:
        print(f" - {port.device}")
    PORTID = "COM3"  # 替换为实际端口
    stepper = StepperController(port=PORTID)  # 替换为实际端口
    # 开始测量
    print("Starting measurement...")
    # 定义每次移动的距离（这里填入每次的不同距离）
    move_distances = [10, 10, 10]
    recorded_data = []

    for i, distance in enumerate(move_distances, 1):
        print(f"point {i}: moving {distance}, begining to measure...")
        stepper.move(distance)
        time.sleep(5)  # waiting for moving
        
        data = audio_engine.play_record(signal)
        recorded_data.append(data)
        
        time.sleep(EXCITATION_DURATION + 1)
        
    data1, data2, data3 = recorded_data
    ## Analyze the measurement data
    # initialize impedance analyzer
    analyzer_12 = ImpedanceAnalyzer(mic1_pos = MIC1_POS, mic2_pos = MIC2_POS, mic1_time_data = data1, mic2_time_data = data2, fs = SAMPLE_RATE, temp_c = TEMPERATURE)
    analyzer_13 = ImpedanceAnalyzer(mic1_pos = MIC1_POS, mic2_pos = MIC3_POS, mic1_time_data = data1, mic2_time_data = data3, fs = SAMPLE_RATE, temp_c = TEMPERATURE)
    # calculate impedance
    freqs = analyzer_12.freqs
    impedance_12 = analyzer_12.calculate_impedance()
    impedance_13 = analyzer_13.calculate_impedance()
    # calculate absorption coefficient
    alpha_12 = analyzer_12.alpha    
    alpha_13 = analyzer_13.alpha
    # plot results
    analyzer_12.plot_results()
    analyzer_13.plot_results()
if __name__ == "__main__":
    main() 
    

