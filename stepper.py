import time
import serial
class StepperController:
    def __init__(self, port: str = None, baudrate: int = 9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(2)  # 等待设备初始化

    def move(self, steps: int):
        cmd = f"{steps}\n".encode('utf-8')
        self.ser.write(cmd) 
        self.ser.flush()

    def close(self):
        self.ser.close()

if __name__ == "__main__":
    stepper = StepperController(port="COM3")  # 替换为实际端口
    try:
        stepper.move(10)  # 向前移动10步
        time.sleep(1)
        stepper.move(-10) # 向后移动10步
    finally:
        stepper.close()