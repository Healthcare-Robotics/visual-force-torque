import serial
import ast
import numpy as np
 
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1.0)
 
def get_roll_pitch():
    data = arduino.read_until()
     
    if len(data) > 4:
        data = str(data)[2:-3]   
        data = ast.literal_eval(data)
        data = np.array(data)
        
        return data

if __name__ == '__main__':
    while True:
        data = get_roll_pitch()
        print(data)