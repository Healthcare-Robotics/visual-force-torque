import numpy as np
import subprocess
import time

class FTCapture:
    def __init__(self, delay=0):
        self.counts_per_force =  1e6
        self.counts_per_torque = 1e6        
        self.first_frame_time = 0
        self.current_frame_time = 0
        self.frame_count = 0
        self.delay = delay # number of frames to delay before returning ft data to sync with camera in live model
        self.history = []

    def get_ft(self):
        (stat, output) = subprocess.getstatusoutput( "./netft 192.168.1.1" )
        output_list = output.splitlines()
    
        # resetting ft data to zero
        ft_values = 6*[0]

        # parsing output strings from network call
        for i in range(6):
            if len(output_list[i + 1][4:]) > 1:
                if i <= 3:
                    ft_values[i] = int(output_list[i + 1][4:])/self.counts_per_force
                else:
                    ft_values[i] = int(output_list[i + 1][4:])/self.counts_per_torque
            else:
                ft_values[i] = 0

        self.current_frame_time = time.time()
        if self.first_frame_time == 0:
            self.first_frame_time = self.current_frame_time
        self.frame_count += 1

        if self.delay > 0:
            self.history.append(ft_values)
            if len(self.history) > self.delay:
                self.history.pop(0)

            ft = self.history[0]
            ft = np.array(ft, dtype='float32')

            return ft
            
        else:
            ft = np.array(ft_values, dtype='float32')

        return ft

if __name__ == "__main__":
    ft = FTCapture()
    start_time = time.time()

    while True:
        ft_data = ft.get_ft()
        current_time = time.time() - start_time
        print(np.round(ft_data, 4))
        print('Average FPS', ft.frame_count / (time.time() - ft.first_frame_time))
        print(ft.frame_count, ' frames captured')