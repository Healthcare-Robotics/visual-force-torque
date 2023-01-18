import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *
import time

HOME_POS_DICT = {'y': 0.1, 'z': 0.75, 'roll': 0, 'pitch': -np.pi/4, 'yaw': 0, 'gripper': -20}

class CleanFlatSurface(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        self.force_thresh = 6
        self.force_limit = 25
        self.delta_x = 0.03
        self.delta_y = 0.05
        self.delta_z = 0.0075
        self.recover_delta = 0.03
        self.state = 'start'
        self.arm_dir = 'out'
        self.base_dir = 'forward'
        self.recover_count = 0

    def control_robot(self, pred_dict):
        robot_ok, pos_dict = read_robot_status(self.client)
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_mag = np.linalg.norm(force)
        
        print('force unit vector:', force / force_mag)
        if force_mag > self.force_limit:
            self.state = 'recover'

        if self.state == 'start':
            self.server.send_payload(HOME_POS_DICT)
            time.sleep(5)
            self.state = 'find_surface'

        # regulating vertical gripper position with a setpoint of self.force_thresh
        elif self.state == 'find_surface':
            if pred_dict['Fz'] < self.force_thresh:
            # if force_mag < self.force_thresh:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': -np.pi/4})
            elif pred_dict['Fz'] > self.force_thresh:
            # elif force_mag > self.force_thresh:
                self.surface_z = pos_dict['z']
                self.state = 'clean'

        # cleaning the surface
        elif self.state == 'clean':
            # recover if too low
            if pos_dict['z'] < self.surface_z - self.recover_delta:
                self.state = 'recover'
            # extending arm
            elif self.arm_dir == 'out':
                if pos_dict['y'] > 0.5:
                    # self.arm_dir = 'in'
                    self.state = 'recover'   
                else:
                    self.server.send_payload({'y':pos_dict['y'] + self.delta_y})
            elif self.arm_dir == 'in':
                if pos_dict['y'] < 0.01:
                    # self.arm_dir = 'out'
                    self.state = 'recover'   
                else:
                    self.server.send_payload({'y':pos_dict['y'] - self.delta_y})

            # regulating vertical gripper position with a setpoint of self.delta
            if pred_dict['Fz'] < self.force_thresh:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': -np.pi/4})
                
            elif pred_dict['Fz'] > self.force_thresh:
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z, 'pitch': -np.pi/4})

        elif self.state == 'move_base':
            if self.base_dir == 'back':
                self.server.send_payload({'x': -self.delta_x})
            elif self.base_dir == 'forward':
                self.server.send_payload({'x': self.delta_x})
            self.state = 'find_surface'

        # getting back onto the surface if below known surface height
        elif self.state == 'recover':
            self.recover_count += 1
            self.server.send_payload({'z':self.surface_z + self.recover_delta})
            time.sleep(1)
            if self.arm_dir == 'out':
                self.arm_dir = 'in'
                self.server.send_payload({'y':pos_dict['y'] - self.delta_y * 2})
                time.sleep(1)

            elif self.arm_dir == 'in':
                self.arm_dir = 'out'
                self.server.send_payload({'y':pos_dict['y'] + self.delta_y * 2})
                time.sleep(1)

            if self.recover_count >= 2:
                self.state = 'move_base'
                self.recover_count = 0
            else:
                self.state = 'find_surface'
        
if __name__ == "__main__":
    cfs = CleanFlatSurface()
    cfs.run_demo(control_func=cfs.control_robot)