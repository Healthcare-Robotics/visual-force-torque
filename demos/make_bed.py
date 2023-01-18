import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *

roll = 0
pitch = -45
yaw = 0
HOME_POS_DICT = {'y': 0.4, 'z': 0.95, 'roll': roll*np.pi/180, 'pitch': pitch*np.pi/180, 'yaw': yaw*np.pi/180, 'gripper': 100}

class MakeBed(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)

        self.blanket_force_thresh = 2.5 # force from blanket tension
        self.min_bed_force = 3 # minimum reliable force indicating contact

        self.delta_x = 0.05
        # self.delta_y = 0.03
        self.y_pull_pos = 0.0
        self.delta_z = 0.01

        self.delta_z_blanket = 0.015 # vertical movement after detecting bed
        self.max_z = HOME_POS_DICT['z']
        self.min_z_bed = 0.65 # the robot shouldn't go below the bed
        self.release_z = 0.75 # height to release the blanket

        self.gripper_hold_pos = -50
        self.gripper_open_pos = HOME_POS_DICT['gripper']

        self.state = 'start'
    
    def control_robot(self, pred_dict, pos_dict):
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_mag = np.linalg.norm(force)

        if self.state == 'start':
            self.server.send_payload(HOME_POS_DICT)
            self.sleep_and_record(5)
            self.state = 'find_blanket'

          # making the bed
        elif self.state == 'find_blanket':
            if abs(force[2]) < self.min_bed_force:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            else:
                # move up a little, then grab and lift the blanket
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z_blanket, 'pitch': HOME_POS_DICT['pitch']})
                self.server.send_payload({'gripper':self.gripper_hold_pos})
                self.server.send_payload({'z':self.max_z})
                self.sleep_and_record(3)
                self.state = 'make_bed'

        elif self.state == 'make_bed':
            if abs(force[0]) < self.blanket_force_thresh:
                # move base until robot x force gets too large
                self.server.send_payload({'x': -self.delta_x})
                self.server.send_payload({'y': self.y_pull_pos})
            elif pos_dict['y'] < 0.1:
                # move down and release if the force is too big and the arm is retracted
                self.server.send_payload({'x': 0.15})
                self.sleep_and_record(1)
                self.server.send_payload({'z':self.release_z})
                self.sleep_and_record(1)
                self.server.send_payload({'gripper':self.gripper_open_pos})
                self.sleep_and_record(1)
                self.stop = True

if __name__ == "__main__":
    mb = MakeBed()
    mb.run_demo(control_func=mb.control_robot)