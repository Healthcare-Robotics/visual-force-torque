import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *

HOME_POS_DICT = {'y': 0.1, 'z': 0.75, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 100}

class ObjectHandover(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        self.delta_force_thresh = 2
        self.state = 'hold on'
        self.gripper_hold_pos = -10
        self.gripper_open_pos = HOME_POS_DICT['gripper']
        self.thresh_count_lim = 5 # number of consecutive frames of force diff above thresh needed to release gripper

        self.server.send_payload(HOME_POS_DICT)
        self.sleep_and_record(2)
        self.server.send_payload({'gripper':self.gripper_hold_pos})
        self.sleep_and_record(2)
        
        self.reset_force()

    def reset_force(self):
        # finding initial force
        robot_ok, pos_dict = read_robot_status(self.client)
        self.thresh_count = 0
        self.frame = self.feed.get_frame()

        # input to model is current gripper position if live, frame from folder if not live, and 0 if held out
        robot_state = get_robot_state(self.config, pos_dict)
        output = predict(self.model, self.frame, robot_state)
        self.prev_force = output[0:3]

    def control_robot(self, pred_dict):
        robot_ok, pos_dict = read_robot_status(self.client)
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_diff = force - self.prev_force
        force_diff_mag = np.linalg.norm(force_diff)
        print('force: ', force)
        print('prev_force:', self.prev_force)
        print('force_diff_mag: ', force_diff_mag)
        if self.state == 'hold on':
            if force_diff_mag > self.delta_force_thresh:
                self.thresh_count += 1
            if self.thresh_count > self.thresh_count_lim:
                self.state = 'let go'
                self.server.send_payload({'gripper':self.gripper_open_pos})
        else:
            self.server.send_payload({'gripper':self.gripper_open_pos})

        self.keyboard_teleop()

        if self.demo_flag:
            self.state = 'hold on'
            self.server.send_payload({'gripper':self.gripper_hold_pos})
            self.sleep_and_record(2)
            self.reset_force()
            self.demo_flag = False
 
if __name__ == "__main__":
    oh = ObjectHandover()
    oh.run_demo(control_func=oh.control_robot)