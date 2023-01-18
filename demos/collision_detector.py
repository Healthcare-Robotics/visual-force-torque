import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *

HOME_POS_DICT = {'y': 0.1, 'z': 0.75, 'roll': 0, 'pitch': -np.pi/4, 'yaw': 0, 'gripper': -20}

class CollisionDetector(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        self.force_limit = 1.5
        self.state = 'go'

    def control_robot(self, pred_dict):
        robot_ok, pos_dict = read_robot_status(self.client)
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_mag = np.linalg.norm(force)
        print('force unit vector:', force / force_mag)
        if force_mag > self.force_limit:
            self.collision_flag = True
            self.state = 'stop'
            self.server.send_payload({'beep':True})
        else:
            self.collision_flag = False
            self.state = 'go'       

if __name__ == "__main__":
    cd = CollisionDetector()
    cd.run_demo(control_func=cd.control_robot)