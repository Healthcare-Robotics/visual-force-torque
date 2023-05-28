import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *
import time

roll = 0
pitch = -30
yaw = 0
# HOME_POS_DICT = {'y': 0.19, 'z': 0.75, 'roll': roll*np.pi/180, 'pitch': pitch*np.pi/180, 'yaw': yaw*np.pi/180, 'gripper': -20} # arm
HOME_POS_DICT = {'y': 0.35, 'z': 0.8, 'roll': roll*np.pi/180, 'pitch': pitch*np.pi/180, 'yaw': yaw*np.pi/180, 'gripper': -20} # leg

class CleanManikin(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        self.force_thresh = 5.0 # 6 # ideal cleaning force
        self.min_force = 4.75 # 5.5 # minimum reliable force indicating contact
        self.min_force_timer = np.inf
        self.force_limit = 25 # don't break the robot
        self.mag_delta_tan = 0.005 # movement along surface when cleaning
        self.mag_delta_norm = 0.002 # movement to regulate force normal to surface
        self.delta_x = 0.02
        self.delta_z = 0.005
        self.x_dir = np.array([1, 0, 0])
        self.y_dir = np.array([0, 1, 0])
        self.z_dir = np.array([0, 0, 1])
        self.min_z = 0.6
        self.state = 'start'
        self.arm_dir = 'in'
        self.base_dir = 'forward'
        self.recover_delta = 0.05
        self.recover_count = 0
        self.last_large_force = np.array([0, 0, 0]) # last recorded force before going below min force

    def find_tangent_deltas(self, force, dir_des):
        # projecting desired direction onto tangent plane specified by force normal
        mag_force = np.linalg.norm(force)
        mag_force_yz = np.linalg.norm(force[1:3])
        force_norm = force / mag_force
        tan_deltas = dir_des - np.dot(dir_des, force_norm) * force_norm # simple because force normal magnitude = 1
        tan_deltas = tan_deltas / np.linalg.norm(tan_deltas)
        tan_deltas = tan_deltas * self.mag_delta_tan # magnitude of tangent delta = mag_delta_tan
        print('tangent_delta', tan_deltas)
        return tan_deltas
        
    def regulate_force(self, force):
        mag_force = np.linalg.norm(force)
        mag_force_yz = np.linalg.norm(force[1:3])

        if mag_force_yz >= self.min_force:
            # move away from force if greater than min
            force_norm = force / np.linalg.norm(force)
            force_norm_yz = force[1:3] / np.linalg.norm(force[1:3])
            norm_deltas = force_norm * self.mag_delta_norm # magnitude of norm delta = mag_delta_norm

        else:
            # move toward last large force
            print('using last large force')
            force_norm = self.last_large_force / np.linalg.norm(self.last_large_force)
            norm_deltas = force_norm * self.mag_delta_norm

        if mag_force_yz > self.force_thresh:
            # move away from force if it's too big
            print('moving away from force')

        elif mag_force_yz < self.force_thresh:
            # move toward force if it's too small
            norm_deltas = -norm_deltas
            print('moving toward force')
        print('norm_delta', norm_deltas)
        return norm_deltas
                
    def control_robot(self, pred_dict, pos_dict):
        # robot_ok, pos_dict = read_robot_status(self.client)
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_mag = np.linalg.norm(force)

        if force_mag > self.min_force:
            self.last_large_force = force

        if self.state != 'clean':
            self.min_force_timer = np.inf

        print('arm dir: ', self.arm_dir)

        if force_mag > self.force_limit:
            self.state = 'recover'
            print('recovering due to large force. don\'t hurt the robot')

        # recover if height is too low
        if pos_dict['z'] < self.min_z:
            self.state = 'recover'
            print('recovering due to min z')

        if self.state == 'start':
            self.server.send_payload(HOME_POS_DICT)
            time.sleep(5)
            robot_ok, pos_dict = read_robot_status(self.client)
            self.state = 'find_surface'

        # regulating vertical gripper position with a setpoint of self.force_thresh
        elif self.state == 'find_surface':
            if force_mag < self.force_thresh:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            elif force_mag > self.force_thresh:
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
                self.state = 'clean'

        # cleaning the surface
        elif self.state == 'clean':
            norm_deltas = self.regulate_force(force)
            if force_mag < self.min_force:
                if self.min_force_timer == np.inf:
                    self.min_force_timer = time.time()

                # if not sustaining min force, recover
                if time.time() - self.min_force_timer > 3:
                    self.min_force_timer = np.inf
                    self.state = 'recover'
                    print('recovering due to min force')
            else:
                self.min_force_timer = np.inf
            
            # cleaning by moving perpendicular to force normal
            if self.arm_dir == 'out':
                if pos_dict['y'] > 0.5:
                    self.state = 'recover'
                    print('recovering due to joint limits')   
                else:
                    tan_deltas = self.find_tangent_deltas(force, self.y_dir)
                    self.server.send_payload({'y':pos_dict['y'] + tan_deltas[1] + norm_deltas[1], 'z':pos_dict['z'] + tan_deltas[2] + norm_deltas[2], 'pitch': HOME_POS_DICT['pitch']})

            elif self.arm_dir == 'in':
                if pos_dict['y'] < 0.01:
                    self.state = 'recover'   
                    print('recovering due to joint limits')   

                else:
                    tan_deltas = self.find_tangent_deltas(force, -self.y_dir)
                    self.server.send_payload({'y':pos_dict['y'] + tan_deltas[1] + norm_deltas[1], 'z':pos_dict['z'] + tan_deltas[2] + norm_deltas[2], 'pitch': HOME_POS_DICT['pitch']})

        elif self.state == 'move_base':
            if self.base_dir == 'back':
                self.server.send_payload({'x': -self.delta_x})
            elif self.base_dir == 'forward':
                self.server.send_payload({'x': self.delta_x})
            self.state = 'find_surface'

        # getting back onto the surface if below known surface height
        elif self.state == 'recover':
            self.recover_count += 1
            self.server.send_payload({'z':HOME_POS_DICT['z']})
            self.sleep_and_record(1)
            self.server.send_payload(HOME_POS_DICT)
            
            if self.arm_dir == 'out':
                self.arm_dir = 'in'

            elif self.arm_dir == 'in':
                self.arm_dir = 'out'

            if self.recover_count >= 2:
                self.state = 'move_base'
                self.recover_count = 0
            else:
                self.state = 'find_surface'

if __name__ == "__main__":
    cm = CleanManikin()
    cm.run_demo(control_func=cm.control_robot)