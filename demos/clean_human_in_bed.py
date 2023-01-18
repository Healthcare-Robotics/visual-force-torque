import numpy as np
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *
import time

roll = 0
pitch = -45
yaw = 0
HOME_POS_DICT = {'y': 0.135, 'z': 0.9, 'roll': roll*np.pi/180, 'pitch': pitch*np.pi/180, 'yaw': yaw*np.pi/180, 'gripper': -20}

class CleanHuman(LiveModel):
    def __init__(self, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

        self.force_limit = 25 # don't break the robot
        self.blanket_force_thresh = 3.5 # force from blanket tension
        self.min_bed_force = 5 # minimum reliable force indicating contact
        self.foot_force_thresh = 2 # force from foot of bed
        self.detect_human_force = 3 # force to detect human contact

        self.cleaning_force_thresh = 8.5 # ideal cleaning force
        self.min_cleaning_force = 7.5 # minimum reliable force indicating contact for cleaning
        self.min_force_timer = np.inf
        self.mag_delta_tan = 0.02 # movement along surface when cleaning
        self.mag_delta_norm = 0.001 # movement to regulate force normal to surface
        self.x_dir = np.array([1, 0, 0])
        self.y_dir = np.array([0, 1, 0])
        self.z_dir = np.array([0, 0, 1])
        self.arm_dir = 'in'
        self.recover_count = 0
        self.last_large_force = np.array([0, 0, 0]) # last recorded force before going below min force

        self.delta_x = 0.05
        self.x_odom = 0
        
        self.delta_y = 0.03

        self.delta_z = 0.01
        self.delta_z_blanket = 0.015 # vertical movement after detecting bed
        self.max_z = HOME_POS_DICT['z']
        self.min_z_bed = 0.65 # the robot shouldn't go below the bed
        self.release_z = 0.75 # height to release the blanket
        self.detect_foot_z = 0.77
        self.detect_human_z = 0.725

        self.gripper_hold_pos = -50
        self.gripper_open_pos = HOME_POS_DICT['gripper']

        self.state = 'start'

    def find_tangent_deltas(self, force, dir_des):
        # projecting desired direction onto tangent plane specified by force normal
        mag_force = np.linalg.norm(force)
        mag_force_yz = np.linalg.norm(force[1:3])
        force_norm = force / mag_force
        tangent_delta = dir_des - np.dot(dir_des, force_norm) * force_norm # simple because force normal magnitude = 1
        tangent_delta = tangent_delta / np.linalg.norm(tangent_delta)
        tangent_delta = tangent_delta * self.mag_delta_tan # magnitude of tangent delta = mag_delta_tan
        return tangent_delta
        
    def regulate_force(self, force):
        robot_ok, pos_dict = read_robot_status(self.client)
        mag_force = np.linalg.norm(force)
        mag_force_yz = np.linalg.norm(force[1:3])

        if mag_force_yz >= self.min_cleaning_force:
            force_norm = force / np.linalg.norm(force)
            force_norm_yz = force[1:3] / np.linalg.norm(force[1:3])
            norm_delta = force_norm * self.mag_delta_norm # magnitude of norm delta = mag_delta_norm

        else:
            print('using last large force')
            force_norm = self.last_large_force / np.linalg.norm(self.last_large_force)
            norm_delta = force_norm * self.mag_delta_norm

        # move away from force if it's too big
        if mag_force_yz > self.cleaning_force_thresh:
            print('moving away from force')
            self.server.send_payload({'y': pos_dict['y'] + norm_delta[1] * 2, 'z': pos_dict['z'] + norm_delta[2] * 2})
            self.sleep_and_record(0.5)

        # move toward force if it's too small
        elif mag_force_yz < self.cleaning_force_thresh:
            norm_delta = -norm_delta
            print('moving toward force')
            self.server.send_payload({'y': pos_dict['y'] + norm_delta[1], 'z': pos_dict['z'] + norm_delta[2]})
            self.sleep_and_record(0.5)
        return norm_delta

    def control_robot(self, pred_dict):
        robot_ok, pos_dict = read_robot_status(self.client)
        force = np.array([pred_dict['Fx'], pred_dict['Fy'], pred_dict['Fz']])
        force_mag = np.linalg.norm(force)
        
        if force_mag > self.min_cleaning_force:
            self.last_large_force = force

        if self.state != 'clean_human':
            self.min_force_timer = np.inf

        if force_mag > self.force_limit:
            self.state = 'recover'
            print('recovering due to large force. don\'t hurt the robot')

        if self.state == 'start':
            self.server.send_payload(HOME_POS_DICT)
            self.sleep_and_record(5)
            self.state = 'find_blanket_top'

        # regulating vertical gripper position with a setpoint of self.blanket_force_thresh
        elif self.state == 'find_blanket_top':
            if abs(force[2]) < self.min_bed_force:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            else:
                # move up a little, then grab and lift the blanket
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z_blanket, 'pitch': HOME_POS_DICT['pitch']})
                self.sleep_and_record(2)
                self.server.send_payload({'gripper':self.gripper_hold_pos})
                self.server.send_payload({'z':self.max_z})
                self.sleep_and_record(5)
                self.state = 'unmake_bed'

        # making the bed
        elif self.state == 'unmake_bed':
            if abs(force[0]) < self.blanket_force_thresh:
                # move base until robot x force gets too large
                self.server.send_payload({'x': self.delta_x})
                self.sleep_and_record(0.5)

            else:
                # move down and release if the force is too big
                self.server.send_payload({'x': 0.05})
                self.sleep_and_record(1)
                self.server.send_payload({'z':self.release_z})
                self.sleep_and_record(1)
                self.server.send_payload({'gripper':self.gripper_open_pos})
                robot_ok, pos_dict = read_robot_status(self.client)
                self.blanket_x = pos_dict['x'] # remembering where the blanket was placed
                self.sleep_and_record(2)
                self.state = 'find_bed_foot'

        elif self.state == 'find_bed_foot':
            if abs(force[0]) < self.foot_force_thresh:
                # move base until robot x force gets too large
                self.server.send_payload({'x': self.delta_x / 2, 'z': self.detect_foot_z})
                self.sleep_and_record(0.5)

            else:
                # pick up the cloth once foot of bed is detected
                self.server.send_payload({'x': -0.25})
                self.sleep_and_record(2)
                self.state = 'find_cloth'

        elif self.state == 'find_cloth':
            if abs(force[2]) < self.min_bed_force:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            else:
                # move up a little, then grab and lift the cloth
                self.server.send_payload({'pitch': HOME_POS_DICT['pitch']})
                self.sleep_and_record(2)
                self.server.send_payload({'gripper':self.gripper_hold_pos})
                self.server.send_payload({'z':self.detect_human_z})
                self.sleep_and_record(2)
                robot_ok, pos_dict = read_robot_status(self.client)
                self.sleep_and_record(1)
                self.state = 'find_human'
        
        elif self.state == 'find_human':
            if abs(force[0]) < self.detect_human_force:
                self.server.send_payload({'x': -self.delta_x / 2, 'y':HOME_POS_DICT['y'] + 0.01})
                self.sleep_and_record(0.5)
            elif abs(force[0]) >= self.detect_human_force:
                robot_ok, pos_dict = read_robot_status(self.client)
                self.state = 'move_to_human'

        elif self.state == 'move_to_human':
            self.server.send_payload({'z': self.max_z})
            self.sleep_and_record(1)
            self.server.send_payload({'x': -0.2})
            self.sleep_and_record(1)
            self.state = 'find_surface'

        elif self.state == 'find_surface':
            if pos_dict['z'] < self.min_z_bed:
                self.state = 'recover'
            if force_mag < self.cleaning_force_thresh:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            elif force_mag > self.cleaning_force_thresh:
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
                self.state = 'clean_human'

        elif self.state == 'clean_human':
            if pos_dict['z'] < self.min_z_bed:
                self.state = 'recover'
                print('recovering due to min z')
            norm_delta = self.regulate_force(force)
            if force_mag < self.min_cleaning_force:
                if self.min_force_timer == np.inf:
                    self.min_force_timer = time.time()

                # if not sustaining min force, recover
                if time.time() - self.min_force_timer > 3:
                    self.min_force_timer = np.inf
                    self.state = 'recover'
                    print('recovering due to min force')
            else:
                self.min_force_timer = np.inf
            
            # cleaning by moving perpendicular to force normal while regulating normal force
            if self.arm_dir == 'out':
                if pos_dict['y'] > 0.5:
                    self.state = 'recover'
                    print('recovering due to joint limits')   
                else:
                    delta = self.find_tangent_deltas(force, self.y_dir)
                    self.server.send_payload({'y':pos_dict['y'] + delta[1]})
                    self.sleep_and_record(0.1)
                    self.server.send_payload({'z':pos_dict['z'] + delta[2], 'pitch': HOME_POS_DICT['pitch']})

            elif self.arm_dir == 'in':
                if pos_dict['y'] < 0.01:
                    self.state = 'recover'   
                    print('recovering due to joint limits')   

                else:
                    delta = self.find_tangent_deltas(force, -self.y_dir)
                    self.server.send_payload({'y':pos_dict['y'] + delta[1]})
                    self.sleep_and_record(0.1)
                    self.server.send_payload({'z':pos_dict['z'] + delta[2], 'pitch': HOME_POS_DICT['pitch']})
            
        elif self.state == 'find_bed_foot_again':
            if abs(force[0]) < self.foot_force_thresh:
                # move base until robot x force gets too large
                self.server.send_payload({'x': self.delta_x / 2, 'z': self.detect_foot_z})
                self.sleep_and_record(0.5)

            else:
                # drop the cloth once foot of bed is detected
                self.server.send_payload({'x': -0.25})
                self.sleep_and_record(2)
                self.server.send_payload({'gripper':self.gripper_open_pos})
                self.sleep_and_record(2)
                robot_ok, pos_dict = read_robot_status(self.client)
                # move to blanket x, find the surface, and pick it up
                self.server.send_payload({'x': self.blanket_x - pos_dict['x'] - 0.05})
                self.sleep_and_record(3)
                self.state = 'find_blanket_bottom'

        elif self.state == 'find_blanket_bottom':
            if abs(force[2]) < self.min_bed_force:
                self.server.send_payload({'z':pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
            else:
                # move up a little, then grab and lift the blanket
                self.server.send_payload({'z':pos_dict['z'] + self.delta_z_blanket, 'pitch': HOME_POS_DICT['pitch']})
                self.server.send_payload({'gripper':self.gripper_hold_pos})
                self.server.send_payload({'z':self.max_z})
                self.sleep_and_record(2)
                self.state = 'make_bed'

        elif self.state == 'make_bed':
            if abs(force[0]) < self.blanket_force_thresh:
                # move base until robot x force gets too large
                self.server.send_payload({'x': -self.delta_x})
            else:
                # move down and release if the force is too big
                self.server.send_payload({'x': 0.05})
                self.server.send_payload({'z':self.release_z})
                self.sleep_and_record(1)
                self.server.send_payload({'gripper':self.gripper_open_pos})
                self.sleep_and_record(1)
                self.stop = True

        elif self.state == 'move_base':
            self.server.send_payload({'x': -self.delta_x})
            self.state = 'find_surface'

        elif self.state == 'recover':
            self.recover_count += 1
            self.server.send_payload({'z':0.8})
            self.sleep_and_record(1)
            self.server.send_payload({'y':HOME_POS_DICT['y'] + 0.01})
            self.sleep_and_record(2)
                
            if self.arm_dir == 'out':
                self.arm_dir = 'in'

            elif self.arm_dir == 'in':
                self.arm_dir = 'out'

            if self.recover_count >= 4:
                self.state = 'find_bed_foot_again'
            elif self.recover_count % 2 == 0:
                self.state = 'move_base'
            else:
                self.state = 'find_surface'
            
if __name__ == "__main__":
    ch = CleanHuman()
    ch.run_demo(control_func=ch.control_robot)