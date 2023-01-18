import serial
import ast
import numpy as np
 
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.live_model import LiveModel
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.transforms import *
import time
import json

HOME_POS_DICT = {'y': 0.1, 'z': 0.75, 'roll': -np.pi/4, 'pitch': -np.pi/4, 'yaw': 0, 'gripper': 0}

class LoadCellCapture(LiveModel):
    def __init__(self, setpoint, ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER):
        super().__init__()
        robot_ok, pos_dict = read_robot_status(self.client)
        self.arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=5.0)
        self.server.send_payload(HOME_POS_DICT)
        time.sleep(1)

        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        # self.delta_z = 0.00025
        self.setpoint = setpoint
        self.eps = 0.01
        self.load_cell_data = 0

        self.frame = self.feed.get_frame()
        ft_data = self.ft.get_ft()

        # input to model is current gripper position if live, frame from folder if not live, and 0 if held out
        robot_state = get_robot_state(self.config, pos_dict)

        self.output_offset = predict(self.model, self.frame, robot_state)
        self.gt_offset = ft_data

        save_dir = './paper/fig_data/load_cell/'
        folder_index = len([f for f in os.listdir(save_dir)])
        self.file_name = save_dir + 'load_cell_data_' + str(folder_index) + '.txt'

    def get_load_cell_data(self):
        data = self.arduino.read_until()
        # print('raw data: ', data)
        
        if len(data) > 4: # if not empty
            data = str(data)[2:-5]   
            data = float(data)
            # converting grams to newtons
            data *= 0.00980665
            data = np.round(data, 3)
            print('load cell data: ', data)

            return data
  
    def control_robot(self, pred, gt):
        robot_ok, pos_dict = read_robot_status(self.client)
        force_pred = pred[0:3]
        force_gt = gt[0:3]
        data = self.get_load_cell_data()

        if data is not None:
            self.load_cell_data = data

        err = self.load_cell_data - self.setpoint
        print('err: ', err)

        if abs(err) > 0.5:
            self.delta_z = 0.001
        else:
            self.delta_z = 0.0001

        if err < -self.eps:
            self.server.send_payload({'z': pos_dict['z'] - self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
        elif err > self.eps:
            self.server.send_payload({'z': pos_dict['z'] + self.delta_z, 'pitch': HOME_POS_DICT['pitch']})
        else:
            print('prediction: ', pred)
            print('ground truth: ', gt)
            print('predicted force mag: ', np.linalg.norm(force_pred))
            print('ground truth force mag: ', np.linalg.norm(force_gt))
            print('load cell data: ', self.load_cell_data)

            with open('./paper/load_cell_data.txt', 'a') as f:
                f.write('\nFT prediction: ')
                f.write(str(pred))
                f.write('\nFT ground truth: ')
                f.write(str(gt))
                f.write('\npredicted force mag (N): ')
                f.write(str(np.linalg.norm(force_pred)))
                f.write('\nground truth force mag (N): ')
                f.write(str(np.linalg.norm(force_gt)))
                f.write('\nload cell data (N): ')
                f.write(str(self.load_cell_data))
                f.write('\n')
            self.stop = True

            load_cell_dict = {'prediction': pred.tolist(), 'ground_truth': gt.tolist(), 'pred_force_mag': float(np.linalg.norm(force_pred)), 'gt_force_mag': float(np.linalg.norm(force_gt)), 'load_cell_data': self.load_cell_data, 'setpoint': self.setpoint} 
            with open(self.file_name, 'w') as file:
                file.write(json.dumps(load_cell_dict))
            
    def run(self):
        while True:
            robot_ok, pos_dict = read_robot_status(self.client)
            # ft frame to robot frame
            if robot_ok:
                frame_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
                theta = pos_dict['pitch'] # gripper pitch angle
                psi = pos_dict['roll'] # gripper roll angle
                gripper_pitch_rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
                # rotating around z axis by psi
                gripper_roll_rotation = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
                total_rotation =  frame_rotation @ gripper_pitch_rotation @ gripper_roll_rotation

                self.frame = self.feed.get_frame()
                ft_data = self.ft.get_ft()

                # input to model is current gripper position if live, frame from folder if not live, and 0 if held out
                robot_state = get_robot_state(self.config, pos_dict)

                output = predict(self.model, self.frame, robot_state)
                output = output - self.output_offset # accounting for angle of gripper
                ft_data = ft_data - self.gt_offset # accounting for angle of gripper

                output_robot_frame = output.copy()
                output_robot_frame[0:3] = output[0:3] @ total_rotation
                output_robot_frame[3:6] = output[3:6] @ total_rotation

                ft_data_robot_frame = ft_data.copy()
                ft_data_robot_frame[0:3] = ft_data[0:3] @ total_rotation
                ft_data_robot_frame[3:6] = ft_data[3:6] @ total_rotation

                pred_dict = dict(zip(self.keys, output.tolist()))
                gt_dict = dict(zip(self.keys, ft_data.tolist()))

                # pred_dict_robot_frame = dict(zip(self.keys, output_robot_frame.tolist()))
                # gt_dict_robot_frame = dict(zip(self.keys, ft_data_robot_frame.tolist()))

                print('\n')                
                self.control_robot(pred=output, gt=ft_data)

                force_gt = ft_data[0:3]
                torque_gt = ft_data[3:6]
                force_pred = output[0:3]
                torque_pred = output[3:6]

                error = output - ft_data
                self.pred_hist = np.concatenate(([output], self.pred_hist), axis=0)
                self.gt_hist = np.concatenate(([ft_data], self.gt_hist), axis=0)

                if self.args.view:
                    self.fig = self.plotter.visualize_ft(force_gt, torque_gt, force_pred, torque_pred, self.frame, self.collision_flag)
                    cv2.imshow('figure', self.fig)
                    self.keyboard_teleop()

                if self.args.record_video:
                    self.result.write(self.fig)

                if self.stop:
                    break
        
        self.server.send_payload(HOME_POS_DICT)
        time.sleep(3)

if __name__ == "__main__":
    config, args = parse_config_args()
    lcc = LoadCellCapture(setpoint=4)
    lcc.run()
