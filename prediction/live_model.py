import torch
import numpy as np
import cv2
import os
import time
from prediction.model import Model
# from prediction.model_robot_state import Model
# from prediction.model_effort_baseline import Model
# from prediction.model_vit import Model
from recording.gripper_camera import Camera
from recording.ft_stretch_v1 import FTCapture
from prediction.transforms import *
from prediction.config_utils import *
from prediction.data_utils import *
from prediction.pred_utils import *
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.plotter import Plotter
import json
from paper.generate_saliency_map import gen_saliency_map
from paper.gradcam import gen_gradcam
from paper.blur_saliency import gen_blur_saliency
from prediction.data_utils import process_ft_history

class LiveModel():
    def __init__(self):
        self.config, self.args = parse_config_args()
        
        self.t = RPFTTransforms(self.config.TRANSFORM, 'test', self.config.PIXEL_MEAN, self.config.PIXEL_STD)
        self.transform = self.t.transforms

        if not self.args.xbox:
            self.client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
            self.server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)

        self.delta_lin = 0.03
        self.delta_ang = 0.125
        self.enable_moving = True
        self.frame_count = 0

        self.demo_flag = False # for keyboard control in demos
        self.collision_flag = False # for collision detection demo
        self.stop = False

        model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(self.args.config, self.args.index, self.args.epoch))
        self.model = Model(gradcam=self.args.record_gradcam)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        if self.args.live:
            resource = 3 if self.args.on_robot else 0
            self.feed = Camera(resource=resource, view=self.args.view)
            self.frame = self.feed.get_frame()
            self.ft = FTCapture(delay=self.args.delay)
            self.video_fps = 5
            
        elif self.args.folder is not None:
            self.img_list, self.ft_list, self.grip_list = get_data_lists(self.args.folder)
            self.frame = cv2.imread(self.img_list[0][1])
            # self.video_fps = 30
            self.video_fps = 10

        else:
            raise Exception('No data folder provided. Provide a folder or use live mode.')
            
        print('fps:', self.video_fps)        

        if self.args.fullscreen == True:
            cv2.namedWindow('figure', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('figure', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
       
        self.pred_hist = np.zeros((1, 6))
        self.gt_hist = np.zeros((1, 6))
        self.lift_effort_hist = np.zeros((1,))
        self.pitch_effort_hist = np.zeros((1,))
        self.lift_pos_hist = np.zeros((1,))
        self.timestamp_hist = np.zeros((1,))
        hist_folder = './logs/ft_history/'
        video_folder = './data/raw/'
        self.video_folder_index = len([f for f in os.listdir(video_folder) if f.startswith(self.args.video_name)])
        self.folder_index = len([f for f in os.listdir(hist_folder) if f.startswith(self.args.video_name)])
        self.hist_file_name = hist_folder + self.args.video_name + '_' + str(self.folder_index) + '.txt'

        if self.args.view:
            self.plotter = Plotter(self.frame)

            if self.args.record_video:
                self.result = cv2.VideoWriter('./videos/' + self.args.video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.video_fps, self.plotter.fig_size)
            elif self.args.record_saliency:
                self.result = cv2.VideoWriter('./videos/' + self.args.video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.video_fps, (224, 224))
            elif self.args.record_gradcam:
                self.result = cv2.VideoWriter('./videos/' + self.args.video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.video_fps, (224, 224))
            elif self.args.record_blur_saliency:
                self.result = cv2.VideoWriter('./videos/' + self.args.video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.video_fps, (224, 224))

    def get_folder_frame(self):
        # returns the image and ft snapshot at frame index
        # print(os.path.join(self.img_list[self.frame_count][1]))
        img_frame = cv2.imread(os.path.join(self.img_list[self.frame_count][1]))
        ft_frame = np.load(os.path.join(self.ft_list[self.frame_count][1]))
        # ft_frame = ft_frame / self.config.SCALE_FT

        with open(os.path.join(self.grip_list[self.frame_count][1]), 'r') as f:
            robot_state = json.load(f)

        if robot_state is None:
            robot_state = {'gripper': 0.0, 'z': 0.0, 'y': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'lift_effort': 0.0, 'arm_effort': 0.0, 'roll_effort': 0.0, 'pitch_effort': 0.0, 'yaw_effort': 0.0, 'gripper_effort': 0.0}
        
        for key in robot_state.keys():
            if type(robot_state[key]) != float or not np.isfinite(robot_state[key]) or np.isnan(robot_state[key]) or np.abs(robot_state[key]) > 100:
                print('weird val: ', key, robot_state[key])
                robot_state[key] =  0.0

        return img_frame, ft_frame, robot_state

    def keyboard_teleop(self):
        robot_ok, self.pos_dict = read_robot_status(self.client)
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord('q'):     # quit
            self.stop = True
        if keycode == ord(' '):     # toggle moving
            self.enable_moving = not self.enable_moving
        if self.enable_moving and robot_ok:
            if keycode == ord(']'):     # drive X
                self.server.send_payload({'x':-self.delta_lin})
            elif keycode == ord('['):     # drive X
                self.server.send_payload({'x':self.delta_lin})
            elif keycode == ord('a'):     # drive Y
                self.server.send_payload({'y':self.pos_dict['y'] - self.delta_lin})
            elif keycode == ord('d'):     # drive Y
                self.server.send_payload({'y':self.pos_dict['y'] + self.delta_lin})
            elif keycode == ord('s'):     # drive Z
                self.server.send_payload({'z':self.pos_dict['z'] - self.delta_lin})
            elif keycode == ord('w'):     # drive Z
                self.server.send_payload({'z':self.pos_dict['z'] + self.delta_lin})
            elif keycode == ord('u'):     # drive roll
                self.server.send_payload({'roll':self.pos_dict['roll'] - self.delta_ang})
            elif keycode == ord('o'):     # drive roll
                self.server.send_payload({'roll':self.pos_dict['roll'] + self.delta_ang})
            elif keycode == ord('k'):     # drive pitch
                self.server.send_payload({'pitch':self.pos_dict['pitch'] - self.delta_ang})
            elif keycode == ord('i'):     # drive pitch
                self.server.send_payload({'pitch':self.pos_dict['pitch'] + self.delta_ang})
            elif keycode == ord('l'):     # drive yaw
                self.server.send_payload({'yaw':self.pos_dict['yaw'] - self.delta_ang / 2})
            elif keycode == ord('j'):     # drive yaw
                self.server.send_payload({'yaw':self.pos_dict['yaw'] + self.delta_ang / 2})
            elif keycode == ord('b'):     # drive gripper
                self.server.send_payload({'gripper':self.pos_dict['gripper'] - 5})
                # print('commanded pos: ', self.pos_dict['gripper'] - 5)
            elif keycode == ord('n'):     # drive gripper
                self.server.send_payload({'gripper':self.pos_dict['gripper'] + 5})
            
            elif keycode == ord('r'):
                self.demo_flag = True
                # print('commanded pos: ', self.pos_dict['gripper'] + 5)

    def sleep_and_record(self, duration):
        # continues recording video frames while robot is waiting for a command to finish
        robot_ok, pos_dict = read_robot_status(self.client)
        start = time.time()
        while time.time() - start < duration:
            if self.args.record_video:
                delay_start = time.time()
                self.frame = self.feed.get_frame()
                ft_data = self.ft.get_ft()
                output = predict(self.model, self.frame, {'gripper':0})
                force_gt = ft_data[0:3]
                torque_gt = ft_data[3:6]
                force_pred = output[0:3]
                torque_pred = output[3:6]

                if self.args.view:
                    self.fig = self.plotter.visualize_ft(force_gt, torque_gt, force_pred, torque_pred, self.frame, self.collision_flag)
                    cv2.imshow('figure', self.fig)
                    if not self.args.xbox:
                        self.keyboard_teleop()
                    self.result.write(self.fig)

                else:
                    # capture data if not live and record_video
                    save_folder = self.args.video_name + '_' + str(self.video_folder_index)
                    cap_live_data(feed=self.feed, frame=self.frame, ft=self.ft, ft_data=ft_data, pos_dict=pos_dict, save_folder=save_folder)

                delay = time.time() - delay_start
                if (1 / self.video_fps - delay) > 0:
                    time.sleep(1 / self.video_fps - delay) # regulating loop time to self.video_fps frame rate
            else:
                time.sleep(0.01)

    def run_demo(self, control_func):
        self.keys = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        while True:
            robot_ok, pos_dict = read_robot_status(self.client)
            # ft frame to robot frame
            frame_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

            theta = pos_dict['pitch'] # gripper pitch angle
            psi = pos_dict['roll'] # gripper roll angle

            gripper_pitch_rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            gripper_roll_rotation = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
            total_rotation =  frame_rotation @ gripper_pitch_rotation @ gripper_roll_rotation

            self.frame = self.feed.get_frame()
            ft_data = self.ft.get_ft()

            # input to model is current gripper position if live, frame from folder if not live, and 0 if held out
            robot_state = get_robot_state(self.config, pos_dict)

            output = predict(self.model, self.frame, robot_state)

            output_robot_frame = output.copy()
            output_robot_frame[0:3] = output[0:3] @ total_rotation
            output_robot_frame[3:6] = output[3:6] @ total_rotation

            ft_data_robot_frame = ft_data.copy()
            ft_data_robot_frame[0:3] = ft_data[0:3] @ total_rotation
            ft_data_robot_frame[3:6] = ft_data[3:6] @ total_rotation

            pred_dict_robot_frame = dict(zip(self.keys, output_robot_frame.tolist()))
            gt_dict_robot_frame = dict(zip(self.keys, ft_data_robot_frame.tolist()))

            control_func(pred_dict_robot_frame, pos_dict)
            # control_func(gt_dict_robot_frame)

            print('state: ', self.state)
            print('x:', pos_dict['x'])
            print('y:', pos_dict['y'])
            print('z:', pos_dict['z'])
            print('  gt mag: ', np.linalg.norm(ft_data[0:3]))
            print('pred mag: ', np.linalg.norm(output[0:3]))
            self.video_fps = self.feed.frame_count / (time.time() - self.feed.first_frame_time)
            print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))
            print('\n')               
             
            force_gt = ft_data[0:3]
            torque_gt = ft_data[3:6]
            force_pred = output[0:3]
            torque_pred = output[3:6]

            # force_gt_robot_frame = output_robot_frame[0:3]
            # torque_gt_robot_frame = output_robot_frame[3:6]
            # force_pred_robot_frame = output_robot_frame[0:3]
            # torque_pred_robot_frame = output_robot_frame[3:6]

            if np.linalg.norm(force_gt) > 25:
                print('force too high, don\'t hurt robot')
                break

            error = output - ft_data
            self.pred_hist = np.concatenate(([output], self.pred_hist), axis=0)
            self.gt_hist = np.concatenate(([ft_data], self.gt_hist), axis=0)
            self.timestamp_hist = np.concatenate(([self.ft.current_frame_time], self.timestamp_hist), axis=0)

            if self.args.view:
                self.fig = self.plotter.visualize_ft(force_gt, torque_gt, force_pred, torque_pred, self.frame, self.collision_flag)
                # self.fig = self.plotter.visualize_ft(force_gt_robot_frame, torque_gt_robot_frame, force_pred_robot_frame, torque_pred_robot_frame, self.frame, self.collision_flag)

                cv2.imshow('figure', self.fig)
                self.keyboard_teleop()

                if self.args.record_video:
                    self.result.write(self.fig)

            elif self.args.record_video:
                # capture data if not live and record_video
                save_folder = self.args.video_name + '_' + str(self.video_folder_index)
                cap_live_data(feed=self.feed, frame=self.frame, ft=self.ft, ft_data=ft_data, pos_dict=pos_dict, save_folder=save_folder)

            if self.stop:
                break
        
        # removing initialized values from history
        self.pred_hist = self.pred_hist[:-1]
        self.gt_hist = self.gt_hist[:-1]

        # saving history
        hist_dict = {'pred_hist': self.pred_hist.tolist(), 'gt_hist': self.gt_hist.tolist(), 'timestamp_hist': self.timestamp_hist.tolist()}
        with open(self.hist_file_name, 'w') as file:
            file.write(json.dumps(hist_dict))
        print('saved FT history to ', self.hist_file_name)

        if self.args.record_video and self.args.live:
            self.result.release()
            hist_dict = process_ft_history(self.hist_file_name)

    def run_model(self):
        while (not self.args.live) or (self.feed.ret):
            start = time.time()

            if self.args.live:
                self.frame = self.feed.get_frame()
                ft_data = self.ft.get_ft()
                curr_timestamp = self.ft.current_frame_time
                
            else:
                # break if at the end of the offline folder
                if self.frame_count >= len(self.img_list):
                    break
                # self.frame, ft_data, grip_data, lift_effort, pitch_effort, lift_pos  = self.get_folder_frame()
                self.frame, ft_data, robot_state  = self.get_folder_frame()

                curr_timestamp = self.ft_list[self.frame_count][0]

                # look at first part of each folder
                # if self.frame_count % 4500 >= 30:
                #     self.frame_count += (4500 - 30)

                # speed up folder viewing
                speed = 1
                if self.frame_count % speed == 0:
                    self.frame_count += speed

                # self.frame_count += 1
            
            # robot_state = dict()

            # input to model is current gripper position if live, frame from folder if not live, and 0 if held out or robot not ok
            if 'gripper' in self.config.ROBOT_STATES and self.args.live and not self.args.xbox:
                robot_ok, self.pos_dict = read_robot_status(self.client)
                if robot_ok:
                    robot_state['gripper'] = self.pos_dict['gripper']
                   
            # elif 'gripper' in self.config.ROBOT_STATES and not self.args.live and not self.args.xbox:
            #     robot_state['gripper'] = grip_data

            # else:
            #     robot_state['gripper'] = 0

            # if len(self.config.ROBOT_STATES) == 12:
            #     robot_state = {'gripper': 0.0, 'z': 0.0, 'y': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'lift_effort': 0.0, 'arm_effort': 0.0, 'roll_effort': 0.0, 'pitch_effort': 0.0, 'yaw_effort': 0.0, 'gripper_effort': 0.0}

            output = predict(self.model, self.frame, robot_state)
            # output = predict_mlp(self.model, self.frame, robot_state)

            force_gt = ft_data[0:3]
            torque_gt = ft_data[3:6]
            force_pred = output[0:3]
            torque_pred = output[3:6]

            error = output - ft_data
            self.pred_hist = np.concatenate((self.pred_hist, [output]), axis=0)
            self.gt_hist = np.concatenate((self.gt_hist, [ft_data]), axis=0)
            # self.lift_effort_hist = np.concatenate((robot_state['lift_effort'], self.lift_effort_hist), axis=0)
            # self.pitch_effort_hist = np.concatenate((robot_state['pitch_effort'], self.pitch_effort_hist), axis=0)
            # self.lift_pos_hist = np.concatenate((robot_state['lift_pos'], self.lift_pos_hist), axis=0)
            # self.timestamp_hist = np.concatenate((self.timestamp_hist, [curr_timestamp]), axis=0)

            self.demo_flag = False

            if self.args.view:
                fig = self.plotter.visualize_ft(force_gt, torque_gt, force_pred, torque_pred, self.frame, self.collision_flag)

                cv2.imshow('figure', fig)
                if not self.args.xbox:
                    self.keyboard_teleop()
                else:
                    cv2.waitKey(1)

                if self.args.record_video:
                    self.result.write(fig)

                if self.args.record_saliency:
                    saliency_map = gen_saliency_map(img=self.frame, gt=ft_data)
                    cv2.imshow('saliency', saliency_map)
                    self.result.write(saliency_map)

                if self.args.record_gradcam:
                    gradcam_map = gen_gradcam(img=self.frame, gt=ft_data)
                    cv2.imshow('gradcam', gradcam_map)
                    self.result.write(gradcam_map)

                if self.args.record_blur_saliency:
                    blur_saliency_map = gen_blur_saliency(img=self.frame)
                    cv2.imshow('blur saliency', blur_saliency_map)
                    self.result.write(blur_saliency_map)

            if self.stop:
                break

            if self.args.live:
                print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

            print('FPS: ', 1.0 / (time.time() - start))
        
        # removing initialized values from history
        self.pred_hist = self.pred_hist[1:]
        self.gt_hist = self.gt_hist[1:]
        self.lift_effort_hist = self.lift_effort_hist[1:]
        self.pitch_effort_hist = self.pitch_effort_hist[1:]
        self.lift_pos_hist = self.lift_pos_hist[1:]
        self.timestamp_hist = self.timestamp_hist[1:]

        # saving history
        hist_dict = {'pred_hist': self.pred_hist.tolist(),
                    'gt_hist': self.gt_hist.tolist(),
                    'timestamp_hist': self.timestamp_hist.tolist(),
                    'lift_effort_hist': self.lift_effort_hist.tolist(),
                    'pitch_effort_hist': self.pitch_effort_hist.tolist(),
                    'lift_pos_hist': self.lift_pos_hist.tolist()}

        with open(self.hist_file_name, 'w') as file:
            file.write(json.dumps(hist_dict))
        print('saved FT history to ', self.hist_file_name)
        hist_dict = process_ft_history(self.hist_file_name)

        if self.args.record_video:
            self.result.release()
        
if __name__ == '__main__':
    live_model = LiveModel()
    live_model.run_model()