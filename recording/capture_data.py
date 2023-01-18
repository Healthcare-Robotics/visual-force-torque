import re
import cv2
import sys
import numpy as np
import time
import os
from recording.gripper_camera import Camera
from recording.ft_stretch_v1 import FTCapture
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from prediction.config_utils import *
from prediction.plotter import Plotter
import json

class DataCapture:
    def __init__(self):
        _, self.args = parse_config_args()
        resource = 3 if self.args.on_robot else 0
        self.feed = Camera(resource=resource, view=self.args.view)
        self.ft = FTCapture()
        frame = self.feed.get_frame()
        self.plotter = Plotter(frame)
        self.delta_lin = 0.02
        self.delta_ang = 0.1
        self.enable_moving = True
        self.stop = False

        if self.args.robot_state:
            self.client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
            self.server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)

        print("ROBOT STATE: ", self.args.robot_state)

        # counting the number of folders in the stage folder beginning with args.folder
        folders = os.listdir(os.path.join('data', self.args.stage))
        
        if len(folders) == 0:
            folder_count = 0
        else:
            folder_count = len([f for f in folders if re.match(self.args.folder, f)])

        self.args.folder = self.args.folder + '_' + str(folder_count)

        # naming the folders where the data will be saved
        self.image_folder = os.path.join('data', self.args.stage, self.args.folder, 'cam')
        self.ft_folder = os.path.join('data', self.args.stage, self.args.folder,'ft')
        self.state_folder = os.path.join('data', self.args.stage, self.args.folder,'robot_state')
        
        # making directories for data if they doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.ft_folder):
            os.makedirs(self.ft_folder) 
        if not os.path.exists(self.state_folder):
            os.makedirs(self.state_folder)

    def capture_data(self):
        # get data snapshots
        frame = self.feed.get_frame()
        ft_data = self.ft.get_ft()

        print('gt: ', ft_data)

        if self.args.robot_state:
            robot_ok, self.pos_dict = read_robot_status(self.client)
        else:
            self.pos_dict = None

        # set file names based on timestamps
        image_name = '{:.3f}'.format(self.feed.current_frame_time).replace('.', '_') + '.jpg'
        ft_name = '{:.3f}'.format(self.ft.current_frame_time).replace('.', '_')
        state_name = '{:.3f}'.format(self.ft.current_frame_time).replace('.', '_') + '.txt'

        # save data to machine
        if self.args.stage in ['train', 'test', 'raw']:
            cv2.imwrite(os.path.join(self.image_folder, image_name), frame)
            np.save(os.path.join(self.ft_folder, ft_name), ft_data)

            with open(os.path.join(self.state_folder, state_name), 'w') as file:
                if self.args.robot_state:
                    file.write(json.dumps(self.pos_dict))
                else:
                    file.write(json.dumps({"no_robot_state":None}))

        else:
            print('Invalid stage argument. Please choose train, test, or raw.')
            sys.exit(1)

        result = {'frame':frame, 'frame_time':self.feed.current_frame_time, 'ft_frame':ft_data, 'ft_frame_time':self.ft.current_frame_time, 'robot_state':self.pos_dict}
        
        if self.feed.view:
            disp_time = str(round(self.feed.current_frame_time - self.feed.first_frame_time, 3))
            cv2.putText(frame, disp_time, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("frames", frame)

            if self.args.robot_state:
                # need to access robot state to control with keyboard
                self.keyboard_teleop()
        else:
            cv2.waitKey(1)

        print('Average FPS', self.feed.frame_count / (time.time() - self.feed.first_frame_time))

        return result

    def keyboard_teleop(self):
        robot_ok, self.pos_dict = read_robot_status(self.client)
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord('q'):     # quit
                self.stop = True
        if keycode == ord(' '):     # toggle moving
            self.enable_moving = not self.enable_moving

        if self.enable_moving:
            if keycode == ord(']'):     # drive X
                self.server.send_payload({'x':-self.delta_lin})
            elif keycode == ord('['):     # drive X
                self.server.send_payload({'x':self.delta_lin})
            elif keycode == ord('a'):     # drive Y
                self.server.send_payload({'y':self.pos_dict['y'] - self.delta_lin})
            elif keycode == ord('d'):     # drive Y
                self.server.send_payload({'y':self.pos_dict['y'] + self.delta_lin})
            elif keycode == ord('s'):     # drive Z
                self.server.send_payload({'z':self.pos_dict['z'] - self.delta_lin * 0.25})
            elif keycode == ord('w'):     # drive Z
                self.server.send_payload({'z':self.pos_dict['z'] + self.delta_lin * 0.25})
            elif keycode == ord('u'):     # drive roll
                self.server.send_payload({'roll':self.pos_dict['roll'] - self.delta_ang})
            elif keycode == ord('o'):     # drive roll
                self.server.send_payload({'roll':self.pos_dict['roll'] + self.delta_ang})
            elif keycode == ord('k'):     # drive pitch
                self.server.send_payload({'pitch':self.pos_dict['pitch'] - self.delta_ang})
            elif keycode == ord('i'):     # drive pitch
                self.server.send_payload({'pitch':self.pos_dict['pitch'] + self.delta_ang})
            elif keycode == ord('l'):     # drive yaw
                self.server.send_payload({'yaw':self.pos_dict['yaw'] - self.delta_ang / 1.5})
            elif keycode == ord('j'):     # drive yaw
                self.server.send_payload({'yaw':self.pos_dict['yaw'] + self.delta_ang / 1.5})
            elif keycode == ord('b'):     # drive gripper
                self.server.send_payload({'gripper':self.pos_dict['gripper'] - 5})
            elif keycode == ord('n'):     # drive gripper
                self.server.send_payload({'gripper':self.pos_dict['gripper'] + 5})

if __name__ == "__main__":
    cap = DataCapture()
    delay = []

    while not cap.stop:
        data = cap.capture_data()
        delay.append(data['ft_frame_time'] - data['frame_time'])
        
    print('saved results to {}'.format(os.path.join('data', cap.args.stage, cap.args.folder)))
    print("delay avg:", np.mean(delay))
    print("delay std:", np.std(delay))