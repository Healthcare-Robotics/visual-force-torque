import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import sys
from prediction.transforms import RPFTTransforms
from prediction.config_utils import *
from prediction.pred_utils import *
import json

class FTData(data.Dataset):
    def __init__(self, folder='/data/raw/', stage='raw', shuffle=True):
        self.config, self.args = parse_config_args()
        transform = self.config.TRANSFORM
        self.stage = stage

        # to handle old configs
        if type(folder) == str:
            self.root = [os.path.join(os.getcwd(), folder)]
        
        elif type(folder) == list:
            self.root = folder

        self.t = RPFTTransforms(transform, stage, self.config.PIXEL_MEAN, self.config.PIXEL_STD)
        self.transform = self.t.transforms
        
        self.dataset = self.get_data(shuffle=shuffle)

    def __getitem__(self, index):
        # obtaining file paths
        img_name, ft_name, state_name = self.dataset[index]

        # loading ft data
        ft = np.load(ft_name)
        ft = torch.from_numpy(ft)

        # loading and formatting image
        img = cv2.imread(img_name)

        if 'mask' in self.config.TRANSFORM and self.stage == 'train':
            img = self.t.mask_img_center(img)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float()/255.0
        
        # loading robot state data
        states = torch.tensor([])

        if self.transform is not None:
            img = self.transform(img)

        if 'flip' in self.config.TRANSFORM and self.stage == 'train':
            flipped = np.random.randint(2)
            if flipped:
                img = torch.flip(img, [2])
        
                # mirroring ft data horizontally
                ft[1] = -ft[1] # Fy = horizontal force
                ft[3] = -ft[3] # Tx = yaw torque
                ft[5] = -ft[5] # Tz = roll torque

        with open(state_name, 'r') as f:
            robot_state = json.load(f)
            if robot_state is None:
                robot_state = {'gripper': 0.0, 'z': 0.0, 'y': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'lift_effort': 0.0, 'arm_effort': 0.0, 'roll_effort': 0.0, 'pitch_effort': 0.0, 'yaw_effort': 0.0, 'gripper_effort': 0.0}
            for key in robot_state.keys():
                if type(robot_state[key]) != float or not np.isfinite(robot_state[key]) or np.isnan(robot_state[key]) or np.abs(robot_state[key]) > 100:
                    print('weird val: ', key, robot_state[key])
                    robot_state[key] =  0.0

        if 'gripper' in self.config.ROBOT_STATES:
            gripper_pos = normalize_gripper_vals(robot_state['gripper'])
            states = torch.cat((states, torch.tensor([gripper_pos])), dim=0)
        if 'lift_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['lift_effort']])), dim=0)
        if 'arm_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['arm_effort']])), dim=0)
        if 'roll_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['roll_effort']])), dim=0)
        if 'pitch_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['pitch_effort']])), dim=0)
        if 'yaw_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['yaw_effort']])), dim=0)
        if 'gripper_effort' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['gripper_effort']])), dim=0)
        if 'z' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['z']])), dim=0)
        if 'y' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['y']])), dim=0)
        if 'roll' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['roll']])), dim=0)
        if 'pitch' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['pitch']])), dim=0)
        if 'yaw' in self.config.ROBOT_STATES:
            states = torch.cat((states, torch.tensor([robot_state['yaw']])), dim=0)
        
        if self.config.GRIP_POS_IN_IMG:
            # setting the first 10 rows of the image to the gripper position
            img[:, :10, :] = normalize_gripper_vals(robot_state['gripper'])

        return img, ft, states

    def __len__(self):
        return len(self.dataset)

    def get_data(self, shuffle):
        img_names = []
        ft_names = []
        state_names = []
        self.dataset = []

        # crawling the directory to sort the data modalities 
        for folder in self.root:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    # saving (time, path)
                    parent = root.split('/')[-1]
                    if parent == 'cam' and file.endswith('.jpg'):
                        img_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'ft' and file.endswith('.npy'):
                        ft_names.append((float(file[:-4]), os.path.join(root, file)))
                    elif parent == 'robot_state' and file.endswith('.txt'):
                        state_names.append((float(file[:-4]), os.path.join(root, file)))

        # sorting the names numerically
        img_names = sorted(img_names, key=lambda x: x[0])
        ft_names = sorted(ft_names, key=lambda x: x[0])
        state_names = sorted(state_names, key=lambda x: x[0])

        self.timestamps = [x[0] for x in img_names]

        print(len(img_names), len(ft_names), len(state_names))

        # add img, ft, and None if img and ft sizes are equal and the robot states are disabled
        if len(img_names) == len(ft_names) and (len(img_names) != len(state_names)):
            for i in range(len(img_names)):
                self.dataset.append((img_names[i][1], ft_names[i][1], None))

        # add img, ft, and state data if all three sizes are equal and the robot states are enabled
        elif len(img_names) == len(ft_names) and len(img_names) == len(state_names):
            for i in range(len(img_names)):
                self.dataset.append((img_names[i][1], ft_names[i][1], state_names[i][1]))
        else:
            print('Error: Number of images, ft data, and state data do not match. Check folder sizes and config.')
            sys.exit(1)

        if shuffle:
            np.random.shuffle(self.dataset)
        else:
            self.dataset = np.array(self.dataset)

        # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...]
        return self.dataset

if __name__ == '__main__':
    config, args = parse_config_args()
    dataset = FTData(folder=config.TRAIN_FOLDER, stage='test')
    print("frames in folder: ", len(dataset.dataset))
    for i in range(10):
        img, ft, grip = dataset[i]
        print(img.shape, ft.shape, grip.shape)
        img = img.permute(1, 2, 0)
        cv2.imshow('img', np.array(img))
        cv2.waitKey(0)
        print(grip)