import torch
import numpy as np
from prediction.transforms import *
from prediction.config_utils import *
import json

def predict(model, frame, robot_state):
    config, args = parse_config_args()
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)
    frame = frame.float() / 255.0
    t = RPFTTransforms(config.TRANSFORM, 'test', config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms
    frame = transform(frame)

    frame = frame.unsqueeze(0)

    robot_state['gripper'] = normalize_gripper_vals(robot_state['gripper'])

    if config.GRIP_POS_IN_IMG:
    # setting the first 10 rows of the image to the gripper position
        frame[:, :, :10, :] = robot_state['gripper']

    robot_state_input = torch.tensor([robot_state['gripper']])

    output = model(frame, robot_state_input)
    output = output.detach().numpy().squeeze()

    # fix output for incorrectly scaled models
    output = output / config.SCALE_FT
    output = np.round(output, 6)

    return output

def predict_mlp(model, frame, robot_state):
    config, args = parse_config_args()
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)
    frame = frame.float() / 255.0
    t = RPFTTransforms(config.TRANSFORM, 'test', config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms
    frame = transform(frame)

    frame = frame.unsqueeze(0)

    robot_state['gripper'] = normalize_gripper_vals(robot_state['gripper'])

    if config.GRIP_POS_IN_IMG:
    # setting the first 10 rows of the image to the gripper position
        frame[:, :, :10, :] = robot_state['gripper']

    robot_state_input = torch.tensor([
                                    # [robot_state['gripper']],
                                    [robot_state['lift_effort']],
                                    [robot_state['arm_effort']],
                                    [robot_state['roll_effort']],
                                    [robot_state['pitch_effort']],
                                    [robot_state['yaw_effort']],
                                    [robot_state['gripper_effort']],
                                    # [robot_state['z']],
                                    # [robot_state['y']],
                                    # [robot_state['roll']],
                                    # [robot_state['pitch']],
                                    # [robot_state['yaw']]
                                    ])

    frame = np.zeros_like(frame)
    output = model(frame, robot_state_input)
    output = output.detach().numpy().squeeze()

    # fix output for incorrectly scaled models
    output = output / config.SCALE_FT
    output = np.round(output, 6)

    return output

def normalize_gripper_vals(gripper_val):
    # map the gripper values from [-100, 60] to [0, 1]
    gripper_val = (gripper_val + 100) / 160.0
    if gripper_val < 0:
        gripper_val = 0
    elif gripper_val > 1:
        gripper_val = 1
    elif gripper_val is None or np.isnan(gripper_val):
        # set to 0 if value is invalid
        gripper_val = 0
    return gripper_val

def get_robot_state(config, pos_dict):
    # input to model is current gripper position if live, frame from folder if not live, and 0 if held out
    robot_state = dict()

    if 'gripper' in config.ROBOT_STATES:
        robot_state['gripper'] = pos_dict['gripper']
    else:
        robot_state['gripper'] = 0
    
    return robot_state

def get_data_lists(folder):
        # Returns a list of file names for the images and ft measurements, where the data is sorted temporally
        img_list = []
        ft_list = []
        grip_list = []

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    img_list.append((float(file[:-4]), os.path.join(root, file)))
                elif file.endswith('.npy'):
                    ft_list.append((float(file[:-4]), os.path.join(root, file)))
                elif file.endswith('.txt'):
                    grip_list.append((float(file[:-4]), os.path.join(root, file)))
        
        img_list = sorted(img_list, key=lambda x: x[0])
        ft_list = sorted(ft_list, key=lambda x: x[0])
        grip_list = sorted(grip_list, key=lambda x: x[0])
        
        return img_list, ft_list, grip_list

def cap_live_data(feed, frame, ft, ft_data, pos_dict, save_folder):
        # set file names based on timestamps
        image_name = '{:.3f}'.format(feed.current_frame_time).replace('.', '_') + '.jpg'
        ft_name = '{:.3f}'.format(ft.current_frame_time).replace('.', '_')
        state_name = '{:.3f}'.format(ft.current_frame_time).replace('.', '_') + '.txt'

        # naming the folders where the data will be saved
        image_folder = os.path.join('data', 'raw', save_folder, 'cam')
        ft_folder = os.path.join('data', 'raw', save_folder,'ft')
        state_folder = os.path.join('data', 'raw', save_folder,'robot_state')

        # making directories for data if they doesn't exist
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        if not os.path.exists(ft_folder):
            os.makedirs(ft_folder) 
        if not os.path.exists(state_folder):
            os.makedirs(state_folder)

        # save data to machine
        cv2.imwrite(os.path.join(image_folder, image_name), frame)
        np.save(os.path.join(ft_folder, ft_name), ft_data)

        with open(os.path.join(state_folder, state_name), 'w') as file:
            file.write(json.dumps(pos_dict))
            # file.write(json.dumps({"no_robot_state":None}))

def transform_img_numpy(img, config, stage='test'):
    t = RPFTTransforms(config.TRANSFORM, stage=stage, pixel_mean=config.PIXEL_MEAN, pixel_std=config.PIXEL_STD)
    transform = t.transforms

    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.float()/255.0
    img = img.unsqueeze(0)
    img = transform(img)
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.numpy()

    return img