import argparse
import os
import yaml
import argparse
from types import SimpleNamespace

def load_config(config_name):
    config_path = os.path.join('./config', config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, default='default')
    parser.add_argument('--epoch', '-e', type=int, default=0, help='model epoch to load')
    parser.add_argument('--delay', '-d', type=int, default=5, help='delay for camera to catch up to ft')
    parser.add_argument('--index', '-i', type=int, default=0, help='keeps track of training sessions using the same config')
    parser.add_argument('--video_name', '-vn', type=str, default='unnamed', help='name of video')
    parser.add_argument('--fullscreen', '-fs', type=bool, default=False, help='fullscreen live model figure')
    parser.add_argument('--live', '-rs', type=bool, default=False, help='use camera stream instead of folder')
    parser.add_argument('--view', '-v', type=bool, default=False, help='view camera and graphs')
    parser.add_argument('--folder', '-f', type=str, default=None, help='folder for data_capture or folder to pull data from if not live')
    parser.add_argument('--stage', '-s', type=str, default=None, help='train, test, or raw')
    parser.add_argument('--action', '-a', type=str, default=None, help='action for data_utils')
    parser.add_argument('--on_robot', '-r', type=bool, default=False, help='run the model on the robot')
    parser.add_argument('--robot_state', type=bool, default=False, help='record robot_state')
    parser.add_argument('--enable_gradcam', type=bool, default=False, help='configure the model to use gradcam')
    parser.add_argument('--xbox', type=bool, default=False, help='use xbox controller for teleop')
    parser.add_argument('--record_video', '-rv', type=bool, default=False, help='record video')
    parser.add_argument('--record_saliency', type=bool, default=False, help='record saliency. does not work with record_video or record_gradcam')
    parser.add_argument('--record_gradcam', type=bool, default=False, help='record gradcam. does not work with record_video or record_saliency')
    parser.add_argument('--record_blur_saliency', type=bool, default=False, help='record blur saliency. does not work with record_video or record_saliency')
    parser.add_argument('--soft', type=bool, default=False, help='use the soft gripper')
    parser.add_argument('--fps', type=int, default=30, help='fps for recording videos')

    args = parser.parse_args()
    return load_config(args.config), args