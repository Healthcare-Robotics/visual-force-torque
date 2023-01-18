import stretch_body.robot
import time

robot = stretch_body.robot.Robot()
robot.startup()

arm_vel = 0.01
arm_accel = 0.01
wrist_vel = 0.001 
wrist_accel = 0.001

def level(input_dict):
    if 'x' in input_dict:
        robot.base.translate_by(input_dict['x'], arm_vel, arm_accel)
    if 'y' in input_dict:
        robot.arm.move_to(input_dict['y'], arm_vel, arm_accel)
    if 'z' in input_dict:
        robot.lift.move_to(input_dict['z'], arm_vel, arm_accel)

    if 'roll' in input_dict:
        robot.end_of_arm.move_to('wrist_roll', input_dict['roll'], wrist_vel, wrist_accel)
    if 'pitch' in input_dict:
        robot.end_of_arm.move_to('wrist_pitch', input_dict['pitch'], wrist_vel, wrist_accel)
    if 'yaw' in input_dict:
        robot.end_of_arm.move_to('wrist_yaw', input_dict['yaw'], wrist_vel, wrist_accel)
    if 'gripper' in input_dict:
        print('moving gripper to ', input_dict['gripper'])
        robot.end_of_arm.move_to('stretch_gripper', input_dict['gripper'], wrist_vel, wrist_accel)

    robot.push_command()

if __name__ == '__main__':
    gripper_dict = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper':0}
    level(gripper_dict)