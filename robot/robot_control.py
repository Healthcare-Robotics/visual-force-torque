#!/usr/bin/env python
import stretch_body.robot
import zmq_client
import zmq_server


class StretchManipulation:
    def __init__(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()

        self.arm_vel = 0.15
        self.arm_accel = 0.15

        self.wrist_vel = 0.000000001 
        self.wrist_accel = 0.0000000001

        self.client = zmq_client.SocketClient(zmq_client.IP_DESKTOP, port=zmq_client.PORT_COMMAND_SERVER)

        self.server = zmq_server.SocketServer(port=zmq_client.PORT_STATUS_SERVER)
        self.thread = zmq_client.DaemonStoppableThread(0.02, target=self.publish_status_loop, name='status_sender')
        self.thread.start()
        print('Socket threads started')

        while True:
            try:
                delta_dict = self.client.receive_blocking()
                print('Received data', delta_dict)
                self.navigate_robot_abs(delta_dict)

            except KeyboardInterrupt:
                self.robot.stop()

    def publish_status_loop(self):
        status = self.robot.get_status()
        self.server.send_payload(status)

    def navigate_robot_abs(self, input_dict):
        print('Lift force', self.robot.lift.status['force'])

        if 'x' in input_dict:
            self.robot.base.translate_by(input_dict['x'], self.arm_vel, self.arm_accel)
        if 'y' in input_dict:
            self.robot.arm.move_to(input_dict['y'], self.arm_vel, self.arm_accel)
        if 'z' in input_dict:
            self.robot.lift.move_to(input_dict['z'], self.arm_vel, self.arm_accel)

        if 'roll' in input_dict:
            self.robot.end_of_arm.move_to('wrist_roll', input_dict['roll'], self.wrist_vel, self.wrist_accel)
        if 'pitch' in input_dict:
            self.robot.end_of_arm.move_to('wrist_pitch', input_dict['pitch'], self.wrist_vel, self.wrist_accel)
        if 'yaw' in input_dict:
            self.robot.end_of_arm.move_to('wrist_yaw', input_dict['yaw'], self.wrist_vel, self.wrist_accel)
        if 'gripper' in input_dict:
            print('moving gripper to ', input_dict['gripper'])
            self.robot.end_of_arm.move_to('stretch_gripper', input_dict['gripper'], self.wrist_vel, self.wrist_accel)

        self.robot.push_command()

if __name__ == '__main__':
    sm = StretchManipulation()