import cv2
import robot.zmq_server as zmq_server
import robot.zmq_client as zmq_client
from robot.robot_utils import *
from recording.gripper_camera import Camera

HOME_POS_DICT = {'y': 0.1, 'z': 0.75, 'roll': 0, 'pitch': -0.3, 'yaw': 0, 'gripper': 55}

def main():
    client = zmq_client.SocketThreadedClient(ip=zmq_client.IP_ROBOT, port=zmq_client.PORT_STATUS_SERVER)
    server = zmq_server.SocketServer(port=zmq_client.PORT_COMMAND_SERVER)
    feed = Camera(view=True)
    delta_lin = 0.02
    delta_ang = 0.1
    enable_moving = True

    # server.send_payload(HOME_POS_DICT)
    # print('Moving to home')

    while True:
        robot_ok, pos_dict = read_robot_status(client)
        frame = feed.get_frame()

        cv2.imshow("frames", frame)

        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord(' '):     # toggle moving
            enable_moving = not enable_moving
        elif keycode == ord('h'):     # go home
            enable_moving = False
            server.send_payload(HOME_POS_DICT)
        if enable_moving:
            if keycode == ord('['):     # drive X
                server.send_payload({'x':-delta_lin})
            elif keycode == ord(']'):     # drive X
                server.send_payload({'x':delta_lin})
            elif keycode == ord('a'):     # drive Y
                server.send_payload({'y':pos_dict['y'] - delta_lin})
            elif keycode == ord('d'):     # drive Y
                server.send_payload({'y':pos_dict['y'] + delta_lin})
            elif keycode == ord('x'):     # drive Z
                server.send_payload({'z':pos_dict['z'] - delta_lin})
            elif keycode == ord('w'):     # drive Z
                server.send_payload({'z':pos_dict['z'] + delta_lin})
            elif keycode == ord('u'):     # drive roll
                server.send_payload({'roll':pos_dict['roll'] - delta_ang})
            elif keycode == ord('o'):     # drive roll
                server.send_payload({'roll':pos_dict['roll'] + delta_ang})
            elif keycode == ord(','):     # drive pitch
                server.send_payload({'pitch':pos_dict['pitch'] - delta_ang})
            elif keycode == ord('i'):     # drive pitch
                server.send_payload({'pitch':pos_dict['pitch'] + delta_ang})
            elif keycode == ord('j'):     # drive yaw
                server.send_payload({'yaw':pos_dict['yaw'] - delta_ang / 2})
            elif keycode == ord('l'):     # drive yaw
                server.send_payload({'yaw':pos_dict['yaw'] + delta_ang / 2})
            elif keycode == ord('b'):     # drive gripper
                server.send_payload({'gripper':pos_dict['gripper'] - 5})
            elif keycode == ord('n'):     # drive gripper
                server.send_payload({'gripper':pos_dict['gripper'] + 5})
            # elif keycode == ord('t'):
            #     server.send_payload({'theta':pos_dict['theta'] - 10})
            # elif keycode == ord('y'):
            #     server.send_payload({'theta':pos_dict['theta'] + 10})
            elif keycode == ord('q'):
                break

if __name__ == "__main__":
    main()