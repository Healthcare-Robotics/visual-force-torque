import sys, tty, termios
from dynamixel_sdk import *
import argparse 
import keyboard


class Gripper:
    def __init__(self):
        self.min_pos = 0
        self.max_pos = 4095
        self.delta = 200
        self.curr_pos = 1650
            
        self.ADDR_TORQUE_ENABLE          = 64
        self.ADDR_GOAL_POSITION          = 116
        self.ADDR_PRESENT_POSITION       = 132
        self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
        self.BAUDRATE                    = 115200
        self.PROTOCOL_VERSION            = 2.0
        self.DXL_ID                      = 14
        self.DEVICENAME                  = '/dev/ttyUSB0'
        self.TORQUE_ENABLE               = 1     # Value for enabling the torque
        self.TORQUE_DISABLE              = 0     # Value for disabling the torque
        self.DXL_MOVING_STATUS_THRESHOLD = self.delta # // 2 + 1    # Dynamixel moving status threshold. Must be more than half of the delta value for incrementing the position

    def move_gripper(self, dxl_goal_position):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
            
        portHandler = PortHandler(self.DEVICENAME)
        packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Open port
        if not (portHandler.openPort() and portHandler.setBaudRate(self.BAUDRATE)):
            print("Failed to open the port or set baud rate")
            print("Press any key to terminate...")
            getch()
            quit()

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            # print("Dynamixel has been successfully connected")

            # Write goal position
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, self.DXL_ID, self.ADDR_GOAL_POSITION, dxl_goal_position)
            
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
            self.curr_pos = dxl_present_position

            while abs(dxl_goal_position - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                # Read present position
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
                self.curr_pos = dxl_present_position


                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
                
                elif dxl_error != 0:
                    print("%s" % packetHandler.getRxPacketError(dxl_error))

                print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.DXL_ID, dxl_goal_position, dxl_present_position))

                if abs(dxl_goal_position - dxl_present_position) <= self.DXL_MOVING_STATUS_THRESHOLD:
                    break

        # Close port
        portHandler.closePort()

    def control_gripper(self):
            if keyboard.is_pressed('z'):
                self.move_gripper(self.curr_pos - self.delta)
            if keyboard.is_pressed('x'):
                self.move_gripper(self.curr_pos + self.delta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=int, default=1650)
    args = parser.parse_args()
    gripper = Gripper()
    delta = 200

    while True:
        gripper.control_gripper()

    # gripper.move_gripper(args.pos)