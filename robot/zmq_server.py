import zmq
import time
import json
import numpy as np

class SocketServer:
    def __init__(self, port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:" + str(port))     # Broadcast to any subscribers on the network
        self.port = port
        self.message_id = 0

    def send_payload(self, payload_dict):
        payload_dict['message_id'] = self.message_id
        self.socket.send_string(json.dumps(payload_dict, indent=4))
        self.message_id += 1

if __name__ == "__main__":
    ss = SocketServer()
    t = 0
    while True:
        delta_z = np.sin(2*np.pi*t / 200) / 1
        # del
        payload_dict = {'delta_z': delta_z}
        print(payload_dict)
        ss.send_payload(payload_dict)
        time.sleep(0.05)
        t += 1