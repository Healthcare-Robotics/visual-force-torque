import cv2
import time

class Camera:
    def __init__(self, resource=0, view=False):
        self.vc = cv2.VideoCapture(resource)
        # self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.first_frame_time = 0
        self.current_frame_time = 0
        self.frame_count = 0
        self.view = view

    def get_frame(self):
        self.ret, frame = self.vc.read()
        self.current_frame_time = time.time()

        if self.first_frame_time == 0:
            self.first_frame_time = time.time()

        if not self.ret:
            print("frame {} was bad".format(self.frame_count))
            
        self.frame_count += 1

        return frame

if __name__ == "__main__":
    feed = Camera(view=True)
    frame = feed.get_frame()

    while feed.ret:
        frame = feed.get_frame()
        print(frame.shape)
        if feed.view:
            cv2.imshow("frames", frame)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        print('Average FPS', feed.frame_count / (time.time() - feed.first_frame_time))
        print(feed.frame_count, ' frames captured')