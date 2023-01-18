from recording.loader import FTData
from prediction.config_utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_img_list(length):
    config, args = parse_config_args()
    dataset = FTData(folder=args.folder, stage='test', shuffle=False)
    # print("frames in training folder: ", len(dataset.dataset))
    img_list = []
    # for i in range(len):
    idx = 0
    while len(img_list) < length:
        # idx = np.random.randint(0, len(dataset.dataset))
        img, ft, grip = dataset[idx]
        img = img.permute(1, 2, 0)
        img = img.numpy()
        # img = img * 255
        # img = img.astype(np.uint8)
        cv2.imshow('img', img)
        keycode = cv2.waitKey(0) & 0xFF

        # cv2.waitKey(0)
        # if grip.numel() == 1:
            # if grip.item() > 0.8 and grip.item() < 0.9 and torch.norm(ft) > 3:

        if keycode == ord('y'):
            img_list.append(img)
        elif keycode == ord(' '):
            idx += 10
        elif keycode == ord('b'):
            idx -= 1
        elif keycode == ord('n'):
            idx += 1

    return img_list

def gen_def_img():
    img_list = get_img_list(2)
    # overlaying images
    overlay = np.zeros_like(img_list[0], dtype=np.float32)
    # overlay = np.zeros((224, 224), dtype=np.uint8)

    for i, img in enumerate(img_list):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.Canny(img,45,255)
        # cv2.imshow('edges', img)
        # cv2.waitKey(0)

        # plt.imshow(img,cmap = 'gray')
        # plt.show()

        if i == 0:
            overlay = overlay + img * 0.66
        else:
            # adding a red tint to the image
            img = img * 0.33
            img[:, :, 2] = img[:, :, 2] + 0.66
            # img[:,:,1] = 0
            # img[:,:,2] = 0 

            # cv2.imshow('tinted img', img)
            # cv2.waitKey(0)

            overlay = overlay + img * 0.33
            # multiplying the red channel by 1.5

        # overlay = overlay + img

        # overlay = overlay + img * 1 / len(img_list)
    
    diff = np.abs(img_list[0] - img_list[1])
    # converting to grayscale
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow('diff', diff)
    cv2.waitKey(0)

    overlay[:, :, 0] += diff

    return overlay

if __name__ == '__main__':
    img = gen_def_img()
    img = (img*255).astype(np.uint8)
    cv2.imwrite('./paper/images/deflection_img.png', img)
    cv2.imshow('img', img)
    cv2.waitKey(0)