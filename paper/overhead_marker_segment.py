import glob
import cv2
import matplotlib.pyplot as plt
# from paper.overhead_marker_train import *
<<<<<<< HEAD
# from sklearn.neighbors import KNeighborsClassifier
=======
from sklearn.neighbors import KNeighborsClassifier
>>>>>>> c144b3cb8d15dc855d6348443b98181b86d0c192
import argparse
import numpy as np


# HUE_THRESH = 10
VAL_THRESH = 140


def get_bbox(img_rgb):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_rgb)

    xlims = None
    ylims = None

    #
    # Declare and register callbacks
    def on_xlims_change(event_ax):
        nonlocal xlims
        print("updated xlims: ", event_ax.get_xlim())
        xlims = event_ax.get_xlim()

    def on_ylims_change(event_ax):
        nonlocal ylims
        print("updated ylims: ", event_ax.get_ylim())
        ylims = event_ax.get_ylim()

    ax.callbacks.connect('xlim_changed', on_xlims_change)
    ax.callbacks.connect('ylim_changed', on_ylims_change)
    plt.show()

    xlims = [int(xlims[0]), int(xlims[1])]
    ylims = [int(ylims[0]), int(ylims[1])]

    return xlims, ylims


def segment_img(before_path, after_path):
    before_bgr = cv2.imread(before_path)
    after_bgr = cv2.imread(after_path)

    xlims, ylims = get_bbox(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB))

    before_bgr = before_bgr[ylims[1]:ylims[0], xlims[0]:xlims[1], :]
    after_bgr = after_bgr[ylims[1]:ylims[0], xlims[0]:xlims[1], :]

    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    before_hsv = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2HSV)

    after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
    after_hsv = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2HSV)

    # before_hue = before_hsv[:, :, 0]
    # after_hue = after_hsv[:, :, 0]

    before_val = before_hsv[:, :, 2]
    after_val = after_hsv[:, :, 2]

    # before_mask = before_hue > HUE_THRESH
    # after_mask = after_hue > HUE_THRESH

    before_mask = before_val < VAL_THRESH
    after_mask = after_val < VAL_THRESH

    before_pixels = np.count_nonzero(before_mask)
    after_pixels = np.count_nonzero(after_mask)

    print('After pixels {}, before pixels {}, ratio {:.3f}'.format(after_pixels, before_pixels, after_pixels/before_pixels))

    plt.subplot(3, 2, 1)
    plt.imshow(before_rgb)
    plt.title('Before rgb')

    # plt.subplot(3, 2, 3)
    # plt.imshow(before_hue)
    # plt.title('Before hue')

    plt.subplot(3, 2, 3)
    plt.imshow(before_val)
    plt.title('Before val')

    plt.subplot(3, 2, 5)
    plt.imshow(before_mask)
    plt.title('Before mask')

    plt.subplot(3, 2, 2)
    plt.imshow(after_rgb)
    plt.title('After rgb')

    # plt.subplot(3, 2, 4)
    # plt.imshow(after_hue)
    # plt.title('After hue')

    plt.subplot(3, 2, 4)
    plt.imshow(after_val)
    plt.title('After val')


    plt.subplot(3, 2, 6)
    plt.imshow(after_mask)
    plt.title('After mask')
    plt.show()


if __name__ == '__main__':
    # HOW TO USE THIS SCRIPT
    # give it a before image and an after image, and it will segment the rest
    # python -m paper.overhead_marker_segment --before data/overhead_marker/test_arm_2_before.jpg --after data/overhead_marker/test_arm_2_after.jpg


    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True)
    parser.add_argument('--after', type=str, required=True)
    args = parser.parse_args()

    segment_img(args.before, args.after)