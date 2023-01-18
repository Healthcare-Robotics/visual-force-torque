import numpy as np
import matplotlib.pyplot as plt
from recording.loader import FTData
from prediction.config_utils import *
import cv2
import json
import argparse

def percent_err_vs_gt_mag(file):
    with open(file) as f:
        hist_dict = json.load(f)

    pred_hist = np.array(hist_dict['pred_hist'])
    gt_hist = np.array(hist_dict['gt_hist'])
    timestamp_hist = np.array(hist_dict['timestamp_hist'])

    err_hist = pred_hist - gt_hist

    print('err_hist.shape', err_hist.shape)
    print('gt_hist.shape', gt_hist.shape)
    print('pred_hist.shape', pred_hist.shape)

    # magnitude of error vectors
    # f_err_mag_hist = np.linalg.norm(err_hist[:,0:3], axis=1)
    # t_err_mag_hist = np.linalg.norm(err_hist[:,3:6], axis=1)    
    f_err_mag_hist = np.abs(err_hist[:,2])
    t_err_mag_hist = np.abs(err_hist[:,3])

    # f_gt_mag = np.linalg.norm(gt_hist[:,0:3], axis=1)
    # t_gt_mag = np.linalg.norm(gt_hist[:,3:6], axis=1)
    f_gt_mag = np.abs(gt_hist[:,2])
    t_gt_mag = np.abs(gt_hist[:,3])

    print('f_err_mag_hist.shape', f_err_mag_hist.shape)
    print('f_gt_mag.shape', f_gt_mag.shape)

    f_err_percent = f_err_mag_hist / f_gt_mag
    plt.scatter(f_gt_mag, f_err_percent, s=1)

    plt.ylim(0, 2)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    percent_err_vs_gt_mag(args.file)