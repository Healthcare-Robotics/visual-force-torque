import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from recording.loader import FTData
from prediction.config_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model import Model
from prediction.transforms import RPFTTransforms
import cv2
from sklearn.metrics import r2_score
import json

plt.style.use(['science', 'ieee'])

def graph_ft_vs_time(indices):
    config, args = parse_config_args()

    with open(args.folder) as f:
        hist_dict = json.load(f)

    pred = np.array(hist_dict['pred_hist'])[indices[0]:indices[1]]
    gt = np.array(hist_dict['gt_hist'])[indices[0]:indices[1]]
    lift_effort = np.array(hist_dict['lift_effort_hist'])[indices[0]:indices[1]]
    pitch_effort = np.array(hist_dict['pitch_effort_hist'])[indices[0]:indices[1]]
    lift_pos = np.array(hist_dict['lift_pos_hist'])[indices[0]:indices[1]]

    pred_mag = np.linalg.norm(pred, axis=1)
    gt_mag = np.linalg.norm(gt, axis=1)

    # lining current up with forces
    # lift_effort_scale = 0.2
    # lift_effort = lift_effort * lift_effort_scale + (gt_mag[0] - lift_effort[0] * lift_effort_scale)

    # making the x and y axes negative
    gt[:, 0] = -gt[:, 0] # Fx
    gt[:, 1] = -gt[:, 1] # Fy
    gt[:, 3] = -gt[:, 3] # Tx
    gt[:, 4] = -gt[:, 4] # Ty

    pred[:, 0] = -pred[:, 0] # Fx
    pred[:, 1] = -pred[:, 1] # Fy
    pred[:, 3] = -pred[:, 3] # Tx
    pred[:, 4] = -pred[:, 4] # Ty

    # filtering
    thresh = 3

    over_thresh = pred[:, 1] >= thresh
    pred = pred[over_thresh, :]
    gt = gt[over_thresh, :]



    fps = 30
    timespan = gt.shape[0] / fps # seconds
    t = np.linspace(0, timespan, gt.shape[0])

    print('gt shape: {}'.format(gt.shape))
    print('pred shape: {}'.format(pred.shape))
    print('effort shape: {}'.format(lift_effort.shape))
    print('t shape: {}'.format(t.shape))

    print('Thresh GT Fy: {}'.format(gt[0, 1]))
    print('Thresh pred Fy: {}'.format(pred[0, 1]))

    # print('max GT Fy: {}'.format(np.max(gt[:, 1])))
    # print('max pred Fy: {}'.format(np.max(pred[:, 1])))

    linewidth = 1
    fontsize = 8

    # fig = plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    fig.set_size_inches(3, 3)

    # fig.plot(t, gt[:, i], label=legend_labels[2*i+1], color="green", linestyle="-", linewidth=linewidth)
    # fig.plot(t, pred[:, i], label=legend_labels[2*i+1], color="blue", linestyle="--", linewidth=linewidth)
    # ax1.plot(t, gt_mag, label='GT', color="green", linestyle="-", linewidth=linewidth)
    # ax1.plot(t, pred_mag, label='Prediction', color="blue", linestyle="--", linewidth=linewidth)
    ax1.plot(t, gt[:,0], label='Ground Truth', color="green", linestyle="-", linewidth=linewidth)
    ax1.plot(t, pred[:,0], label='Prediction', color="blue", linestyle="--", linewidth=linewidth)
    # ax2.plot(t, lift_effort, label='Lift Motor Effort', color="orange", linestyle="--", linewidth=linewidth)
    # ax2.plot(t, pitch_effort, label='Pitch Motor Effort', color="red", linestyle="--", linewidth=linewidth)
    # ax2.plot(t, (lift_pos-lift_pos[0])*100, label='Lift Position', color="purple", linestyle="--", linewidth=linewidth)

    # fig.set_title(titles[i], fontsize=fontsize)
    # setting the y limits

    
    # ax2.set_ylim([-1, 1])
    ax1.set_xlabel('Time (s)', fontsize=fontsize, labelpad=5)
    ax1.set_ylabel('X-Axis Force (N)', fontsize=fontsize, labelpad=0)
    ax2.set_ylabel('Motor Effort', fontsize=fontsize, labelpad=0)
    fig.legend(loc='center left', bbox_to_anchor=(0.575, 0.79), fontsize=6)

    ax1.set_xlim([0, t.max()])
    ax1.set_ylim([-12.5, 12.5])
    ax2.set_ylim([-50, 50])

    # ax1.xaxis.set_major_locator(MultipleLocator(5))
    # ax1.yaxis.set_major_locator(MultipleLocator(5))

    # ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    # ax1.grid(which='both')

    # ax1.grid(which='major', color='#CCCCCC', linestyle='--')
    # ax1.grid(which='minor', color='#CCCCCC', linestyle='--')


    plot_path = './paper/figures/ft_vs_time/{}_{}_{}_{}.png'.format('FT Effort vs Time Combined', args.config, args.index, args.epoch)
    plot_path = plot_path.replace(' ', '_')
    plt.savefig(plot_path)
    plt.show()


if __name__ == '__main__':
    # graph_ft_vs_time(indices=[400*30, 520*30])
    # graph_ft_vs_time(indices=[560*30, 620*30])
    # graph_ft_vs_time(indices=[1269*30, 1436*30])
    # graph_ft_vs_time(indices=[1336*30, 1436*30])
    # graph_ft_vs_time(indices=[40*30, 100*30])
    # graph_ft_vs_time(indices=[30*30, 60*30])
    # graph_ft_vs_time(indices=[1557*30, 1580*30])
    graph_ft_vs_time(indices=[0, -1])
