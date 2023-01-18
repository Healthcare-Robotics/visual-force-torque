import numpy as np
import matplotlib.pyplot as plt
from prediction.config_utils import *
import cv2
from sklearn.metrics import r2_score
import json
import matplotlib.colors as mcolors
import copy


def percent_err_vs_gt(folder):
    config, args = parse_config_args()

    with open('./logs/ft_history/resnet18_final_dataset_crop_lighting_7_2_0.txt') as f:
                hist_dict = json.load(f)

    pred = np.array(hist_dict['pred_hist'])
    gt = np.array(hist_dict['gt_hist'])
    err = np.linalg.norm(pred-gt, axis=1)

    gt_mags = np.linalg.norm(gt, axis=1)
    pred_mags = np.linalg.norm(pred, axis=1)
    f_err_percent = err / gt_mags * 100

    # plt.scatter(gt_mags, thetas, marker='.', color='black', s=1)
    nbins = 100
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 200
    my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
    my_cmap.set_bad(my_cmap(0))

    print('gt_mags: ', gt_mags.shape)
    print('f_err_percent: ', f_err_percent.shape)

    h = plt.hist2d(gt_mags, f_err_percent, bins=(nbins, nbins), cmap=my_cmap, range=[[xmin, xmax], [ymin, ymax]], norm=mcolors.LogNorm())
    # h = plt.hist2d(gt_mags, thetas, bins=(nbins, nbins), cmap=my_cmap, range=[[0, 10], [0, 180]], cmin=10)

    # cbar = plt.colorbar(h[3], ax=plt.gca())
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Force Error / Ground Truth Force (%)')
    
    plt.savefig('./paper/figures/scatterplots/percent_err_vs_gt_2d_hist_' + args.config + '.png', dpi=500)
    plt.show()
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    percent_err_vs_gt('./data/test/')
    # print('xy: ', find_xy_angle([0, 1, 0]))
    # print('xz: ', find_xz_angle([0, 0, -1]))
