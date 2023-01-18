import numpy as np
import matplotlib.pyplot as plt
from prediction.config_utils import *
import cv2
from sklearn.metrics import r2_score
import json
import matplotlib.colors as mcolors
import copy

def find_overall_angle(pred, gt):
    # overall angle error
    dot_prods = np.einsum('ij,ij->i', pred[:,0:3], gt[:,0:3])
    gt_mags = np.linalg.norm(gt, axis=1)
    pred_mags = np.linalg.norm(pred, axis=1)

    thetas = np.arccos(dot_prods / (gt_mags * pred_mags)) * 180 / np.pi

    return thetas

def graph_angle_scatterplot(folder):
    config, args = parse_config_args()

    with open('./logs/ft_history/resnet18_final_dataset_crop_lighting_7_2_0.txt') as f:
                hist_dict = json.load(f)

    pred = np.array(hist_dict['pred_hist'])
    gt = np.array(hist_dict['gt_hist'])

    gt_mags = np.linalg.norm(gt, axis=1)
    thetas = find_overall_angle(pred[:,0:3], gt[:,0:3])

    # plt.scatter(gt_mags, thetas, marker='.', color='black', s=1)
    nbins = 100
    my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
    my_cmap.set_bad(my_cmap(0))
    h = plt.hist2d(gt_mags, thetas, bins=(nbins, nbins), cmap=my_cmap, range=[[0, 10], [0, 180]], norm=mcolors.LogNorm(), cmin=10)
    # h = plt.hist2d(gt_mags, thetas, bins=(nbins, nbins), cmap=my_cmap, range=[[0, 10], [0, 180]], cmin=10)

    cbar = plt.colorbar(h[3], ax=plt.gca())
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Angular Error (deg)')
    
    plt.savefig('./paper/figures/scatterplots/overall_angle_vs_mag_2d_hist_' + args.config + '.png', dpi=500)
    plt.show()
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    graph_angle_scatterplot('./data/test/')
    # print('xy: ', find_xy_angle([0, 1, 0]))
    # print('xz: ', find_xz_angle([0, 0, -1]))
