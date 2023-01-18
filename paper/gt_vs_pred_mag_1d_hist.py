import numpy as np
import matplotlib.pyplot as plt
from prediction.config_utils import *
import cv2
from sklearn.metrics import r2_score
import json
import matplotlib.colors as mcolors
import copy


def graph_mag_scatterplot(folder):
    config, args = parse_config_args()


    hist_dict = {'pred_hist': [[0,0,0,0,0,0]], 'gt_hist': [[0,0,0,0,0,0]]}

    for root, dirs, files in os.walk(folder):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    hist_dict_temp = json.load(f)

                    pred_temp = np.array(hist_dict_temp['pred_hist'])
                    gt_temp = np.array(hist_dict_temp['gt_hist'])

                    hist_dict['pred_hist'] = np.concatenate((hist_dict['pred_hist'], pred_temp), axis=0)
                    hist_dict['gt_hist'] = np.concatenate((hist_dict['gt_hist'], gt_temp), axis=0)

    gt = np.array(hist_dict['gt_hist'])
    pred = np.array(hist_dict['pred_hist'])

    print('gt: ', gt[:,0].shape)
    print('pred: ', pred[:,0].shape)

    gt_mags = np.linalg.norm(gt, axis=1)
    pred_mags = np.linalg.norm(pred, axis=1)

    # filtering
    # thresh = 2.5

    # over_thresh = pred[:, 1] >= 2.5
    # pred = pred[over_thresh, :]
    # gt = gt[over_thresh, :]

    # plt.scatter(gt_mags, thetas, marker='.', color='black', s=1)

    nbins = 50
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 10
    my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
    my_cmap.set_bad(my_cmap(0))

    h1 = plt.hist(gt_mags, bins=nbins, color='green', range=[0,10])
    h2 = plt.hist(pred_mags, bins=nbins, color='blue', range=[0,10])
    # h1 = plt.hist(gt[:,0], bins=nbins, color='green', range=[-10,10])
    # h2 = plt.hist(pred[:,0], bins=nbins, color='blue', range=[-10,10])

    # plt.axvline(x=2.5, color='red', linestyle='--')  

    # h = plt.hist2d(gt_mags, thetas, bins=(nbins, nbins), cmap=my_cmap, range=[[0, 10], [0, 180]], cmin=10)

    # cbar = plt.colorbar(h[3], ax=plt.gca())
    plt.xlabel('Force Magnitude (N)')
    plt.ylabel('Num Bins')
    
    plt.savefig('./paper/figures/scatterplots/force_histogram_manikin' + args.config + '.png', dpi=500)
    plt.show()
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    config, args = parse_config_args()
    graph_mag_scatterplot(args.folder)
    # print('xy: ', find_xy_angle([0, 1, 0]))
    # print('xz: ', find_xz_angle([0, 0, -1]))
