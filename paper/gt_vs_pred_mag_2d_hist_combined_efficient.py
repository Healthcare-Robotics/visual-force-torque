import numpy as np
import matplotlib.pyplot as plt
from recording.loader import FTData
from prediction.config_utils import *
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from prediction.model import Model
from prediction.transforms import RPFTTransforms
import cv2
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
import copy
import json

plt.style.use(['science', 'ieee'])

def graph_mag_scatterplot():
    config, args = parse_config_args()

    axes=['$F_x$', '$F_y$', '$F_z$', None,'$T_x$', '$T_y$', '$T_z$']

    fig, ax =  plt.subplots(1, 7, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.125, 1, 1, 1]})
    
    hists = []

    with open(args.folder) as f:
                hist_dict = json.load(f)

    pred = np.array(hist_dict['pred_hist'])
    gt = np.array(hist_dict['gt_hist'])

    # making the x and y axes negative
    gt[:, 0] = -gt[:, 0] # Fx
    gt[:, 1] = -gt[:, 1] # Fy
    gt[:, 3] = -gt[:, 3] # Tx
    gt[:, 4] = -gt[:, 4] # Ty

    pred[:, 0] = -pred[:, 0] # Fx
    pred[:, 1] = -pred[:, 1] # Fy
    pred[:, 3] = -pred[:, 3] # Tx
    pred[:, 4] = -pred[:, 4] # Ty

    # gt_initial_mag = np.linalg.norm(gt[:,0:3], axis=1)
    # f_min = 3

    # # filtering out the values where the GT force is small
    # pred = pred[gt_initial_mag > f_min, :]
    # gt = gt[gt_initial_mag > f_min, :]

    # filtering out all values where any element is less than 1

    # filter_shape = pred[gt[:, 0] > 1].shape
    # pred = np.zeros(filter_shape)
    # gt = np.zeros(filter_shape)
    # for axis in range(6):
    #     print(pred[:, axis].shape)
    #     print(pred[gt[:, axis] > 1].shape)
    #     print(pred[gt[:, axis] > 1, axis].shape)


    #     pred[:, axis] = pred[gt[:, axis] > 1, axis]
    #     gt[:, axis] = gt[gt[:, axis] > 1, axis]

    # # filtering out individual axes
    # # pred[:,0:3] = pred[:,0:3][pred[:,0:3] > 1, :]
    # # gt[:,0:3] = gt[:,0:3][gt[:,0:3] > 1, :]
    # # pred[:,3:6] = pred[:,3:6][pred[:,3:6] > 1, :]
    # # gt[:,3:6] = gt[:,3:6][gt[:,3:6] > 1, :]

    # inserting 0 for the missing axis
    pred = np.insert(pred, 3, 0, axis=1)
    gt = np.insert(gt, 3, 0, axis=1)


    print(pred.shape)
    print(gt.shape)

    label_fontsize = 13

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            if idx == 0:
                # Fx
                ax[idx].set_ylabel('Estimated Force (N)', fontsize=label_fontsize, labelpad=5)
                ax[idx].set_yticks([-10, -5, 0, 5, 10])

            elif idx == 1:
                # Fy
                # making a large x label
                ax[idx].set_xlabel('Ground Truth Force (N)', fontsize=label_fontsize, labelpad=15)
                ax[idx].set_yticklabels([])

            elif idx == 2:
                # Fz
                ax[idx].set_yticklabels([])

            elif idx == 3:
                # None
                ax[idx].remove()
                continue

            elif idx == 4:
                # Tx
                ax[idx].set_ylabel('Estimated Torque (Nm)', fontsize=label_fontsize, labelpad=7)
                ax[idx].set_yticks([-2, -1, 0, 1, 2])

            elif idx == 5:
                # Ty
                ax[idx].set_xlabel('Ground Truth Torque (Nm)', fontsize=label_fontsize, labelpad=15)
                ax[idx].set_yticklabels([])
            elif idx == 6:
                # Tz
                ax[idx].set_yticklabels([])

            ax[idx].set_title(axis, fontsize=label_fontsize)
            
            pred_mags = pred[:, idx]
            gt_mags = gt[:, idx]

            if idx < 3:
                # forces
                xlim = 10
                ylim = 10
                ax[idx].set_xticks([-10, -5, 0, 5, 10])

            elif idx >= 3:
                # torques
                xlim = 2.5
                ylim = 2.5
                ax[idx].set_xticks([-2, -1, 0, 1, 2])

            nbins = 75

            # ax[idx].set_xlim(xmin=-xlim, xmax=xlim)
            # ax[idx].set_ylim(ymin=-ylim, ymax=ylim)

            my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
            my_cmap.set_bad(my_cmap(0))

            # filtering mags
            pred_mags_filtered = pred_mags[np.abs(gt_mags) > 0.5]
            gt_mags_filtered = gt_mags[np.abs(gt_mags) > 0.5]


            # h = ax[idx].hist2d(gt_mags_filtered, pred_mags_filtered, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], norm=mcolors.LogNorm(vmin=1, vmax=10000), cmin=10)
            h = ax[idx].hist2d(gt_mags, pred_mags, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], norm=mcolors.LogNorm(vmin=1, vmax=10000), cmin=2)
            # h = ax[idx].hist2d(gt_mags, pred_mags, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], cmin=10)

            hists.append(h[0])

            # print(axis, ' max: ', h[0][h[0] > 0].max())

            # y=x
            ax[idx].plot([-xlim, xlim], [-ylim, ylim], ls="--", c='r', lw=0.5)
            # ax[idx].annotate("$y = x$", textcoords='axes fraction', xytext=(0.15, 0.7), xy=(0.15,0.7), fontsize=7, color='red')

            # annotating in the top left corner
            pred_mags = np.expand_dims(pred_mags, axis=1)
            gt_mags = np.expand_dims(gt_mags, axis=1)

            # pred_mags_filtered = np.expand_dims(pred_mags, axis=1)
            # gt_mags_filtered = np.expand_dims(gt_mags, axis=1)

            # ax[idx].annotate("$r^2 =$ {:.2f}".format(r2_score(y_true=gt_mags, y_pred=pred_mags)), textcoords='axes fraction', xytext=(0.075, 0.85), xy=(0.075,0.85), fontsize=12, color='white')
            # ax[idx].annotate("$r^2 =$ {:.2f}".format(r2_score(y_true=gt_mags_filtered, y_pred=pred_mags_filtered)), textcoords='axes fraction', xytext=(0.075, 0.85), xy=(0.075,0.85), fontsize=12, color='white')
            # ax[idx].annotate("$r^2 =$ {:.2f}".format(r2_score(y_true=gt_mags[np.abs(gt_mags) > 1], y_pred=pred_mags[np.abs(gt_mags) > 1])), textcoords='axes fraction', xytext=(0.075, 0.85), xy=(0.075,0.85), fontsize=12, color='white')

            ax[idx].set_xlim([-xlim, xlim])
            ax[idx].set_ylim([-ylim, ylim])
            ax[idx].set_aspect('equal')
            ax[idx].tick_params(axis='both', which='major', width=1, length=5)

        # fig.subplots_adjust(right=0.825)
        
        # im = ax[-1].imshow(h[0], cmap=my_cmap, norm=mcolors.LogNorm(vmin=2, vmax=10000))
        # cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])

        # fig.colorbar(im, cax=cbar_ax, label='Samples Per Pixel')
        # cbar_ax.yaxis.label.set_size(label_fontsize)
        # cbar_ax.yaxis.labelpad = 15

        fig.savefig('./paper/figures/scatterplots/mag_2d_histogram_combined_' + args.config + '.png', dpi=500, bbox_inches='tight')
        # plt.show()

if __name__ == '__main__':
    graph_mag_scatterplot()