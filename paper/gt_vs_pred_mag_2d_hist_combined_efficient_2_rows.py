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

    axes=['$F_x$', '$F_y$', '$F_z$','$T_x$', '$T_y$', '$T_z$']

    fig, ax =  plt.subplots(2, 3, figsize=(8, 6))
    
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

    print(pred.shape)
    print(gt.shape)

    label_fontsize = 13

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            row = int(idx >= 3)
            col = int(idx % 3)
            if idx == 0:
                # Fx
                ax[row][col].set_ylabel('Predicted Force (N)', fontsize=label_fontsize, labelpad=5)
                ax[row][col].set_yticks([-10, -5, 0, 5, 10])

            elif idx == 1:
                # Fy
                # making a large x label
                ax[row][col].set_xlabel('Ground Truth (N)', fontsize=label_fontsize, labelpad=10)
                ax[row][col].set_yticklabels([])

            elif idx == 2:
                # Fz
                ax[row][col].set_yticklabels([])

            elif idx == 3:
                # Tx
                ax[row][col].set_ylabel('Predicted Torque (Nm)', fontsize=label_fontsize, labelpad=5)
                ax[row][col].set_yticks([-2, -1, 0, 1, 2])

            elif idx == 4:
                # Ty
                ax[row][col].set_xlabel('Ground Truth (Nm)', fontsize=label_fontsize, labelpad=10)
                ax[row][col].set_yticklabels([])

            elif idx == 5:
                # Tz
                ax[row][col].set_yticklabels([])

            ax[row][col].set_title(axis, fontsize=label_fontsize)
            
            pred_mags = pred[:, idx]
            gt_mags = gt[:, idx]

            if idx < 3:
                # forces
                xlim = 10
                ylim = 10
                ax[row][col].set_xticks([-10, -5, 0, 5, 10])

            elif idx >= 3:
                # torques
                xlim = 2.5
                ylim = 2.5
                ax[row][col].set_xticks([-2, -1, 0, 1, 2])

            nbins = 75

            # ax[row][col].set_xlim(xmin=-xlim, xmax=xlim)
            # ax[row][col].set_ylim(ymin=-ylim, ymax=ylim)

            my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
            my_cmap.set_bad(my_cmap(0))

            h = ax[row][col].hist2d(gt_mags, pred_mags, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], norm=mcolors.LogNorm(vmin=1, vmax=10000), cmin=2)
            # h = ax[row][col].hist2d(gt_mags, pred_mags, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], cmin=10)

            hists.append(h[0])

            # print(axis, ' max: ', h[0][h[0] > 0].max())

            # y=x
            ax[row][col].plot([-xlim, xlim], [-ylim, ylim], ls="--", c='r', lw=0.5)
            # ax[row][col].annotate("$y = x$", textcoords='axes fraction', xytext=(0.15, 0.7), xy=(0.15,0.7), fontsize=7, color='red')

            # annotating in the top left corner
            pred_mags = np.expand_dims(pred_mags, axis=1)
            gt_mags = np.expand_dims(gt_mags, axis=1)
            ax[row][col].annotate("$r^2 =$ {:.2f}".format(r2_score(y_true=gt_mags, y_pred=pred_mags)), textcoords='axes fraction', xytext=(0.15, 0.8), xy=(0.15,0.8), fontsize=9, color='white')

            ax[row][col].set_xlim([-xlim, xlim])
            ax[row][col].set_ylim([-ylim, ylim])
            ax[row][col].set_aspect('equal')
            ax[row][col].tick_params(axis='both', which='major', width=1, length=5)

        fig.subplots_adjust(right=0.825)
        
        im = ax[1][2].imshow(h[0], cmap=my_cmap, norm=mcolors.LogNorm(vmin=2, vmax=10000))
        cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])

        fig.colorbar(im, cax=cbar_ax, label='Samples Per Pixel')
        cbar_ax.yaxis.label.set_size(label_fontsize)
        cbar_ax.yaxis.labelpad = 15

        fig.savefig('./paper/figures/scatterplots/mag_2d_histogram_combined_2_rows_' + args.config + '.png', dpi=500, bbox_inches='tight')
        # plt.show()

if __name__ == '__main__':
    graph_mag_scatterplot()