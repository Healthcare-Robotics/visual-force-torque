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

def graph_mag_scatterplot(folder):
    config, args = parse_config_args()
    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model(gradcam=args.enable_gradcam)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = FTData(folder=folder, stage='test', shuffle=False) # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...] 

    train_sampler = RandomSampler(dataset, replacement=True, num_samples=int(len(dataset) * 1))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, sampler=train_sampler)

    pred_mags = np.zeros((len(loader),))
    gt_mags = np.zeros((len(loader),))

    axes=['$F_x$', '$F_y$', '$F_z$', None,'$T_x$', '$T_y$', '$T_z$']

    fig, ax =  plt.subplots(1, 7, figsize=(15, 3))
    hists = []

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            if idx == 0:
                # Fx
                ax[idx].set_ylabel('Predicted Force (N)')
            elif idx == 1:
                # Fy
                ax[idx].set_xlabel('Ground Truth (N)')
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
                ax[idx].set_ylabel('Predicted Torque (Nm)')
            elif idx == 5:
                # Ty
                ax[idx].set_xlabel('Ground Truth (Nm)')
                ax[idx].set_yticklabels([])
            elif idx == 6:
                # Tz
                ax[idx].set_yticklabels([])

            ax[idx].set_title(axis)

            pred_mags = np.zeros((len(loader),))
            gt_mags = np.zeros((len(loader),))
            for i, (img, targets, robot_states) in enumerate(loader):
                img = img.to(device)
                targets = targets.to(device)
                robot_states = robot_states.to(device)
                model = model.to(device)
                outputs = model(img, robot_states)
                # outputs = outputs / config.SCALE_FT

                outputs = outputs.cpu().numpy()
                targets = targets.cpu().numpy()

                # inserting 0 for the missing axis
                outputs = np.insert(outputs, 3, 0, axis=1)
                targets = np.insert(targets, 3, 0, axis=1)

                pred_mag = outputs[0][idx]
                gt_mag = targets[0][idx]

                pred_mags[i] = pred_mag
                gt_mags[i] = gt_mag 

            # pred_mags = np.random.rand(10000) * 20 - 10
            # gt_mags = np.random.rand(10000) * 20 - 10

            if idx < 3:
                # forces
                xlim = 10
                ylim = 10

            elif idx >= 3:
                # torques
                xlim = 2.5
                ylim = 2.5

            nbins = 75

            # ax[idx].set_xlim(xmin=-xlim, xmax=xlim)
            # ax[idx].set_ylim(ymin=-ylim, ymax=ylim)

            my_cmap = copy.copy(plt.cm.get_cmap('jet')) # copy the default cmap
            my_cmap.set_bad(my_cmap(0))

            h = ax[idx].hist2d(gt_mags, pred_mags, bins=(nbins, nbins), cmap=my_cmap, range=[[-xlim, xlim], [-ylim, ylim]], norm=mcolors.LogNorm(vmin=1, vmax=10000), cmin=10)
            hists.append(h[0])

            print(axis, ' max: ', h[0][h[0] > 0].max())

            # y=x
            ax[idx].plot([-xlim, xlim], [-ylim, ylim], ls="--", c='r', lw=0.5)

            # annotating in the top left corner
            pred_mags = np.expand_dims(pred_mags, axis=1)
            gt_mags = np.expand_dims(gt_mags, axis=1)
            ax[idx].annotate("$r^2 =$ {:.3f}".format(r2_score(gt_mags, pred_mags)), textcoords='axes fraction', xytext=(0.15, 0.8), xy=(0.15,0.8), fontsize=7, color='white')
            # ax[idx].annotate("$y = x$", textcoords='axes fraction', xytext=(0.15, 0.7), xy=(0.15,0.7), fontsize=7, color='red')

            ax[idx].set_xlim([-xlim, xlim])
            ax[idx].set_ylim([-ylim, ylim])
            ax[idx].set_aspect('equal')

        fig.subplots_adjust(right=0.825)
        
        im = ax[-1].imshow(h[0], cmap=my_cmap, norm=mcolors.LogNorm(vmin=2, vmax=10000))
        cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
        fig.colorbar(im, cax=cbar_ax)

        fig.savefig('./paper/figures/scatterplots/mag_2d_histogram_combined_' + args.config + '.png', dpi=500)
        plt.show()

if __name__ == '__main__':
    graph_mag_scatterplot(folder='./data/final_dataset/test/')