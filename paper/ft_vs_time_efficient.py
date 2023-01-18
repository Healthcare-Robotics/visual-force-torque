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

    # model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    # model = Model(gradcam=args.enable_gradcam)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    # dataset = FTData(folder=folder, stage='test', shuffle=False) # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...] 

    # t = dataset.timestamps[indices[0]:indices[1]] # timestamps
    # t = np.array(t)
    # t = t - t[0] # make t start at 0

    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    # pred = np.zeros((1, 6))
    # gt = np.zeros((1, 6))

    with open(args.folder) as f:
        hist_dict = json.load(f)

    pred = np.array(hist_dict['pred_hist'])[indices[0]:indices[1]]
    gt = np.array(hist_dict['gt_hist'])[indices[0]:indices[1]]

    # making the x and y axes negative
    gt[:, 0] = -gt[:, 0] # Fx
    gt[:, 1] = -gt[:, 1] # Fy
    gt[:, 3] = -gt[:, 3] # Tx
    gt[:, 4] = -gt[:, 4] # Ty

    pred[:, 0] = -pred[:, 0] # Fx
    pred[:, 1] = -pred[:, 1] # Fy
    pred[:, 3] = -pred[:, 3] # Tx
    pred[:, 4] = -pred[:, 4] # Ty

    # # filtering
    # thresh = 2.5

    # over_thresh = pred[:, 1] >= 2.5
    # pred = pred[over_thresh, :]
    # gt = gt[over_thresh, :]

    fps = 30
    timespan = gt.shape[0] / fps # seconds
    t = np.linspace(0, timespan, gt.shape[0])

    print('gt shape: {}'.format(gt.shape))
    print('pred shape: {}'.format(pred.shape))
    print('t shape: {}'.format(t.shape))

    # print('Thresh GT Fy: {}'.format(gt[0, 1]))
    # print('Thresh pred Fy: {}'.format(pred[0, 1]))

    # print('max GT Fy: {}'.format(np.max(gt[:, 1])))
    # print('max pred Fy: {}'.format(np.max(pred[:, 1])))

    

    # with torch.no_grad():
        # for i, (img, targets, robot_states) in enumerate(loader):
            # if i > indices[0] and i < indices[1]:
            #     img = img.to(device)
            #     targets = targets.to(device)
            #     robot_states = robot_states.to(device)
            #     model = model.to(device)
            #     outputs = model(img, robot_states)
            #     outputs = outputs / config.SCALE_FT

            #     # adding the prediction and ground truth to vectors for plotting
            #     pred = np.append(pred, outputs.cpu().numpy(), axis=0)
            #     gt = np.append(gt, targets.cpu().numpy(), axis=0)

    linewidth = 1
    fontsize = 8

    # legend_labels = ['GT $F_x$', 'Predicted $F_x$', 'GT $F_y$', 'Predicted $F_y$', 'GT $F_z$', 'Predicted $F_z$', 'GT $T_x$', 'Predicted $T_x$', 'GT $T_y$', 'Predicted $T_y$', 'GT $T_z$', 'Predicted $T_z$']
    legend_labels = ['Ground Truth', 'Estimation']
    fig_names = ['Fx vs Time', 'Fy vs Time', 'Fz vs Time', 'Tx vs Time', 'Ty vs Time', 'Tz vs Time']
    # titles = ['X', 'Y', 'Z']
    titles = ['$F_x$', '$F_y$', '$F_z$', '$T_x$', '$T_y$', '$T_z$']


    fig, ax =  plt.subplots(2,3)
    fig.set_size_inches(5, 3)
    # fig.set_size_inches(50, 3)


    for i in range(6):
        # ax[0][0].plot(t, gt[:, i], label=legend_labels[2*i+1], color="green", linestyle="-", linewidth=linewidth)
        row = int(i>=3)
        col = int(i%3)
        # ax[row][col].plot(t, gt[:, i], label=legend_labels[2*i], color="green", linestyle="-", linewidth=linewidth)
        # ax[row][col].plot(t, pred[:, i], label=legend_labels[2*i+1], color="blue", linestyle="--", linewidth=linewidth)
        ax[row][col].plot(t, gt[:, i], label=legend_labels[0], color="green", linestyle="-", linewidth=linewidth)
        ax[row][col].plot(t, pred[:, i], label=legend_labels[1], color="blue", linestyle="--", linewidth=linewidth)

        # ax[row][col].set_title(titles[i], fontsize=fontsize)
        # setting the y limits

        if row == 0:
            # ax[row][col].set_ylim([-15, 15])
            ax[row][col].set_ylim([-10, 10])
            # ax[row][col].set_ylim([-7.5, 7.5])


            ax[row][col].xaxis.set_major_locator(MultipleLocator(10))
            ax[row][col].yaxis.set_major_locator(MultipleLocator(5))


            if col == 0:
                ax[row][col].set_ylabel('Force (N)', fontsize=fontsize, labelpad=0)

        if row == 1:
            ax[row][col].set_xlabel('Time (s)', fontsize=fontsize)
            # ax[row][col].set_ylim([-3, 3])
            ax[row][col].set_ylim([-2.25, 2.25])
            # ax[row][col].set_ylim([-1.5, 1.5])

            ax[row][col].xaxis.set_major_locator(MultipleLocator(10))
            ax[row][col].yaxis.set_major_locator(MultipleLocator(1))

            if col == 0:
                ax[row][col].set_ylabel('Torque (Nm)', fontsize=fontsize, labelpad=0)
        
        if col != 0:
            ax[row][col].set_yticklabels([])
    
        if row != 1:
            ax[row][col].set_xticklabels([])
        
        if i == 2:
            ax[row][col].legend(loc='center left', bbox_to_anchor=(0.3, 0.825), fontsize=6)


        ax[row][col].set_title(titles[i], fontsize=fontsize)

        # increasing vertical space between subplots
        fig.subplots_adjust(hspace=0.3)
        

        # ax[row][col].xaxis.set_minor_locator(AutoMinorLocator(5))
        # ax[row][col].yaxis.set_minor_locator(AutoMinorLocator(5))

        # ax[row][col].grid(which='both')

        ax[row][col].grid(which='major', color='#CCCCCC', linestyle='--')
        # ax[row][col].grid(which='minor', color='#CCCCCC', linestyle='--')


    # plt.tight_layout()
    plot_path = './paper/figures/ft_vs_time/{}_{}_{}_{}.png'.format('FT vs Time Combined', args.config, args.index, args.epoch)
    plot_path = plot_path.replace(' ', '_')
    plt.savefig(plot_path)
    plt.show()

    # for i in range(6):
    #     plt.plot(t, gt[:, i], label=legend_labels[2*i], color="green", linestyle="-", linewidth=linewidth)
    #     plt.plot(t, pred[:, i], label=legend_labels[2*i+1], color="blue", linestyle="--", linewidth=linewidth)
    #     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", ncol=2, borderaxespad=0.)
    #     plt.title(fig_names[i], fontsize=fontsize)
    #     plt.xlabel('Time (s)', fontsize=fontsize)
    #     if i <=3:
    #         plt.ylabel('Force (N)', fontsize=fontsize)
    #     else:
    #         plt.ylabel('Torque (Nm)', fontsize=fontsize)
        
    #     plt.tight_layout()
    #     subplot_path = './paper/figures/ft_vs_time/{}_{}_{}_{}.png'.format(fig_names[i], args.config, args.index, args.epoch)
    #     subplot_path = subplot_path.replace(' ', '_')
    #     plt.savefig(subplot_path)
    #     plt.show()

if __name__ == '__main__':
    # graph_ft_vs_time(indices=[400*30, 520*30])
    # graph_ft_vs_time(indices=[560*30, 620*30])
    # graph_ft_vs_time(indices=[1269*30, 1436*30])
    # graph_ft_vs_time(indices=[1336*30, 1436*30])
    # graph_ft_vs_time(indices=[1380*30, 1425*30])
    # graph_ft_vs_time(indices=[1400*30, 1500*30])
    # graph_ft_vs_time(indices=[0*30, 500*30])
    # graph_ft_vs_time(indices=[350*30, 390*30])
    # graph_ft_vs_time(indices=[275*30, 340*30])
    graph_ft_vs_time(indices=[0, -1])
