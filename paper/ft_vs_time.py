import numpy as np
import matplotlib.pyplot as plt
from recording.loader import FTData
from prediction.config_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model import Model
from prediction.transforms import RPFTTransforms
import cv2
from sklearn.metrics import r2_score

plt.style.use(['science', 'ieee'])

def graph_ft_vs_time(folder, indices):
    config, args = parse_config_args()

    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model(gradcam=args.enable_gradcam)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = FTData(folder=folder, stage='test', shuffle=False) # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...] 

    t = dataset.timestamps[indices[0]:indices[1]] # timestamps
    t = np.array(t)
    t = t - t[0] # make t start at 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    pred = np.zeros((1, 6))
    gt = np.zeros((1, 6))

    with torch.no_grad():
        for i, (img, targets, robot_states) in enumerate(loader):
            if i > indices[0] and i < indices[1]:
                img = img.to(device)
                targets = targets.to(device)
                robot_states = robot_states.to(device)
                model = model.to(device)
                outputs = model(img, robot_states)
                outputs = outputs / config.SCALE_FT

                # adding the prediction and ground truth to vectors for plotting
                pred = np.append(pred, outputs.cpu().numpy(), axis=0)
                gt = np.append(gt, targets.cpu().numpy(), axis=0)

    print('gt shape: {}'.format(gt.shape))
    print('pred shape: {}'.format(pred.shape))
    print('t shape: {}'.format(t.shape))
    
    linewidth = 1
    fontsize = 8

    legend_labels = ['GT Fx', 'Predicted Fx', 'GT Fy', 'Predicted Fy', 'GT Fz', 'Predicted Fz', 'GT Tx', 'Predicted Tx', 'GT Ty', 'Predicted Ty', 'GT Tz', 'Predicted Tz']
    fig_names = ['Fx vs Time', 'Fy vs Time', 'Fz vs Time', 'Tx vs Time', 'Ty vs Time', 'Tz vs Time']
    titles = ['X', 'Y', 'Z']

    fig, ax =  plt.subplots(2,3)
    fig.set_size_inches(10, 6)

    for i in range(6):
        # ax[0][0].plot(t, gt[:, i], label=legend_labels[2*i+1], color="green", linestyle="-", linewidth=linewidth)
        row = int(i>=3)
        col = int(i%3)
        ax[row][col].plot(t, gt[:, i], label=legend_labels[2*i+1], color="green", linestyle="-", linewidth=linewidth)
        ax[row][col].plot(t, pred[:, i], label=legend_labels[2*i+1], color="blue", linestyle="--", linewidth=linewidth)

        # ax[row][col].set_title(titles[i], fontsize=fontsize)
        # setting the y limits

        if row == 0:
            ax[row][col].set_title(titles[i], fontsize=fontsize)
            ax[row][col].set_ylim([-20, 20])

            if col == 0:
                ax[row][col].set_ylabel('Force (N)', fontsize=fontsize)

        if row == 1:
            ax[row][col].set_xlabel('Time (s)', fontsize=fontsize)
            ax[row][col].set_ylim([-3, 3])

            if col == 0:
                ax[row][col].set_ylabel('Torque (Nm)', fontsize=fontsize)
        
        if col != 0:
            ax[row][col].set_yticklabels([])
    
        if row != 1:
            ax[row][col].set_xticklabels([])
        

    # plt.tight_layout()
    plot_path = './paper/figures/ft_vs_time/{}_{}_{}_{}.png'.format('FT vs Time Combined', folder.split('/')[-1], args.config, args.index, args.epoch)
    plot_path = plot_path.replace(' ', '_')
    plt.savefig(plot_path)
    plt.show()

    for i in range(6):
        plt.plot(t, gt[:, i], label=legend_labels[2*i], color="green", linestyle="-", linewidth=linewidth)
        plt.plot(t, pred[:, i], label=legend_labels[2*i+1], color="blue", linestyle="--", linewidth=linewidth)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", ncol=2, borderaxespad=0.)
        plt.title(fig_names[i], fontsize=fontsize)
        plt.xlabel('Time (s)', fontsize=fontsize)
        if i <=3:
            plt.ylabel('Force (N)', fontsize=fontsize)
        else:
            plt.ylabel('Torque (Nm)', fontsize=fontsize)
        
        plt.tight_layout()
        subplot_path = './paper/figures/ft_vs_time/{}_{}_{}_{}_{}.png'.format(fig_names[i], folder.split('/')[-1], args.config, args.index, args.epoch)
        subplot_path = subplot_path.replace(' ', '_')
        plt.savefig(subplot_path)
        plt.show()

if __name__ == '__main__':
    graph_ft_vs_time(folder='./data/test/test_one_finger_7_17_0', indices=[0, 4500])