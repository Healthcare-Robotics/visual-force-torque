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

    pred_mags = np.zeros((len(loader), 3))
    gt_mags = np.zeros((len(loader), 3))

    # overall magnitude plot
    with torch.no_grad():
        for i, (img, targets, robot_states) in enumerate(loader):
            img = img.to(device)
            targets = targets.to(device)
            robot_states = robot_states.to(device)
            model = model.to(device)
            outputs = model(img, robot_states)
            outputs = outputs / config.SCALE_FT

            pred_mag = np.linalg.norm(outputs[0][0:3].cpu().numpy())
            gt_mag = np.linalg.norm(targets[0][0:3].cpu().numpy())

            pred_mags[i] = pred_mag
            gt_mags[i] = gt_mag 

    plt.scatter(gt_mags, pred_mags, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth Force Magnitude')
    plt.ylabel('Predicted Force Magnitude')
    plt.title('Predicted Force Magnitude vs Ground Truth Magnitude')

    plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, pred_mags)), (0, 20))
    plt.savefig('./paper/figures/mag_scatterplot_' + args.config + '.png')
    plt.clf()
    plt.cla()


    # axis magnitude plots
    axes=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            pred_mags = np.zeros((len(loader), 1))
            gt_mags = np.zeros((len(loader), 1))
            for i, (img, targets, robot_states) in enumerate(loader):
                img = img.to(device)
                targets = targets.to(device)
                robot_states = robot_states.to(device)
                model = model.to(device)
                outputs = model(img, robot_states)
                outputs = outputs / config.SCALE_FT

                pred_mag = outputs[0][idx].cpu().numpy()
                gt_mag = targets[0][idx].cpu().numpy()

                pred_mags[i] = pred_mag
                gt_mags[i] = gt_mag 

            plt.scatter(gt_mags, pred_mags, marker='.', color='black', s=1)
            plt.xlabel('Ground Truth ' + axis + ' Magnitude')
            plt.ylabel('Predicted ' + axis + ' Magnitude')
            plt.title('Predicted ' + axis + ' Magnitude vs Ground Truth')

            # annotating in the top left corner
            plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, pred_mags)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
            # saving high resolution version of the figure
            plt.savefig('./paper/figures/' + axis + '_mag_scatterplot_' + args.config + '.png', dpi=500)
            plt.clf()
            plt.cla()

if __name__ == '__main__':
    graph_mag_scatterplot(folder='./data/final_dataset/test/')