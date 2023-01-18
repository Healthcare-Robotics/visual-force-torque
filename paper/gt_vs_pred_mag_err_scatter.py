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

def graph_mag_err_scatterplot(folder):
    config, args = parse_config_args()
    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = FTData(folder=folder, stage='test', shuffle=False) # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...] 

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    err_mags = np.zeros((len(loader), 3))
    gt_mags = np.zeros((len(loader), 3))

    with torch.no_grad():
        for i, (img, targets, robot_states) in enumerate(loader):
            img = img.to(device)
            targets = targets.to(device)
            robot_states = robot_states.to(device)
            model = model.to(device)
            outputs = model(img, robot_states)
            outputs = outputs / config.SCALE_FT

            error_mag = np.linalg.norm(outputs[0][0:3].cpu().numpy() - targets[0][0:3].cpu().numpy())
            gt_mag = np.linalg.norm(targets[0][0:3].cpu().numpy())

            err_mags[i] = error_mag
            gt_mags[i] = gt_mag 

            # plotting error mag vs gt mag
            plt.scatter(gt_mag, error_mag, marker='.', color='black', s=1)
            plt.xlabel('Ground Truth Force Magnitude')
            plt.ylabel('Force Error Magnitude')
            plt.title('Error Magnitude vs Ground Truth Magnitude')

    plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, err_mags)), (0, 15))
    plt.savefig('./paper/figures/mag_err_scatterplot.png')

    
    axes=['x', 'y', 'z']

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            err_mags = np.zeros((len(loader), 1))
            gt_mags = np.zeros((len(loader), 1))
            for i, (img, targets, robot_states) in enumerate(loader):
                img = img.to(device)
                targets = targets.to(device)
                robot_states = robot_states.to(device)
                model = model.to(device)
                outputs = model(img, robot_states)
                outputs = outputs / config.SCALE_FT

                error_mag = np.linalg.norm(outputs[0][idx].cpu().numpy() - targets[0][idx].cpu().numpy())
                gt_mag = np.linalg.norm(targets[0][idx].cpu().numpy())

                err_mags[i] = error_mag
                gt_mags[i] = gt_mag 

                # plotting error mag vs gt mag
                plt.scatter(gt_mag, error_mag, marker='.', color='black', s=1)
                plt.xlabel(axis + ' Ground Truth Force Magnitude')
                plt.ylabel(axis + ' Force Error Magnitude')
                plt.title(axis + ' Error Magnitude vs Ground Truth Magnitude')

            plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, err_mags)), (0, 15))
            plt.savefig('./paper/figures/' + axis + '_mag_err_scatterplot.png')

if __name__ == '__main__':
    graph_mag_err_scatterplot('./data/test/')