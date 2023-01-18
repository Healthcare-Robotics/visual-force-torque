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

def find_xy_angle(vec):
    # finds angle in xy plane starting at -y axis (right side from camera perspective) rotating cw about z axis
    return (np.arctan2(-vec[0], -vec[1]) * 180 / np.pi)

def find_xz_angle(vec):
    # finds angle in xz plane starting at z axis (pointing out from camera perspective) rotating cw about y axis
    return (np.arctan2(-vec[0], vec[2]) * 180 / np.pi)

def find_angle_err(pred, gt):
    # finds angle error given two angles in the range [-180, 180]
    err = pred - gt
    if err > 180:
        err -= 360
    elif err <= -180:
        err += 360
    return err

def graph_angle_scatterplot(folder):
    config, args = parse_config_args()
    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model(gradcam=args.enable_gradcam)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = FTData(folder=folder, stage='test', shuffle=False) # the dataset is a list of tuples [(img_name, ft_name, [OPTIONAL] state_name), ...] 

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    pred_xy_angles = np.zeros((len(loader), 3))
    gt_xy_angles = np.zeros((len(loader), 3))
    xy_angle_errs = np.zeros((len(loader), 3))

    pred_xz_angles = np.zeros((len(loader), 3))
    gt_xz_angles = np.zeros((len(loader), 3))
    xz_angle_errs = np.zeros((len(loader), 3))

    with torch.no_grad():
        for i, (img, targets, robot_states) in enumerate(loader):
            img = img.to(device)
            targets = targets.to(device)
            robot_states = robot_states.to(device)
            model = model.to(device)
            outputs = model(img, robot_states)
            outputs = outputs / config.SCALE_FT

            pred_xy_angle = find_xy_angle(outputs[0][0:3].cpu().numpy())
            gt_xy_angle = find_xy_angle(targets[0][0:3].cpu().numpy())

            pred_xy_angles[i] = pred_xy_angle
            gt_xy_angles[i] = gt_xy_angle
            xy_angle_errs[i] = find_angle_err(pred_xy_angle, gt_xy_angle)


            pred_xz_angle = find_xz_angle(outputs[0][0:3].cpu().numpy())
            gt_xz_angle = find_xz_angle(targets[0][0:3].cpu().numpy())

            pred_xz_angles[i] = pred_xz_angle
            gt_xz_angles[i] = gt_xz_angle
            xz_angle_errs[i] = find_angle_err(pred_xz_angle, gt_xz_angle)

    plt.scatter(gt_xy_angles, pred_xy_angles, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth XY Angle (deg)')
    plt.ylabel('Predicted XY Angle (deg)')
    plt.title('Predicted vs Ground Truth XY Angle')
    plt.annotate("r2 = {:.3f}".format(r2_score(gt_xy_angles, pred_xy_angles)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
    
    plt.savefig('./paper/figures/xy_angle_scatterplot_' + args.config + '.png', dpi=500)
    plt.clf()
    plt.cla()

    plt.scatter(gt_xy_angles, xy_angle_errs, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth XY Angle (deg)')
    plt.ylabel('XY Angle Error (deg)')
    plt.title('XY Angle Error vs Ground Truth')
    plt.annotate("r2 = {:.3f}".format(r2_score(gt_xy_angles, xy_angle_errs)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
    
    plt.savefig('./paper/figures/xy_angle_err_scatterplot_' + args.config + '.png', dpi=500)
    plt.clf()
    plt.cla()

    plt.scatter(gt_xz_angles, pred_xz_angles, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth XZ Angle (deg)')
    plt.ylabel('Predicted XZ Angle (deg)')
    plt.title('Predicted vs Ground Truth XZ Angle')
    plt.annotate("r2 = {:.3f}".format(r2_score(gt_xz_angles, pred_xz_angles)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
    
    plt.savefig('./paper/figures/xz_angle_scatterplot_' + args.config + '.png', dpi=500)
    plt.clf()
    plt.cla()

    plt.scatter(gt_xz_angles, xz_angle_errs, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth XZ Angle (deg)')
    plt.ylabel('XZ Angle Error (deg)')
    plt.title('XZ Angle Error vs Ground Truth')
    plt.annotate("r2 = {:.3f}".format(r2_score(gt_xz_angles, xz_angle_errs)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))

    plt.savefig('./paper/figures/xz_angle_err_scatterplot_' + args.config + '.png', dpi=500)
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    graph_angle_scatterplot('./data/test/')
    # print('xy: ', find_xy_angle([0, 1, 0]))
    # print('xz: ', find_xz_angle([0, 0, -1]))
