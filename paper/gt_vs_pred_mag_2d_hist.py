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

    pred_mags = np.zeros((len(loader),))
    gt_mags = np.zeros((len(loader),))

    with torch.no_grad():
        for i, (img, targets, robot_states) in enumerate(loader):
            img = img.to(device)
            targets = targets.to(device)
            robot_states = robot_states.to(device)
            model = model.to(device)

            # double check model input
            # frame = img.cpu()
            # frame = frame.squeeze(0)
            # frame = frame.permute(1, 2, 0)
            # cv2.imshow('frame', np.array(frame))
            # cv2.waitKey(1)

            outputs = model(img, robot_states)

            pred_mag = np.linalg.norm(outputs[0][0:3].cpu().numpy())
            gt_mag = np.linalg.norm(targets[0][0:3].cpu().numpy())

            pred_mags[i] = pred_mag
            gt_mags[i] = gt_mag 

    # Estimate the 2D histogram
    nbins = 500
    print('gt_mags shape: ', gt_mags.shape)
    print('pred_mags shape: ', pred_mags.shape)

    H, xedges, yedges = np.histogram2d(gt_mags,pred_mags,bins=nbins)
    
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    
    # Plot 2D histogram using pcolor
    fig2 = plt.figure()
    plt.pcolormesh(xedges,yedges,Hmasked, cmap='inferno')

    # plt.scatter(gt_mags, pred_mags, marker='.', color='black', s=1)
    plt.xlabel('Ground Truth Force Magnitude')
    plt.ylabel('Predicted Force Magnitude')
    plt.title('Predicted Force Magnitude vs Ground Truth Magnitude')
    cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Counts')

    plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, pred_mags)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
    plt.savefig('./paper/figures/scatterplots/mag_2d_histogram_' + args.config + '.png', dpi=500)
    # plt.show()
    plt.clf()
    plt.cla()

    axes=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

    with torch.no_grad():
        for idx, axis in enumerate(axes):
            pred_mags = np.zeros((len(loader),))
            gt_mags = np.zeros((len(loader),))
            for i, (img, targets, robot_states) in enumerate(loader):
                img = img.to(device)
                targets = targets.to(device)
                robot_states = robot_states.to(device)
                model = model.to(device)
                outputs = model(img, robot_states)
                # outputs = outputs / config.SCALE_FT
                outputs = outputs / 0.1

                pred_mag = outputs[0][idx].cpu().numpy()
                gt_mag = targets[0][idx].cpu().numpy()

                pred_mags[i] = pred_mag
                gt_mags[i] = gt_mag 

            # plt.scatter(gt_mags, pred_mags, marker='.', color='black', s=1)
            nbins = 500
            H, xedges, yedges = np.histogram2d(gt_mags,pred_mags,bins=nbins)
    
            # H needs to be rotated and flipped
            H = np.rot90(H)
            H = np.flipud(H)
            
            # Mask zeros
            Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
            
            # Plot 2D histogram using pcolor
            fig2 = plt.figure()
            plt.pcolormesh(xedges,yedges,Hmasked, cmap='inferno')
            plt.xlabel('Ground Truth ' + axis)
            plt.ylabel('Predicted ' + axis)
            plt.title('Predicted ' + axis + ' vs Ground Truth')
            cbar = plt.colorbar()
            # cbar.ax.set_ylabel('Counts')

            # annotating in the top left corner
            pred_mags = np.expand_dims(pred_mags, axis=1)
            gt_mags = np.expand_dims(gt_mags, axis=1)
            plt.annotate("r2 = {:.3f}".format(r2_score(gt_mags, pred_mags)), textcoords='figure fraction', xytext=(0.15, 0.8), xy=(0.15,0.8))
            # saving high resolution version of the figure
            plt.savefig('./paper/figures/scatterplots/' + axis + '_mag_2d_histogram_' + args.config + '.png', dpi=500)
            # plt.show()
            plt.clf()
            plt.cla()

if __name__ == '__main__':
    graph_mag_scatterplot(folder='./data/final_dataset/test/')