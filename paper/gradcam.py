import torch
import numpy as np
import matplotlib.pyplot as plt
from recording.loader import FTData
from prediction.config_utils import *
from prediction.pred_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model import Model
from prediction.transforms import RPFTTransforms
import cv2

def gen_gradcam(img, gt):
    config, args = parse_config_args()

    t = RPFTTransforms(config.TRANSFORM, 'test', config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms

    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model(gradcam=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # img = cv2.imread(img_path)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.float() / 255.0
    img = transform(img)
    img = img.unsqueeze(0)

    img = img.to(device)
    gt = torch.from_numpy(gt).to(device)
    model = model.to(device)
    img.requires_grad = True

    robot_state_input = torch.tensor([0])

    # double check model input
    # img = img.squeeze(0)
    # img = img.permute(1, 2, 0)
    # cv2.imshow('img', np.array(img))

    output = model(img, robot_state_input)

    # fix output for incorrectly scaled models
    output = output / config.SCALE_FT

    loss = (torch.norm(output[:, 0:3] - gt[0:3])**2 + config.LOSS_RATIO * torch.norm(output[:, 3:6] - gt[3:6])**2) / 3
    print('loss', loss)
    loss.backward()

    # f_mse = torch.nn.functional.mse_loss(torch.tensor(output[:, 0:3]), torch.tensor(gt[0:3]))
    # t_mse = torch.nn.functional.mse_loss(torch.tensor(output[:, 3:6]), torch.tensor(gt[3:6]))
    # loss = f_mse + config.LOSS_RATIO * t_mse
    # loss.requires_grad = True
    # loss.backward()

    grads = model.get_activation_gradients()

    # pool gradients across channels
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    # get activations of last convolutional layer
    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(model.num_visual_features): # low res
    # for i in range(256): # high res
        activations[:, i, :, :] *= pooled_grads[i]

    # average the channels of the activations, apply ReLU, and normalize
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.cpu().detach().numpy()
    # heatmap = np.maximum(heatmap, 0)
    heatmap = np.abs(heatmap)
    heatmap /= np.max(heatmap)
    print('heatmap shape', heatmap.shape)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

    img = img.reshape(-1, 224, 224)
    img = img.cpu().detach().numpy().transpose(1, 2, 0)

    mixed_img = cv2.addWeighted(np.uint8(img*255), 0.6, heatmap, 0.4, 0)

    # cv2.imshow('img', img)
    # cv2.imshow('heatmap', heatmap)
    # cv2.imshow('mixed', mixed_img)
    # cv2.waitKey(0)

    return mixed_img

if __name__ == '__main__':
    img = cv2.imread('./assets/one_finger_test_frame.jpg')
    mixed_img = gen_gradcam(img=img, gt=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    cv2.imshow('mixed', mixed_img)
    cv2.waitKey(0)