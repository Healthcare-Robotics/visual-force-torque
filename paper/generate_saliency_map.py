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

def gen_saliency_map(img, gt):
    config, args = parse_config_args()

    t = RPFTTransforms(config.TRANSFORM, 'test', config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms

    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model(gradcam=args.enable_gradcam)
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
    model = model.to(device)
    img.requires_grad = True

    robot_state_input = torch.tensor([0])

    # double check model input
    # img = img.squeeze(0)
    # img = img.permute(1, 2, 0)
    # cv2.imshow('img', np.array(img))

    output = model(img, robot_state_input)
    # gradient of this should be similar to dL/dI since GT vals are constant
    # fake_loss = torch.norm(output[0:3])**2 + config.LOSS_RATIO * torch.norm(output[3:6])**2
    # fake_loss.backward()

    f_mse = torch.nn.functional.mse_loss(output[:, 0:3], gt[:, 0:3])
    t_mse = torch.nn.functional.mse_loss(output[:, 3:6], gt[:, 3:6])
    loss = f_mse + config.LOSS_RATIO * t_mse

    loss.backward()

    # output = torch.zeros((1, 6))
    output = output.cpu().detach().numpy().squeeze()

    # fix output for incorrectly scaled models
    output = output / config.SCALE_FT

    output = np.round(output, 6)

    # the saliency is the max gradient of each pixel in the image wrt the loss. Max is taken across channels
    saliency, _ = torch.max(img.grad.data.abs(), dim=1) 
    print('saliency max: ',torch.max(saliency))
    saliency *= 10
    # saliency = saliency / torch.max(saliency)
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    img = img.reshape(-1, 224, 224)

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 1)
    img = img.cpu().detach().numpy().transpose(1, 2, 0)
    saliency.unsqueeze(2)
    saliency = saliency.cpu().detach().numpy()
    saliency = np.uint8(255 * saliency)
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
    mixed_img = cv2.addWeighted(np.uint8(img*255), 0.25, saliency, 0.75, 0)

    # cv2.imshow('img', img)
    # cv2.imshow('saliency', saliency)
    # cv2.imshow('mixed', mixed_img)
    # cv2.waitKey(0)

    return mixed_img

if __name__ == '__main__':
    mixed_img = gen_saliency_map('./assets/one_finger_test_frame.jpg')
    cv2.imshow('mixed', mixed_img)
    cv2.waitKey(0)