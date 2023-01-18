import numpy as np
import matplotlib.pyplot as plt
from recording.loader import FTData
from prediction.config_utils import *
from prediction.pred_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model import Model
from prediction.transforms import RPFTTransforms
# from prediction.live_model import get_data_lists
import cv2

def gen_blur_saliency(img, grid_size=16):
    config, args = parse_config_args()

    t = RPFTTransforms(config.TRANSFORM, 'test', config.PIXEL_MEAN, config.PIXEL_STD)
    transform = t.transforms

    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model = Model()
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

    robot_state_input = torch.tensor([0])

    output = model(img, robot_state_input)

    x_step = img.size(dim=3) // grid_size
    y_step = img.size(dim=2) // grid_size

    heatmap = np.zeros((grid_size, grid_size))

    for y in range(grid_size):
        for x in range(grid_size):
            img_copy = img.clone()
            # print('img shape', img_copy.size())

            # blurring each image patch
            block_mean = torch.mean(img_copy[:, :, y*y_step:(y+1)*y_step, x*x_step:(x+1)*x_step], dim=(2, 3))
            block_mean = block_mean.unsqueeze(2).unsqueeze(3)
            img_copy[:, :, y*y_step:(y+1)*y_step, x*x_step:(x+1)*x_step] = block_mean

            ablated_output = model(img_copy, robot_state_input)
            # print('output', output)
            # print('ablated output', ablated_output)
            norm = torch.norm(output[:, 0:3] - ablated_output[:, 0:3]) + config.LOSS_RATIO * torch.norm(output[:, 3:6] - ablated_output[:, 3:6])
            heatmap[y, x] = norm.item()
            # print('norm', norm)

            img_copy = img_copy.reshape(-1, 224, 224)
            img_copy = img_copy.cpu().detach().numpy().transpose(1, 2, 0)
            # cv2.imshow('img', np.array(img_copy))
            # cv2.waitKey(0)

    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

    img = img.reshape(-1, 224, 224)
    img = img.cpu().detach().numpy().transpose(1, 2, 0)
    mixed_img = cv2.addWeighted(np.uint8(img*255), 0.5, heatmap, 0.5, 0)

    # cv2.imshow('heatmap', mixed_img)
    # cv2.waitKey(0)
    return mixed_img

def get_random_heatmaps(folder, num_images=10, grid_size=16):
    img_list, ft_list, grip_list = get_data_lists(folder)
    
    save_folder = './paper/heatmaps/'
    folder_len = len(os.listdir(save_folder))

    for i in range(num_images):
        rand_index = np.random.randint(0, len(img_list))
        img_path = img_list[rand_index][1]
        img = cv2.imread(img_path)
        heatmap = gen_blur_saliency(img=img, grid_size=grid_size)
        cv2.imwrite(save_folder + 'heatmap_' + str(folder_len + i) + '.jpg', heatmap)
        print('saving heatmap ', folder_len + i)

if __name__ == '__main__':
    # folder = './data/test/test_one_finger_7_17_0/'
    img_path = './data/test/test_one_finger_7_17_0/cam/1658111180.8431635.jpg'
    gt_path = './data/test/test_one_finger_7_17_0/ft/1658111180.849535.npy'
    img = cv2.imread(img_path)
    gt = np.load(gt_path)
    # heatmap = gen_blur_saliency(img)
    get_random_heatmaps('./data/test/test_one_finger_7_17_0/', num_images=10, grid_size=64)