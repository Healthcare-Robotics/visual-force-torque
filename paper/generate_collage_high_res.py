from recording.loader import FTData
from prediction.config_utils import *
from prediction.pred_utils import *
from prediction.plotter import Plotter
import cv2
import numpy as np
from prediction.model import Model
import time


def gen_collage(idx):
    config, args = parse_config_args()
    folder = args.folder
    model = Model(gradcam=args.enable_gradcam)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.getcwd(), 'checkpoints/{}_{}/model_{}.pth'.format(args.config, args.index, args.epoch))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    
    img_list, ft_list, grip_list = get_data_lists(folder)
    img = cv2.imread(img_list[idx][1])
    ft = np.load(ft_list[idx][1])
    
    robot_state = {'gripper': 0.0}
    output = predict(model, img, robot_state)

    force_pred = output[0:3]
    torque_pred = output[3:6]
    force_gt = ft[0:3]
    torque_gt = ft[3:6]

    if True:    
    # if np.linalg.norm(force_gt) < 5:
    #     return None

    # else:
        plotter = Plotter(img)

        plotter.render_3d_view(force_gt, torque_gt, force_pred, torque_pred)
        time.sleep(1)
        
        render = cv2.imread('./assets/o3d_frame.png')
        ar = render.shape[1] / render.shape[0]
        # render = cv2.resize(render, (int(img.shape[0] * ar), img.shape[0]))
        img = img[:, :-50, :]
        render = render.astype(np.uint8)

        return img, render

def gen_rand_collages(num_collages):
    config, args = parse_config_args()
    folder = args.folder
    img_list, ft_list, grip_list = get_data_lists(folder)

    save_folder = './paper/collages/'
    folder_len = len(os.listdir(save_folder))
    i = 0

    # for i in range(num_collages):
    while i < num_collages:
        rand_index = np.random.randint(0, len(img_list))
        img_path = img_list[rand_index][1]
        img = cv2.imread(img_path)
        img, render = gen_collage(rand_index)
        cv2.imwrite(save_folder + 'frame_' + str(folder_len + i) + '.png', img)
        cv2.imwrite(save_folder + 'render_' + str(folder_len + i) + '.png', render)
        print('saving frame and render ', folder_len + i)
        i += 1

if __name__ == '__main__':
    gen_rand_collages(num_collages=20)