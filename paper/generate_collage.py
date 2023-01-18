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

    
    # img_list, ft_list, grip_list = get_data_lists(folder)
    # img = cv2.imread(img_list[idx][1])
    # ft = np.load(ft_list[idx][1])

    dataset = FTData(folder=args.folder, stage='test')
    img, ft, grip = dataset[idx]
    print(img.shape)

    img = img.unsqueeze(0)
    output = model(img, grip)

    ft = ft.cpu().detach().numpy()
    output = output.cpu().detach().numpy().squeeze()

    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.cpu().detach().numpy()
    img = img * 255
    img = img.astype(np.uint8)

    # output = predict(model, img, robot_state)

    # cv2.imshow('img1', img)
    # img = transform_img_numpy(img, config)
    # cv2.imshow('img2', img)
    # cv2.waitKey(0)

    force_pred = output[0:3]
    torque_pred = output[3:6]
    force_gt = ft[0:3]
    torque_gt = ft[3:6]

    thresh = 1 if args.soft else 3

    if np.linalg.norm(force_gt) < thresh:
    # if force_gt[0] < 1:
        return None

    else:
        plotter = Plotter(img)

        plotter.render_3d_view(force_gt, torque_gt, force_pred, torque_pred)
        # plotter.render_3d_view(np.array([-3,3,3]), np.array([1,1,1]), force_pred, torque_pred)

        time.sleep(1)
        
        render = cv2.imread('./assets/o3d_frame.png')

        # cropping render
        if args.soft:
            render = render[200:-150, 450:-740]
        else:
            render = render[50:-150, 450:-590]

        ar = img.shape[1] / img.shape[0]

        # resizing img to match render height
        img = cv2.resize(img, (render.shape[0], int(render.shape[0] * ar)))

        render = render.astype(np.uint8)

        collage = np.concatenate([img, render], axis=1)

        return collage

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
        collage = gen_collage(rand_index)
        if collage is not None:
            cv2.imwrite(save_folder + 'collage_' + str(folder_len + i) + '.png', collage)
            print('saving collage ', folder_len + i)
            i += 1

if __name__ == '__main__':
    # gen_rand_collages(folder='./data/final_dataset_held_out_tasks/test', num_collages=10)
    gen_rand_collages(num_collages=50)
