import numpy as np
import os
from recording.loader import FTData
from tqdm import tqdm
import os
from PIL import Image
from tqdm import tqdm
import json
from prediction.config_utils import *

def compute_loss_ratio(stage, folder):
    # for computing the loss ratio or validating ft sensor was calibrated correctly
    dataset = FTData(folder=folder, stage=stage)
    dataset = dataset.dataset
    mean = None
    std = None

    count = 0
    force_sum = np.array([0], dtype='float32')
    torque_sum = np.array([0], dtype='float32')

    f_sq_sum = np.array([0], dtype='float32')
    t_sq_sum = np.array([0], dtype='float32')

    for i in tqdm(range(len(dataset))):
        img_path, ft_path, state = dataset[i]
        ft = np.load(ft_path)
        ft = np.array(ft, dtype='float32')
        ft = np.abs(ft)

        forces = ft[0:3]
        torques = ft[3:6]

        force_sum += np.sum(forces)
        torque_sum += np.sum(torques)

        f_sq_sum += np.sum(np.square(forces))
        t_sq_sum += np.sum(np.square(torques))

        count += 1
                
    print(count, " arrays processed")

    count = count * 3.0
                
    force_mean = force_sum / count
    torque_mean = torque_sum / count

    force_var = (f_sq_sum / count) - (force_mean ** 2) 
    torque_var = (t_sq_sum / count) - (torque_mean ** 2)


    force_std = np.sqrt(force_var)
    torque_std = np.sqrt(torque_var)               

    print("force mean: ", force_mean)
    print("torque mean: ", torque_mean)
    print("force stdev: ", force_std)
    print("torque stdev: ", torque_std)
    
    loss_ratio = force_std.item() / torque_std.item()

    return loss_ratio

def compute_img_mean_std(dir_name):

    mean = None
    std = None

    img_size = (640, 480)
    count = 0
    sum = np.array([0, 0, 0], dtype="int64")
    sq_sum = np.array([0, 0, 0], dtype="int64")

    for dirpath, dirnames, img_list in os.walk(dir_name):
        print('processing ', dirpath, '...')
        for img in tqdm(img_list):
            if not dirnames and img.endswith(".jpg"):
                img_path = os.path.join(dirpath, img)
                img = Image.open(img_path)
                img = np.array(img, dtype="uint8")
                for channel in range(3):
                    sum[channel] += np.sum(img[:,:,channel])
                    sq_sum[channel] += np.sum(np.square(img[:,:,channel]))
                count += 1
                
    print(count, " images processed")

    count = count * img_size[0] * img_size[1] 
    sum = sum / 255
    sq_sum = sq_sum / 255
                
    mean = sum / count
    var = (sq_sum / count) - (mean ** 2) 
    print("var: ", var)
    std = np.sqrt(var)          

    print("pixel mean: ", mean)
    print("pixel standard deviation: ", std)
    
    return (mean, std)

def view_gripper_values(dir_name):
    grip_data = []
    grip_vals = np.array([])
    for dirpath, dirnames, grip_list in os.walk(dir_name):
        for grip in grip_list:
            if not dirnames and grip.endswith(".txt"):
                grip_data.append((float(grip[:-4]), os.path.join(dirpath, grip)))

    # sorting the names numerically
    grip_data = sorted(grip_data, key=lambda x: x[0])

    for grip_value, grip_path in grip_data:
        with open(grip_path, 'r') as f:
            robot_state = json.load(f)
            if robot_state is None:
                robot_state = {'gripper': 0.0}
        grip = robot_state['gripper']
        grip_vals = np.append(grip_vals, grip)

    print(grip_vals)
    print('max: ', np.max(grip_vals))
    print('min: ', np.min(grip_vals))

def process_ft_history(file_list):
    print('files: ', file_list)

    if type(file_list) == str:
        file_list = [file_list]

    history = {'pred_hist': [], 'gt_hist': []}

    for file in file_list:
        with open(file) as f:
            hist_dict = json.load(f)
            history['pred_hist'] = history['pred_hist'] + hist_dict['pred_hist']
            history['gt_hist'] = history['gt_hist'] + hist_dict['gt_hist']

            print('len of current file: ', len(hist_dict['pred_hist']))
            print('len of total file: ', len(history['pred_hist']))


    pred_hist = np.array(history['pred_hist'])
    gt_hist = np.array(history['gt_hist'])

    gt_initial_mag = np.linalg.norm(gt_hist[:,0:3], axis=1)

    # filtering out the values where the force is small
    # pred_hist = pred_hist[gt_initial_mag > 1, :]
    # gt_hist = gt_hist[gt_initial_mag > 1, :]

    print('pred_hist: ', pred_hist.shape)
    print('gt_hist: ', gt_hist.shape)

    err_hist = pred_hist - gt_hist
    err_mean = np.mean(np.abs(err_hist), axis=0)
    err_median = np.median(err_hist, axis=0)
    err_std = np.std(err_hist, axis=0)

    ft_mean = np.mean(gt_hist, axis=0)

    # avg magnitude of force and torque on each axis
    ft_abs_mean = np.mean(np.abs(gt_hist), axis=0)

    # avg total magnitude of force and torque
    f_mag_hist = np.linalg.norm(gt_hist[:,0:3], axis=1)
    t_mag_hist = np.linalg.norm(gt_hist[:,3:6], axis=1)
    f_pred_mag_hist = np.linalg.norm(pred_hist[:,0:3], axis=1)
    t_pred_mag_hist = np.linalg.norm(pred_hist[:,3:6], axis=1)
    f_mag_mean = np.mean(f_mag_hist)
    t_mag_mean = np.mean(t_mag_hist)

    f_mag_max = np.max(f_mag_hist)
    t_mag_max = np.max(t_mag_hist)

    # avg total magnitude of error vector
    f_err_mag_hist = np.linalg.norm(err_hist[:,0:3], axis=1)
    t_err_mag_hist = np.linalg.norm(err_hist[:,3:6], axis=1)
    f_err_mag_mean = np.mean(f_err_mag_hist)
    t_err_mag_mean = np.mean(t_err_mag_hist)
    f_err_mag_std = np.std(f_err_mag_hist)
    t_err_mag_std = np.std(t_err_mag_hist)


    # mean abs error between gt and predicted magnitudes
    mag_err_f_hist = np.abs(f_mag_hist - f_pred_mag_hist)
    mag_err_t_hist = np.abs(t_mag_hist - t_pred_mag_hist)
    mag_err_f_mean = np.mean(mag_err_f_hist)
    mag_err_t_mean = np.mean(mag_err_t_hist)
    mag_err_f_std = np.std(np.abs(f_pred_mag_hist - f_mag_hist))
    mag_err_t_std = np.std(np.abs(t_pred_mag_hist - t_mag_hist))

    f_err_mag_norm_avg = np.mean(f_err_mag_hist / f_mag_hist)
    t_err_mag_norm_avg = np.mean(t_err_mag_hist / t_mag_hist)
    
    mag_err_f_norm_avg = np.mean(np.abs(f_pred_mag_hist - f_mag_hist) / f_mag_hist)
    mag_err_t_norm_avg = np.mean(np.abs(t_pred_mag_hist - t_mag_hist) / t_mag_hist)

    # row wise dot product of gt and pred vectors
    f_dp_hist = np.einsum('ij,ij->i', pred_hist[:,0:3], gt_hist[:,0:3])
    t_dp_hist = np.einsum('ij,ij->i', pred_hist[:,3:6], gt_hist[:,3:6])

    # angle between gt and predicted force vectors
    theta_hist = np.arccos(f_dp_hist / (f_mag_hist * f_pred_mag_hist)) * 180 / np.pi
    theta_err_mean = np.mean(theta_hist)
    theta_err_std = np.std(theta_hist)  

    # weighted cosine similarity
    f_dp_norm = f_dp_hist / (f_mag_hist * f_pred_mag_hist)
    norm_mags = f_mag_hist / np.sum(f_mag_hist)
    wcs_f = np.sum(f_dp_norm * norm_mags)

    t_dp_norm = t_dp_hist / (t_mag_hist * t_pred_mag_hist)
    norm_mags = t_mag_hist / np.sum(t_mag_hist)
    wcs_t = np.sum(t_dp_norm * norm_mags)

    wcs_f_2 = np.sum(gt_hist[:,0:3] * pred_hist[:,0:3]) / (np.linalg.norm(f_mag_hist) * np.linalg.norm(f_pred_mag_hist))
    wcs_t_2 = np.sum(gt_hist[:,3:6] * pred_hist[:,3:6]) / (np.linalg.norm(t_mag_hist) * np.linalg.norm(t_pred_mag_hist))

    f_mse = np.mean(np.mean(np.square(err_hist[:,0:3]), axis=1), axis=0)
    t_mse = np.mean(np.mean(np.square(err_hist[:,3:6]), axis=1), axis=0)

    f_rmse = np.sqrt(f_mse)
    t_rmse = np.sqrt(t_mse)

    zg_f_mse = np.mean(np.mean(np.square(gt_hist[:,0:3]), axis=1), axis=0)
    zg_t_mse = np.mean(np.mean(np.square(gt_hist[:,3:6]), axis=1), axis=0)

    mg_f_mse = np.mean(np.mean(np.square(ft_mean[0:3] - gt_hist[:,0:3]), axis=1), axis=0)
    mg_t_mse = np.mean(np.mean(np.square(ft_mean[3:6] - gt_hist[:,3:6]), axis=1), axis=0)

    mg_mse_axes = np.mean(np.square(ft_mean - gt_hist), axis=0)

    axis_rmse = np.sqrt(np.mean(np.square(err_hist), axis=0))

    best_const_guess = np.mean(np.sqrt(np.mean(np.square(gt_hist), axis=0)), axis=0)
    best_const_rmse = np.sqrt(np.mean(np.square(gt_hist - best_const_guess), axis=0))

    print('err_mean: ', err_mean)
    print('err_std:', err_std)
    print('ft_abs_mean:', ft_abs_mean)
    print('f_mag_mean:', f_mag_mean)
    print('t_mag_mean:', t_mag_mean)
    print('f_err_mag_mean:', f_err_mag_mean)
    print('t_err_mag_mean:', t_err_mag_mean)
    print('mag_err_f_mean:', mag_err_f_mean)
    print('mag_err_t_mean:', mag_err_t_mean)
    print('theta_err_mean', theta_err_mean)
    print('f_mag_max:', f_mag_max)
    print('t_mag_max:', t_mag_max)
    print('f_err_mag_norm_avg: ', f_err_mag_norm_avg)
    print('t_err_mag_norm_avg: ', t_err_mag_norm_avg)
    print('mag_err_f_norm_avg:', mag_err_f_norm_avg)
    print('mag_err_t_norm_avg:', mag_err_t_norm_avg)
    print('force weighted cosine similarity: ', wcs_f)
    print('torque weighted cosine similarity: ', wcs_t)
    print('force weighted cosine similarity #2: ', wcs_f_2)
    print('torque weighted cosine similarity #2: ', wcs_t_2)
    print('f_mse: ', f_mse)
    print('t_mse: ', t_mse)
    print('f_rmse: ', f_rmse)
    print('t_rmse: ', t_rmse)
    print('axis_rmse: ', np.round(axis_rmse,3))
    print('zg_f_rmse: ', np.round(np.sqrt(zg_f_mse),3))
    print('zg_t_rmse: ', np.round(np.sqrt(zg_t_mse),3))
    print('mg_f_rmse: ', np.round(np.sqrt(mg_f_mse),3))
    print('mg_t_rmse: ', np.round(np.sqrt(mg_t_mse),3))
    print('mg_rmse: ', np.round(np.sqrt(mg_mse_axes),3))
    print('best_const_rmse: ', np.round(best_const_rmse,3))

    return hist_dict


if __name__ == "__main__":
    config, args = parse_config_args()

    stage = args.stage
    folder = args.folder

    # action_folder = './logs/ft_history/task_eval/bathroom/'
    # files = os.listdir(action_folder)
    # folder = [os.path.join(action_folder, f) for f in files if f.endswith('.txt')]

    if args.action == 'check_ft':
    # cross check if the ft sensor was calibrated correctly and the data is consistent across folders
        folder_list = sorted(os.listdir(os.path.join("data/final_dataset", stage)))
        for f in tqdm(folder_list):
            print('\n folder:' + folder + ':')
            loss_ratio = compute_loss_ratio(stage, os.path.join("data/final_dataset", stage, f))
            print("loss ratio: ", loss_ratio)

    elif args.action == 'loss_ratio':
        loss_ratio = compute_loss_ratio(stage, folder)
        print("loss ratio: ", loss_ratio)
    
    elif args.action == 'img_mean_std':
        mean, std = compute_img_mean_std(folder)
        print("pixel mean: ", mean)
        print("pixel std: ", std)

    elif args.action == 'view_gripper':
        # to check gripper values
        view_gripper_values(dir_name=folder)

    elif args.action == 'process_ft_history':
        # to get stats from recorded ft history
        process_ft_history(file_list=folder)