import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

plt.style.use(['science', 'ieee'])

def plot_gt_pred_vs_load_cell_csv():
    num_forces = 5
    pred_means = np.zeros((num_forces,))
    pred_stds = np.zeros((num_forces,))
    gt_means = np.zeros((num_forces,))
    gt_stds = np.zeros((num_forces,))

    df = pd.read_csv('./paper/gt_pred_vs_load_cell.csv')

    for i in range(num_forces):
        pred_data = df[str(2*(i+1)) + 'N Setpoint Pred']
        gt_data = df[str(2*(i+1)) + 'N Setpoint GT']
        
        pred_means[i] = np.mean(pred_data)
        pred_stds[i] = np.std(pred_data)
        gt_means[i] = np.mean(gt_data)
        gt_stds[i] = np.std(gt_data)
    
    # plotting means and error bars
    plt.errorbar(2*np.arange(num_forces) + 2, pred_means, yerr=pred_stds, fmt='o', color='blue', label='Predicted', markersize=3, capsize=2.5)
    plt.errorbar(2*np.arange(num_forces) + 2, gt_means, yerr=gt_stds, fmt='o', color='green', label='F/T Sensor', markersize=3, capsize=2.5)
    plt.plot(2*np.arange(num_forces + 1), 2*np.arange(num_forces + 1), color='black', linestyle='--', label='Load Cell')
    plt.legend()
    plt.xlabel('Load Cell Force (N)')
    plt.ylabel('Predicted/Ground Truth Force (N)')
    plt.savefig('./paper/figures/gt_pred_vs_load_cell.png')
    plt.show()

def plot_gt_pred_vs_load_cell_json():
    data_folder = './paper/fig_data/setpoint/'
    dict_list = []
    for file in os.listdir(data_folder):
        with open(data_folder + file) as f:
            dict_list.append(json.load(f))

    # print('dict_list', dict_list)

    setpoint_list = [dict['setpoint'] for dict in dict_list]
    setpoint_list = list(set(setpoint_list))
    setpoint_list.sort()

    num_forces = len(setpoint_list)
    pred_means = np.zeros((num_forces,))
    pred_stds = np.zeros((num_forces,))
    gt_means = np.zeros((num_forces,))
    gt_stds = np.zeros((num_forces,))

    for i, setpoint in enumerate(setpoint_list):
        pred_data = [dict['pred_force_mag'] for dict in dict_list if dict['setpoint'] == setpoint]
        gt_data = [dict['gt_force_mag'] for dict in dict_list if dict['setpoint'] == setpoint]
        
        pred_means[i] = np.mean(pred_data)
        pred_stds[i] = np.std(pred_data)
        gt_means[i] = np.mean(gt_data)
        gt_stds[i] = np.std(gt_data)
    
    # plotting means and error bars
    plt.errorbar(2*np.arange(num_forces) + 2, pred_means, yerr=pred_stds, fmt='o', color='blue', label='Predicted', markersize=3, capsize=2.5)
    plt.errorbar(2*np.arange(num_forces) + 2, gt_means, yerr=gt_stds, fmt='o', color='green', label='Ground Truth', markersize=3, capsize=2.5)
    # plt.plot(2*np.arange(num_forces + 1), 2*np.arange(num_forces + 1), color='black', linestyle='--', label='Load Cell')
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.legend()
    plt.xlabel('Force Setpoint (N)')
    plt.ylabel('Predicted/Ground Truth Force (N)')
    plt.savefig('./paper/figures/pred_vs_gt_setpoint.png')
    plt.show()

if __name__ == '__main__':
    # plot_gt_pred_vs_load_cell_csv()
    plot_gt_pred_vs_load_cell_json()