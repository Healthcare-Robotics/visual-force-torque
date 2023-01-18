import torch
from torch.utils.data import DataLoader
from prediction.model import Model
# from prediction.model_effort_baseline import Model
# from prediction.model_robot_state import Model
# from prediction.model_vit import Model
from recording.loader import FTData
import os
from tqdm import tqdm
from prediction.data_utils import *
from prediction.config_utils import *
from torch.utils.tensorboard import SummaryWriter
from prediction.data_utils import compute_loss_ratio

def train_epoch(model, optimizer, train_loader, loss_ratio):
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    running_f_mse = 0
    running_t_mse = 0
    for (img, targets, robot_states) in tqdm(train_loader):
        img = img.to(device)
        targets = targets.to(device)
        robot_states = robot_states.to(device)
        model = model.to(device)
        optimizer.zero_grad()
        outputs = model(img, robot_states)

        f_outputs = outputs[:, 0:3]
        t_outputs = outputs[:, 3:6]
        
        f_targets = targets[:, 0:3]
        t_targets = targets[:, 3:6]

        f_mse = torch.nn.functional.mse_loss(f_outputs, f_targets)
        t_mse = torch.nn.functional.mse_loss(t_outputs, t_targets)

        # weighting the loss by the relative std of the forces and torques
        train_loss = f_mse + loss_ratio * t_mse
        train_loss.backward()
        optimizer.step()

        running_f_mse += f_mse
        running_t_mse += t_mse

    avg_f_mse = running_f_mse / len(train_loader)
    avg_t_mse = running_t_mse / len(train_loader)

    return avg_f_mse, avg_t_mse

def val_epoch(model, test_loader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_f_mse = 0
    running_t_mse = 0
    
    with torch.no_grad():
        for (img, targets, robot_states) in tqdm(test_loader):
            img = img.to(device)
            targets = targets.to(device) * 0.1 # TODO: remove this once the training data is rescaled (Test data is already rescaled)
            robot_states = robot_states.to(device)
            model = model.to(device)
            outputs = model(img, robot_states)

            f_outputs = outputs[:, 0:3]
            t_outputs = outputs[:, 3:6]
            
            f_targets = targets[:, 0:3]
            t_targets = targets[:, 3:6]

            f_mse = torch.nn.functional.mse_loss(f_outputs, f_targets)
            t_mse = torch.nn.functional.mse_loss(t_outputs, t_targets)

            running_f_mse += f_mse
            running_t_mse += t_mse

    avg_f_mse = running_f_mse / len(test_loader)
    avg_t_mse = running_t_mse / len(test_loader)

    return avg_f_mse, avg_t_mse

if __name__ == "__main__":
    config, args = parse_config_args()

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_DECAY_STEP, gamma=config.LR_DECAY_RATE)

    training_set = FTData(folder=config.TRAIN_FOLDER, stage='train')
    testing_set = FTData(folder=config.TEST_FOLDER, stage='test')

    print("Training set size: ", len(training_set))
    print("Testing set size: ", len(testing_set))

    # loss_ratio = compute_loss_ratio(training_set)
    loss_ratio = config.LOSS_RATIO
    print("Loss ratio: ", loss_ratio)

    train_loader = DataLoader(training_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) # , prefetch_factor=512, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testing_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) # , prefetch_factor=512, pin_memory=True, persistent_workers=True)

    # number of files in ./checkpoints that contain args.config
    folder_index = len([f for f in os.listdir(config.MODEL_DIR) if f.startswith(args.config)])
    writer = SummaryWriter(log_dir='./logs/checkpoint_history/' + args.config + '_' + str(folder_index))

    loss_history = []

    for epoch in range(config.NUM_EPOCHS):
        print("Training epoch {}/{}...".format(epoch + 1, config.NUM_EPOCHS))
        f_mse_train, t_mse_train = train_epoch(model, optimizer, train_loader, loss_ratio)

        print("Validating epoch {}/{}...".format(epoch + 1, config.NUM_EPOCHS))
        f_mse_val, t_mse_val = val_epoch(model, test_loader)

        # weighting the losses by the relative std of the forces and torques in the training data
        train_loss = f_mse_train + loss_ratio * t_mse_train
        val_loss = f_mse_val + loss_ratio * t_mse_val

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('f_mse_train', f_mse_train, epoch)
        writer.add_scalar('t_mse_train', t_mse_train, epoch)
        writer.add_scalar('f_mse_val', f_mse_val, epoch)
        writer.add_scalar('t_mse_val', t_mse_val, epoch)
        writer.flush()

        scheduler.step()
        print('Epoch {}/{}: Training loss = {}, Validation loss = {}'.format(epoch+1, config.NUM_EPOCHS, train_loss, val_loss))

        # creating the checkpoint folder structure
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)

        if not os.path.exists(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index))):
            os.makedirs(os.path.join(config.MODEL_DIR, '{}_{}'.format(args.config, folder_index)))

        model_name = '{}_{}/model_{}.pth'.format(args.config, folder_index, epoch)
        model_path = os.path.join(config.MODEL_DIR, model_name)

        torch.save(model.state_dict(), model_path)
        print('Model saved to {}'.format(model_path))

        # saving losses to a txt file
        with open(os.path.join('logs', args.config + '_' + str(folder_index), '{}_log_{}.txt'.format(args.config, 'raw')), 'a') as f:
            log = model_name + ': Epoch {}/{}: Training loss = {}, Validation loss = {}\n'.format(epoch+1, config.NUM_EPOCHS, train_loss, val_loss)
            loss_history.append((val_loss, model_name))
            f.write(log)

    # saving losses to a txt file
    loss_history.sort(key=lambda x: x[0])
    with open(os.path.join('logs', args.config + '_' + str(folder_index), '{}_log_{}.txt'.format(args.config, 'sorted')), 'a') as f:
        f.write('\n\nBest model: {}\n'.format(loss_history[0][1]))
        f.write('\n\n All Losses:\n')
        for loss in loss_history:
            f.write(loss[1] + ': ' + str(loss[0].item()) + '\n') 