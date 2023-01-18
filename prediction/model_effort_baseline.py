# Model which takes robot state and estimates forces and torques
# robot state input can be set in prediction/pred_utils.py (predict_mlp() function)

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, efficientnet_b0, squeezenet1_1
from prediction.config_utils import *

class Model(nn.Module):
    def __init__(self, gradcam=False):
        super().__init__()
        config, args = parse_config_args()
        self.num_extra_features = len(config.ROBOT_STATES)

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.gradients = None
        self.gradcam = gradcam

        if config.MODEL_NAME == 'resnet18':
            model = resnet18(pretrained=True)
            self.num_visual_features = 512
        elif config.MODEL_NAME == 'resnet50':
            model = resnet50(pretrained=True)
            self.num_visual_features = 2048
        elif config.MODEL_NAME == 'efficientnetb0':
            model = efficientnet_b0(pretrained=True)
            self.num_visual_features = 1280
        elif config.MODEL_NAME == 'squeezenet':
            # len(module.children()) = 2
            model = squeezenet1_1(pretrained=True)
            self.num_visual_features = 1352
            # change the last conv2d layer
            model.classifier._modules["1"] = nn.Conv2d(512, 6, kernel_size=(1, 1))
            # change the internal num_classes variable rather than redefining the forward pass
            model.num_classes = 6

        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.num_extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, img, states):
        if self.num_extra_features > 0:
            x = torch.reshape(states, (states.shape[0], states.shape[1], 1, 1)).float()
        
        model_output = self.fc_layers(x.reshape(-1, self.num_extra_features))

        return model_output