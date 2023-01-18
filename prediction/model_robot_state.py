# Model which takes in an image and the robot state and estimates forces and torques
# robot state input can be set in prediction/pred_utils.py (predict() function)
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
            # changing the last conv2d layer
            model.classifier._modules["1"] = nn.Conv2d(512, 6, kernel_size=(1, 1))
            # changing the internal num_classes variable rather than redefining the forward pass
            model.num_classes = 6

        # print('model: ', model)

        if self.gradcam:
            # splitting up the conv layers, low res [GRAD-CAM]
            self.conv_layers_1 = nn.Sequential(
                *list(model.children())[:-2]
                )
                
            self.conv_layers_2 = nn.Sequential(
                list(model.children())[-2]
                )

        else:
            # cutting off the last layer
            self.conv_layers = nn.Sequential(
                *list(model.children())[:-1]
                )

        # freezing all layers of the model except the last few
        num_children = len(list(model.children()))
        # print("num_children: ", num_children)
        num_thawed_layers = config.THAWED_LAYERS # last 2 layers of resnet are not conv layers

        if num_thawed_layers > 0:
            for i, child in enumerate(model.children()):
                # freeze everything except the thawed layers
                if i < (num_children - num_thawed_layers):
                    # print("Freezing layer: ", child)
                    for param in child.parameters():
                        param.requires_grad = False
                # else:
                    # print("Not freezing layer: ", child)
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(config.DROPOUT),

            # # adding layer for nonlinearities in robot state
            # nn.Linear(self.num_visual_features + self.num_extra_features, 512),
            nn.Linear(self.num_extra_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),

            # nn.Linear(self.num_visual_features + self.num_extra_features, 6),
            nn.Linear(512, 6),

            # nn.Linear(128, 6)
        )

    def forward(self, img, states):
        if self.gradcam:
            x = self.conv_layers_1(img)

            # register the hook for the gradients of the activations [GRAD-CAM]
            if x.requires_grad:
                h = x.register_hook(self.activations_hook)

            x = self.conv_layers_2(x)

        else:
            x = self.conv_layers(img)

        if self.num_extra_features > 0:
            # states = torch.reshape(states, (states.shape[0], states.shape[1], 1, 1)).float()
            x = torch.cat((x, states), dim=1)
            x = states
        
        model_output = self.fc_layers(x.reshape(-1, self.num_visual_features + self.num_extra_features))
        # model_output = self.fc_layers(x.reshape(-1, self.num_extra_features))

        return model_output

    # hook for gradients of activations [GRAD-CAM]
    def activations_hook(self, grad):
        self.gradients = grad

    # gradient extraction [GRAD-CAM]
    def get_activation_gradients(self):
        return self.gradients
    
    # activation extraction [GRAD-CAM]
    def get_activations(self, x):
        return self.conv_layers_1(x)