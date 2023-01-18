import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, efficientnet_b0, squeezenet1_1
from vit_pytorch import ViT
import timm
from prediction.config_utils import *

class Model(nn.Module):
    def __init__(self, gradcam=False):
        super().__init__()
        self.config, _ = parse_config_args()
        self.num_extra_features = len(self.config.ROBOT_STATES)

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.gradients = None
        self.gradcam = gradcam

        if self.config.MODEL_NAME == 'resnet18':
            self.model = resnet18(pretrained=True)
            self.num_visual_features = 512
        elif self.config.MODEL_NAME == 'resnet50':
            self.model = resnet50(pretrained=True)
            self.num_visual_features = 2048
        elif self.config.MODEL_NAME == 'efficientnetb0':
            self.model = efficientnet_b0(pretrained=True)
            self.num_visual_features = 1280
        elif self.config.MODEL_NAME == 'squeezenet':
            # len(module.children()) = 2
            self.model = squeezenet1_1(pretrained=True)
            self.num_visual_features = 1352
            # change the last conv2d layer
            self.model.classifier._modules["1"] = nn.Conv2d(512, 6, kernel_size=(1, 1))
            # change the internal num_classes variable rather than redefining the forward pass
            self.model.num_classes = 6
        elif self.config.MODEL_NAME == 'vit':
            self.model = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 6,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            # dropout = 0.1,
            # emb_dropout = 0.1,
            # pool = 'mean'
            )

            self.num_visual_features = 1024

        elif self.config.MODEL_NAME == 'vit_pretrained':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.num_visual_features = 768
            # redefining the last layer
            self.model.head = nn.Linear(self.num_visual_features, 6)


        # printing number of parameters
        # print('number of parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # print('model: ', self.model)
        # for i, layer in enumerate(self.model.children()):
        #     print('layer ', i, layer)

        # print('vit conv layers: ', *list(self.model.children())[:-1])

        if self.config.MODEL_NAME == 'vit':
            self.fc_layers = nn.Sequential(
                list(self.model.children())[-1],
                nn.AdaptiveAvgPool2d(1),
                # nn.Linear(1024*49, 6),
            )

        else:
            if self.gradcam:
                # splitting up the conv layers, low res [GRAD-CAM]
                self.conv_layers_1 = nn.Sequential(
                    *list(self.model.children())[:-2]
                    )
                    
                self.conv_layers_2 = nn.Sequential(
                    list(self.model.children())[-2]
                    )

                # # splitting up the conv layers, high res [GRAD-CAM]
                # self.conv_layers_1 = nn.Sequential(
                #     *list(self.model.children())[:-3]
                #     )
                    
                # self.conv_layers_2 = nn.Sequential(
                #     *list(self.model.children())[-3:-1]
                #     )
            else:
                # cutting off the last layer
                self.conv_layers = nn.Sequential(
                    *list(self.model.children())[:-1]
                    )

            
            

            self.fc_layers = nn.Sequential(
                nn.Dropout(self.config.DROPOUT),
                nn.Linear(self.num_visual_features + self.num_extra_features, 6),
                # nn.Dropout(self.config.DROPOUT),
                # nn.ReLU(),

                # # adding layer for nonlinearities in robot state
                # nn.Linear(128, 128),
                # nn.ReLU(),

                # nn.Linear(128, 6)
            )

        # freezing all layers of the model except the last config.THAWED_LAYERS layers
        num_children = len(list(self.model.children()))
        # print("num_children: ", num_children)
        num_thawed_layers = self.config.THAWED_LAYERS # last 2 layers of resnet are not conv layers

        if num_thawed_layers > 0:
            for i, child in enumerate(self.model.children()):
                # freeze everything except the thawed layers
                if i < (num_children - num_thawed_layers):
                    print("Freezing layer: ", child)
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    print("Not freezing layer: ", child)

    def forward(self, img, states):
        if self.config.MODEL_NAME == 'vit' or self.config.MODEL_NAME == 'vit_pretrained':
            model_output = self.model(img)
            # print('vit model output: ', model_output.shape)
        
        else:
            if self.gradcam:
                x = self.conv_layers_1(img)

                # register the hook for the gradients of the activations [GRAD-CAM]
                if x.requires_grad:
                    h = x.register_hook(self.activations_hook)

                x = self.conv_layers_2(x)

            else:
                x = self.conv_layers(img)

            if self.num_extra_features > 0:
                states = torch.reshape(states, (states.shape[0], 1, 1, 1)).float()
                x = torch.cat((x, states), dim=1)
            
            model_output = self.fc_layers(x.reshape(-1, self.num_visual_features + self.num_extra_features))

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