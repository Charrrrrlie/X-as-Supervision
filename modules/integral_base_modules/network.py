from modules.integral_base_modules.resnet import resnet_spec, ResNetBackbone
from modules.integral_base_modules.deconv_head import DeconvHead

from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torchvision.models as models

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def get_pose_net(cfg, num_joints):
    block_type, layers, channels, name = resnet_spec[cfg.num_layers]
    backbone_net = ResNetBackbone(block_type, layers, cfg.input_channel)
    head_net = DeconvHead(channels[-1], cfg.num_deconv_layers, cfg.num_deconv_filters, cfg.num_deconv_kernel,
                          cfg.final_conv_kernel, num_joints, cfg.depth_dim)
    pose_net = ResPoseNet(backbone_net, head_net)

    pose_net = init_pose_net(pose_net, cfg)

    return pose_net

def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 50
    config.num_deconv_layers = 3
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config

def init_pose_net(pose_net, cfg):
    block_type, layers, channels, name = resnet_spec[cfg.num_layers]
    if cfg.from_model_zoo:
        pretrained_net = eval('models.{}(weights=\'ResNet{}_Weights.DEFAULT\')'.\
                              format(name, cfg.num_layers)).state_dict()
        pretrained_net.pop('fc.weight', None)
        pretrained_net.pop('fc.bias', None)
        pose_net.backbone.load_state_dict(pretrained_net)
    return pose_net