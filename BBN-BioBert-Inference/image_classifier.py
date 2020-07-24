import torch
import torch.nn as nn
from bbn_resnet import bbn_res50, bbn_res34, bbn_ress50, bbn_ress101
from modules import GAP, Identity, FCNorm


class Network(nn.Module):
    def __init__(self, backbone_type='bbn_res34', num_classes=331):
        super(Network, self).__init__()
        if backbone_type == 'bbn_res34':
            self.backbone = bbn_res34(pretrain=False, pretrained_model="")
            num_features = 512 * 2
        elif backbone_type == 'bbn_res50':
            self.backbone = bbn_res50(pretrain=False, pretrained_model="")
            num_features = 2048 * 2
        elif backbone_type == 'bbn_ress50':
            self.backbone = bbn_ress50(pretrain=False, pretrained_model="")
            num_features = 2048 * 2
        elif backbone_type == 'bbn_ress101':
            self.backbone = bbn_ress101(pretrain=False, pretrained_model="")
            num_features = 2048 * 2
        else:
            raise NotImplementedError

        self.module = GAP()
        self.classifier = nn.Linear(num_features, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.module(x)
        x1 = x.view(x.shape[0], -1)
        x = self.classifier(x1)
        return x, x1

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")
