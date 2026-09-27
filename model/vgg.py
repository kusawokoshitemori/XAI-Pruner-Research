import timm
from timm.models import VGG
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

original_cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def update_cfg(cfg, pruned_cfg=None):

    index = 0
    for i, item in enumerate(cfg):
        if isinstance(item, int):
            cfg[i] = pruned_cfg[index]
            index += 1
    return cfg

def create_VGG(model_name, num_classes, pruned=False, pruned_cfg=None):
    if not pruned:
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
        model.pre_logits = nn.Identity()
        model.head.fc = nn.Linear(512, num_classes)

    else:
        variant = model_name.split('_')[0]
        cfg = original_cfgs[variant]
        new_cfg = update_cfg(cfg, pruned_cfg=pruned_cfg)

        if 'bn' in model_name:
            model = VGG(cfg=new_cfg, num_classes=num_classes, norm_layer=nn.BatchNorm2d)
        else:
            model = VGG(cfg=new_cfg, num_classes=num_classes)


        model.pre_logits = nn.Identity()
        model.head.fc = nn.Linear(pruned_cfg[-1], num_classes)

    return model





