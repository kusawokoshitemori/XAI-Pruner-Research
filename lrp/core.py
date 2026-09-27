from .module import *

class Replacer:
    def __init__(self, lrp_map, module_map):
        self.lrp_map = lrp_map
        self.module_map = module_map


    def register(self, model):
        for layer_type, new_layer in self.lrp_map.items():
            for name, child in model.named_children():
                if isinstance(child, layer_type) and not(isinstance(child, new_layer)):
                    wrapped_layer = self.module_map[new_layer](child)
                    setattr(model, name, wrapped_layer)

                elif any(child.named_children()):
                    self.register(child)


INIT_MAP = {
        BasicBlock : BasicBlockLRP,
        Bottleneck : BottleneckLRP,
        nn.Conv2d : Conv2dLRP,
        nn.BatchNorm2d : BatchNorm2dLRP,
        nn.ReLU : ReLULRP,
        nn.MaxPool2d : MaxPool2dLRP,
        nn.AdaptiveAvgPool2d : AdaptiveAvgPool2dLRP,
        nn.Linear : EpsilonLinearLRP
}






