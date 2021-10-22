from efficientnet_pytorch import EfficientNet
import torch.nn as nn

effnet = EfficientNet.from_pretrained("efficientnet-b0", advprop=True, num_classes=4)
effnet._conv_stem.in_channels = 1
weight = effnet._conv_stem.weight.mean(1, keepdim=True)
effnet._conv_stem.weight = nn.Parameter(weight)

print(effnet)
