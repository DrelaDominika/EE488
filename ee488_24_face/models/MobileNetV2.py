#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
from torchvision.models import mobilenet_v2

def MainModel(nOut=256, **kwargs):
    # Load the MobileNet_v2 model without pretrained weights
    model = mobilenet_v2(pretrained=False)

    # Replace the final classification layer with a new one having nOut output features
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, nOut)

    return model
