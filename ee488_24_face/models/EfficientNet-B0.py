#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as nn
from torchvision.models import efficientnet_b0

def MainModel(nOut=256, **kwargs):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, nOut) # Modify as needed
    return model
