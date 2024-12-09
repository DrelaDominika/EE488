#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
from torchvision.models import inception_v3

def MainModel(nOut=256, **kwargs):
    # Load the Inception_v3 model without pretrained weights
    model = inception_v3(pretrained=False, aux_logits=False)

    # Replace the final fully connected (fc) layer with a new one having nOut output features
    model.fc = nn.Linear(model.fc.in_features, nOut)

    return model
