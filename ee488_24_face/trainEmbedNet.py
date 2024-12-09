#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import glob
import datetime
import numpy
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="Face Recognition Training")

## Data loader
parser.add_argument('--batch_size', type=int, default=100, help='Batch size, defined as the number of classes per batch')
parser.add_argument('--max_img_per_cls', type=int, default=500, help='Maximum number of images per class per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of data loader threads')

## Training details
parser.add_argument('--test_interval', type=int, default=5, help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--trainfunc', type=str, default="softmax", help='Loss function to use')

## Optimizer and Scheduler
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
parser.add_argument('--scheduler', type=str, default="cosine", help='Learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.0001, help='Lower initial learning rate for fine-tuning')
parser.add_argument("--weight_decay", type=float, default=1e-5, help='Weight decay for regularization')

## Loss functions
parser.add_argument('--margin', type=float, default=0.1, help='Loss margin')
parser.add_argument('--scale', type=float, default=30, help='Loss scale')
parser.add_argument('--nPerClass', type=int, default=1, help='Number of images per class per batch')
parser.add_argument('--nClasses', type=int, default=9500, help='Number of classes in the softmax layer')

## Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights')
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs')

## Training and evaluation data
parser.add_argument('--train_path', type=str, default="data/train", help='Absolute path to the train set')
parser.add_argument('--train_ext', type=str, default="jpg", help='Training files extension')
parser.add_argument('--test_path', type=str, default="data/val", help='Absolute path to the test set')
parser.add_argument('--test_list', type=str, default="data/val_pairs.csv", help='Evaluation list')

## Model definition
parser.add_argument('--model', type=str, default="ResNet18", help='Model to use (ResNet18 or ResNet50)')
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
parser.add_argument('--output', type=str, default="", help='Save a log of output to this file name')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')

args = parser.parse_args()

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Script to compute EER
# ## ===== ===== ===== ===== ===== ===== ===== =====

def compute_eer(all_labels, all_scores):
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    EER = fpr[numpy.nanargmin(numpy.abs(fpr - fnr))]
    return EER

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path + "/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Load model
    model = EmbedNet(**vars(args)).cuda()

    ## Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ## Data loaders
    trainLoader = get_data_loader(transform=train_transform, **vars(args))
    trainer = ModelTrainer(model, **vars(args))

    ## Resume training
    modelfiles = glob.glob(f'{args.save_path}/epoch0*.model')
    modelfiles.sort()
    ep = 1

    if modelfiles:
        trainer.loadParameters(modelfiles[-1])
        print(f"Model {modelfiles[-1]} loaded!")
        ep = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## Scheduler setup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.__optimizer__, T_max=args.max_epoch, eta_min=1e-6)

    ## Training loop
    for epoch in range(ep, args.max_epoch + 1):
        clr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch:04d} started with LR {clr:.6f}")

        loss = trainer.train_network(trainLoader)
        logger.info(f"Epoch {epoch:04d} completed with Loss {loss:.5f}")

        if epoch % args.test_interval == 0:
            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
            EER = compute_eer(lab, sc)
            logger.info(f"Epoch {epoch:04d}, Validation EER: {EER*100:.2f}%")
            trainer.saveParameters(f"{args.save_path}/epoch{epoch:04d}.model")

        scheduler.step()

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.save_path, exist_ok=True)
    main_worker(args)

if __name__ == '__main__':
    main()
