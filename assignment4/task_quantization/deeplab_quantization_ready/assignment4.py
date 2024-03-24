import datetime
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.utils.data
from torch import nn
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization.quantize_fx import fuse_fx
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large
from tqdm import tqdm

import utils
from quantization_utils.fake_quantization import fake_quantization
from quantization_utils.static_quantization import quantize_static
from train import evaluate
from train import get_dataset
from train import train_one_epoch



def test1():

    # Вытащил дефолтные аргументы, чтобы не упражняться с argparse в ноутбуке
    with Path('assignment4/task_quantization/deeplab_quantization_ready/torch_default_args.pickle').open('rb') as file:
        args = pickle.load(file)

    # Подобирайте под ваше железо
    args.data_path = '/home/d.chudakov/datasets/coco/'
    args.epochs = 1
    args.batch_size = 32
    args.workers = 8

    print(args)

    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args, is_train=False)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.cuda()
    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    print(confmat)


    return

test1()