import sys
from pathlib import Path

# Calculate the absolute path to the parent directory (two levels up)
parent_dir = Path(__file__).resolve().parent.parent.parent

# Convert the Path object to a string and add it to sys.path
sys.path.append(str(parent_dir))

import os

import torch

from copy import deepcopy
from datasets import load_metric
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

# utils у нас появились при скачивании вспомогательного кода. При желании можно в них провалиться-поизучать
from utils.data import init_dataloaders, n_params
from utils.model import evaluate_model
from utils.model import init_model_with_pretrain, init_model_student_with_pretrain

from torch import nn
from transformers.models.segformer.modeling_segformer import SegformerLayer
from transformers import SegformerForSemanticSegmentation
from utils.train import train, TrainParams


teacher_path = 'assignment1/runs/baseline_ckpt.pth'
save_dir = 'assignment1/runs/distillation'

tb_writer = SummaryWriter(save_dir)

# маппинг названия классов и индексов
id2label = {
    0: "background",
    1: "human",
}
label2id = {v: k for k, v in id2label.items()}


def create_small_network(model):
    """ Оставляет только по одному SegformerLayer в каждом ModuleList"""
    ...

    #old_list = model.segformer.encoder.modules
    #new_list = 

    model.config.depths = [1,1,1,1]
    small_model = SegformerForSemanticSegmentation(model.config)

    devices = torch.device("cuda:0")
    state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'segformer.encoder.block' in k:
            if k.split('.')[4] == '0':
                continue
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    small_model.load_state_dict(new_state_dict, strict=False)
 
    return small_model.cuda()



def run():

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=8,
        num_workers=8,
    )

    teacher_model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path=teacher_path) 
    student_model = create_small_network(deepcopy(teacher_model))

    student_model = init_model_student_with_pretrain("assignment1/runs/distillation/ckpt_4.pth")

    #evaluate_model(teacher_model, valid_dataloader, id2label)
    #evaluate_model(student_model, valid_dataloader, id2label)

    print(f'Teacher model size: {n_params(teacher_model)}')
    print(f'Student model size: {n_params(student_model)}')

    train_params = TrainParams(
        n_epochs=3,
        lr=18e-5,
        batch_size=24,
        n_workers=8,
        device=torch.device('cuda'),
        loss_weight=0.5,
        last_layer_loss_weight=0.5,
        intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
        intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
    )

    if False:
        train(
            teacher_model=teacher_model,
            student_model=deepcopy(student_model),
            train_params=train_params,
            student_teacher_attention_mapping={}, # заполним потом
        )

    with torch.no_grad():
        teacher_attentions = teacher_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions
        student_attentions = student_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions
    
    teacher_attentions[0].shape

    assert len(teacher_attentions) == 8
    assert len(student_attentions) == 4

    student_teacher_attention_mapping = {i: i*2 + 1 for i in range(4)}
    

    train(
        teacher_model=teacher_model,
        student_model=deepcopy(student_model),
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping,
        tb_writer=tb_writer,
        save_dir=save_dir, 
        id2label=id2label
    )

run()