import os

import typing as tp
import torch

from copy import deepcopy
from datasets import load_metric
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

# utils у нас появились при скачивании вспомогательного кода. При желании можно в них провалиться-поизучать
from utils.data import init_dataloaders
from utils.model import evaluate_model
from utils.model import init_model_with_pretrain, init_model_student_with_pretrain

from torch import nn
from transformers.models.segformer.modeling_segformer import SegformerLayer
from transformers import SegformerForSemanticSegmentation



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

def n_params(model):
    return sum(p.numel() for p in model.parameters())


from dataclasses import dataclass

@dataclass
class TrainParams:
    n_epochs: int
    lr: float
    batch_size: int
    n_workers: int
    device: torch.device

    loss_weight: float
    last_layer_loss_weight: float
    intermediate_attn_layers_weights: tp.Tuple[float, float, float, float]
    intermediate_feat_layers_weights: tp.Tuple[float, float, float, float]


def calc_last_layer_loss(student_logits, teacher_logits, weight, temperature=1.0):
    
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kl_divergence_loss = nn.KLDivLoss(reduction='batchmean')
    loss = kl_divergence_loss(student_probs, teacher_probs) * (temperature ** 2) * weight

    return loss
    
# здесь пока не обращаем внимания, чуть позже её напишем
def calc_intermediate_layers_attn_loss(student_logits, teacher_logits, weights, student_teacher_attention_mapping):
    total_loss = 0.0
    for student_idx, teacher_idx in student_teacher_attention_mapping.items():
        student_att = student_logits[student_idx]
        teacher_att = teacher_logits[teacher_idx]
        
        loss = F.mse_loss(student_att, teacher_att)
        
        # Apply weight
        weighted_loss = loss * weights[student_idx]
        
        # Accumulate loss
        total_loss += weighted_loss
    
    return total_loss

# здесь пока не обращаем внимания, чуть позже её напишем
def calc_intermediate_layers_feat_loss(student_feat, teacher_feat, weights):
    total_loss = 0.0
    assert len(student_feat) == len(teacher_feat) == len(weights), "Mismatch in the number of layers or weights"
    
    for student_feat, teacher_feat, weight in zip(student_feat, teacher_feat, weights):
        loss = F.mse_loss(student_feat, teacher_feat)
        total_loss += loss * weight
    
    return total_loss



def train(
    teacher_model,
    student_model,
    train_params: TrainParams,
    student_teacher_attention_mapping
):
    metric = load_metric('mean_iou')
    teacher_model.to(train_params.device)
    student_model.to(train_params.device)

    teacher_model.eval()

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=train_params.batch_size,
        num_workers=train_params.n_workers,
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=train_params.lr)
    step = 0
    for epoch in range(train_params.n_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for idx, batch in pbar:
            student_model.train()
            # get the inputs;
            pixel_values = batch['pixel_values'].to(train_params.device)
            labels = batch['labels'].to(train_params.device)

            optimizer.zero_grad()

            # forward + backward + optimize
            student_outputs = student_model(
                pixel_values=pixel_values, 
                labels=labels, 
                output_attentions=True,
                output_hidden_states=True,
            )
            loss, student_logits = student_outputs.loss, student_outputs.logits

            # Чего это мы no_grad() при тренировке поставили?!
            with torch.no_grad():
                teacher_output = teacher_model(
                    pixel_values=pixel_values, 
                    labels=labels, 
                    output_attentions=True,
                    output_hidden_states=True,
                )


            last_layer_loss = calc_last_layer_loss(
                student_logits,
                teacher_output.logits,
                train_params.last_layer_loss_weight,
                temperature=3.
            )

            student_attentions, teacher_attentions = student_outputs.attentions, teacher_output.attentions
            student_hidden_states, teacher_hidden_states = student_outputs.hidden_states, teacher_output.hidden_states

            intermediate_layer_att_loss = calc_intermediate_layers_attn_loss(
                student_attentions,
                teacher_attentions,
                train_params.intermediate_attn_layers_weights,
                student_teacher_attention_mapping,
            )
            
            intermediate_layer_feat_loss = calc_intermediate_layers_feat_loss(
                student_hidden_states,
                teacher_hidden_states,
                train_params.intermediate_feat_layers_weights,
            )

            total_loss = loss* train_params.loss_weight + last_layer_loss
            if intermediate_layer_att_loss is not None:
                total_loss += intermediate_layer_att_loss
            
            if intermediate_layer_feat_loss is not None:
                total_loss += intermediate_layer_feat_loss

            step += 1

            total_loss.backward()
            optimizer.step()
            pbar.set_description(f'total loss: {total_loss.item():.3f}')

            for loss_value, loss_name in (
                (loss, 'loss'),
                (total_loss, 'total_loss'),
                (last_layer_loss, 'last_layer_loss'),
                (intermediate_layer_att_loss, 'intermediate_layer_att_loss'),
                (intermediate_layer_feat_loss, 'intermediate_layer_feat_loss'),
            ):
                if loss_value is None: # для выключенной дистилляции атеншенов
                    continue
                tb_writer.add_scalar(
                    tag=loss_name,
                    scalar_value=loss_value.item(),
                    global_step=step,
                )

        #после модификаций модели обязательно сохраняйте ее целиком, чтобы подгрузить ее в случае чего
        torch.save(
            {
                'model': student_model,
                'state_dict': student_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            },
            f'{save_dir}/ckpt_{epoch}.pth',
        )

        eval_metrics = evaluate_model(student_model, valid_dataloader, id2label)

        for metric_key, metric_value in eval_metrics.items():
            if not isinstance(metric_value, float):
                continue
            tb_writer.add_scalar(
                tag=f'eval_{metric_key}',
                scalar_value=metric_value,
                global_step=epoch,
            )

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
    )


run()