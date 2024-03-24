import sys
from pathlib import Path

# Calculate the absolute path to the parent directory (two levels up)
parent_dir = Path(__file__).resolve().parent.parent.parent

# Convert the Path object to a string and add it to sys.path
sys.path.append(str(parent_dir))

import os

import typing as tp
import torch
import numpy as np
from copy import deepcopy
from datasets import load_metric
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from torch import nn
from transformers.models.segformer.modeling_segformer import SegformerLayer

from utils.data import init_dataloaders
from utils.model import evaluate_model
from utils.model import init_model_with_pretrain, init_model_student_with_pretrain

from utils.train import TrainParams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# маппинг названия классов и индексов
id2label = {
    0: "background",
    1: "human",
}
label2id = {v: k for k, v in id2label.items()}

mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()

save_dir = 'assignment3/runs/svd'

tb_writer = SummaryWriter(save_dir)

def truncated_svd(W, l, transpose=False):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """
    # посчитаем SVD
    U, s, V = torch.linalg.svd(W, full_matrices=False)  # Compute the SVD
    #U, s, V = torch.svd(W, some=True)
    #

    Ul = U[:, :l]
    sl = s[:l]  # Keep the first l singular values
    V = V.t()
    Vl = V[:l, :]
    
    # Обьеденим  Sigma_l and V_l
    SV = torch.diag(sl) @ Vl
    
    if transpose: # Транспонируем
        Ul, SV = Ul.T, SV.T
    return Ul, SV


class TruncatedSVDLayer(nn.Module):
    def __init__(self, replaced_gemm, rank=None, preserve_ratio=0.9, device='cpu', transpose=False):
        super().__init__()
        self.replaced_gemm = replaced_gemm
        self.W = self.replaced_gemm.weight
        self.b = self.replaced_gemm.bias
        self.transpose = transpose

        W = self.W.data

        print("W = {}".format(W.shape))
        if rank is None:
            rank = int(preserve_ratio * W.size(0))
        
        # считаем U and SV
        self.U, self.SV = truncated_svd(W, rank, transpose = self.transpose)
        print("U = {}".format(self.U.shape))
        # Cоздаем слой иницализорованный U - нужного размера
        self.fc_u = nn.Linear(self.U.size(1), self.U.size(0)).to(device)
        self.fc_u.weight.data = self.U

        print("SV = {}".format(self.SV.shape))
        # Cоздаем слой иницализорованный SV - нужного размера
        self.fc_sv = nn.Linear(self.SV.size(1), self.SV.size(0), bias=False).to(device)
        self.fc_sv.weight.data = self.SV
        # забываем старый слой
        #self.W = ...
        #self.replaced_gemm = ...

        del self.W, self.replaced_gemm

    def forward(self, x):
        # применим SV к x
        x = self.fc_sv(x)
        # применим U
        x = self.fc_u(x)
        # не забудим про bias
        if self.b is not None:
            x += self.b.view(1, -1).expand_as(x)
        return x

def create_small_network(
    model,
    decode_head_layer_id=-1,
    preserve_ratio=0.3,
    device='cuda',
    transpose=False
):
    """Выбрали слой,  сжали его и прозвели замену"""
    proj = model.decode_head.linear_c[decode_head_layer_id].proj
    compressed = TruncatedSVDLayer(proj, 
                                   rank=None, 
                                   preserve_ratio=preserve_ratio,
                                   device=device, 
                                   transpose=transpose)
    
    model.decode_head.linear_c[decode_head_layer_id].proj = compressed
    return model


def train(
    teacher_model,
    student_model,
    train_params: TrainParams,
    student_teacher_attention_mapping,
):
    metric = load_metric('mean_iou')
    teacher_model.to(train_params.device)
    student_model.to(train_params.device)

    teacher_model.eval()

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset",
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



# вспомогаетальные функции - что они делают понятно из названия?
def get_n_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def disable_old_layers_grads(model, decode_head_layer_id = -1):
    model.decode_head.linear_c[decode_head_layer_id].requires_grad = False

def enable_all_grads(model):
    for name,param in model.named_parameters():
        param.requires_grad = True

def calc_last_layer_loss(student_logits, teacher_logits, weight):
    return mse_loss(student_logits, teacher_logits) * weight

def calc_intermediate_layers_attn_loss(student_attentions, teacher_attentions, weights, student_teacher_attention_mapping):
    intermediate_kl_loss = 0
    for i, (stud_attn_idx, teach_attn_idx) in enumerate(student_teacher_attention_mapping.items()):
        intermediate_kl_loss += weights[i] * kl_loss(
            input=torch.log(student_attentions[stud_attn_idx]),
            target=teacher_attentions[teach_attn_idx],
        )
    return intermediate_kl_loss

def calc_intermediate_layers_feat_loss(student_feats, teacher_feats, weights):
    intermediate_mse_loss = 0.
    for i in range(len(student_feats)):
        intermediate_mse_loss += weights[i] * mse_loss(
            input=student_feats[i],
            target=teacher_feats[i],
        )
    return intermediate_mse_loss

def test_svd():
    
    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=24,
        num_workers=8)
   
    model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path='assignment3/runs/ckpt_4.pth')
    full_metrics = evaluate_model(model, valid_dataloader, id2label)
    decomposed_model = create_small_network(deepcopy(model), decode_head_layer_id=-4, preserve_ratio=0.1)
    
    decomposed_metrics = evaluate_model(decomposed_model, valid_dataloader, id2label)

    # дававйте проверим iou drop и как сократилось число параметров
    d = full_metrics['mean_iou']-decomposed_metrics['mean_iou']
    print(f'Metric Diff: {d}')
    print(f'Model size compression ratio: {get_n_params(decomposed_model)/get_n_params(model)}')

    decomposed_model = deepcopy(model) # возьмем не сжатую модель
    for i in [-1,-2,-3,-4]: #
        # cжимаем все слои сразу
        decomposed_model = create_small_network(deepcopy(decomposed_model),decode_head_layer_id=i,transpose=False, preserve_ratio=0.1)

    print(decomposed_model.decode_head)

    decomposed_metrics = evaluate_model(decomposed_model, valid_dataloader, id2label)
    d = full_metrics['mean_iou']-decomposed_metrics['mean_iou']
    # дававйте проверим iou drop и как сократилось число параметров
    print(f'Metric Diff: {d}')
    print(f'Model size compression ratio: {get_n_params(decomposed_model)/get_n_params(model)}')

    train_params = TrainParams(
        n_epochs=1,
        lr=18e-5,
        batch_size=24,
        n_workers=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=3,
        loss_weight=1,
        last_layer_loss_weight=0.,
        intermediate_attn_layers_weights=(0, 0, 0, 1.),
        intermediate_feat_layers_weights=(0, 0, 0, 1.),)
    
    student_teacher_attention_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

    train(
        teacher_model=model,
        student_model=decomposed_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping
    )
    return

def freeze_all_layers(model):

    for idx, (name, child) in enumerate(model.named_children()):
        for param in child.parameters():
            param.requires_grad = False

def unfreeze_all_layers(model):
    
    for idx, (name, child) in enumerate(model.named_children()):
        for param in child.parameters():
            param.requires_grad = True


def test_freeze():

    model = init_model_with_pretrain(
    label2id=label2id,
    id2label=id2label,
    pretrain_path='assignment3/runs/ckpt_4.pth')
    
    num_mlp_in_head = len(model.decode_head.linear_c) # посчитаем число proj слоев в голове-декодере
    layer_ids = -(np.arange(num_mlp_in_head)+1) # возьмем их в обратном порядке - можете задать id в ручную обычным list

    print(f'Number proj in head: {layer_ids}')

    #student from distill
    decomposed_model = deepcopy(model) # возьмем не сжатую модель
    iou_drops = [] # для сохранения результата =
    compress_ratios = []

    train_params = TrainParams(
        n_epochs=1,
        lr=18e-5,
        batch_size=24,
        n_workers=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=3,
        loss_weight=1,
        last_layer_loss_weight=0.,
        intermediate_attn_layers_weights=(0, 0, 0, 1.),
        intermediate_feat_layers_weights=(0, 0, 0, 1.),)
    
    student_teacher_attention_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=24,
        num_workers=8)
    
    full_metrics = evaluate_model(model, valid_dataloader, id2label)
    
    for i in layer_ids: #
        
        # cледуем шагам из задания 2
        decomposed_model = create_small_network(decomposed_model, decode_head_layer_id=i,
            preserve_ratio=0.3,
            device='cuda',
            transpose=False)
        
        freeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)

        train(teacher_model=model,
            student_model=decomposed_model,
            train_params=train_params,
            student_teacher_attention_mapping=student_teacher_attention_mapping)

        unfreeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)

        train(teacher_model=model,
            student_model=decomposed_model,
            train_params=train_params,
            student_teacher_attention_mapping=student_teacher_attention_mapping)
        
        compress_ratios.append(get_n_params(decomposed_model)/get_n_params(model))
        decomposed_metrics = evaluate_model(decomposed_model, valid_dataloader, id2label)
        diff = full_metrics['mean_iou']-decomposed_metrics['mean_iou']
        iou_drops.append(diff)

    print(decomposed_model.decode_head.linear_c) # давайте напечатаем слои
    decomposed_metrics = evaluate_model(decomposed_model, valid_dataloader, id2label) # финальная аккураси
    
    d = full_metrics['mean_iou']-decomposed_metrics['mean_iou']
    # дававйте проверим iou drop и как сократилось число параметров
    print(f'Metric Diff: {d}')
    print(f'Model size compression ratio: {get_n_params(decomposed_model)/get_n_params(model)}')

    print(iou_drops)
    print(compress_ratios)


def test_all():

    model = init_model_with_pretrain(
    label2id=label2id,
    id2label=id2label,
    pretrain_path='assignment3/runs/ckpt_4.pth')
    
    num_mlp_in_head = len(model.decode_head.linear_c) # посчитаем число proj слоев в голове-декодере
    layer_ids = -(np.arange(num_mlp_in_head)+1) # возьмем их в обратном порядке - можете задать id в ручную обычным list

    print(f'Number proj in head: {layer_ids}')

    #student from distill
    decomposed_model = deepcopy(model) # возьмем не сжатую модель
    iou_drops = [] # для сохранения результата =
    compress_ratios = []

    train_params = TrainParams(
        n_epochs=1,
        lr=18e-5,
        batch_size=24,
        n_workers=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=3,
        loss_weight=1,
        last_layer_loss_weight=0.,
        intermediate_attn_layers_weights=(0, 0, 0, 1.),
        intermediate_feat_layers_weights=(0, 0, 0, 1.),)
    
    student_teacher_attention_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=24,
        num_workers=8)
    
    full_metrics = evaluate_model(model, valid_dataloader, id2label)

    decomposed_model = deepcopy(model) # возьмем не сжатую модель
    for i in layer_ids: #
        # cжимаем все слои сразу
        decomposed_model = create_small_network(decomposed_model, decode_head_layer_id=i,
            preserve_ratio=0.3,
            device='cuda',
            transpose=False)
        
        freeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)


    train(teacher_model=model,
        student_model=decomposed_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping)
    
    
    for i in layer_ids:
        unfreeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)

    train(teacher_model=model,
        student_model=decomposed_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping)


    decomposed_metrics = evaluate_model(decomposed_model, valid_dataloader, id2label)
    iou_diff_shot = full_metrics['mean_iou']-decomposed_metrics['mean_iou']

    print(iou_diff_shot)
    print(get_n_params(decomposed_model) / get_n_params(model))

    return

def test_rank():
    
    model = init_model_with_pretrain(
    label2id=label2id,
    id2label=id2label,
    pretrain_path='assignment3/runs/ckpt_4.pth')
    
    num_mlp_in_head = len(model.decode_head.linear_c) # посчитаем число proj слоев в голове-декодере
    layer_ids = -(np.arange(num_mlp_in_head)+1) # возьмем их в обратном порядке - можете задать id в ручную обычным list

    print(f'Number proj in head: {layer_ids}')

    #student from distill
    decomposed_model = deepcopy(model) # возьмем не сжатую модель
    iou_drops = [] # для сохранения результата =
    compress_ratios = []

    train_params = TrainParams(
        n_epochs=1,
        lr=18e-5,
        batch_size=24,
        n_workers=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=3,
        loss_weight=1,
        last_layer_loss_weight=0.,
        intermediate_attn_layers_weights=(0, 0, 0, 1.),
        intermediate_feat_layers_weights=(0, 0, 0, 1.),)
    
    student_teacher_attention_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=24,
        num_workers=8)
    
    full_metrics = evaluate_model(model, valid_dataloader, id2label)

    ranks_ratio = [0.7, 0.5, 0.3, 0.1]

    decomposed_metrics = []
    iou_diff_shot = []
    params_ratio = []

    for ratio in ranks_ratio: 

        decomposed_model = deepcopy(model) # возьмем не сжатую модель

        for i in layer_ids: #
            # cжимаем все слои сразу
            decomposed_model = create_small_network(decomposed_model, decode_head_layer_id=i,
                preserve_ratio=ratio,
                device='cuda',
                transpose=False)
            
            freeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)


        train(teacher_model=model,
            student_model=decomposed_model,
            train_params=train_params,
            student_teacher_attention_mapping=student_teacher_attention_mapping)
        
        
        for i in layer_ids:
            unfreeze_all_layers(decomposed_model.decode_head.linear_c[i].proj)

        train(teacher_model=model,
            student_model=decomposed_model,
            train_params=train_params,
            student_teacher_attention_mapping=student_teacher_attention_mapping)


        decomposed_metrics.append(evaluate_model(decomposed_model, valid_dataloader, id2label))
        iou_diff_shot.append(full_metrics['mean_iou']-decomposed_metrics[-1]['mean_iou'])
        params_ratio.append(get_n_params(decomposed_model) / get_n_params(model))

    print(decomposed_metrics)
    print(iou_diff_shot)
    print(params_ratio)

    return


test_svd()
#test_freeze()
#test_all()
#test_rank()