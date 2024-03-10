import torch 
from dataclasses import dataclass
import typing as tp
from datasets import load_metric
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from .data import init_dataloaders
from tqdm import tqdm
from .model import evaluate_model


@dataclass
class TrainParams:
    n_epochs: int
    lr: float
    batch_size: int
    n_workers: int
    device: torch.device
    temperature: float
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
    kv_loss = nn.KLDivLoss(reduction='batchmean')
    for student_idx, teacher_idx in student_teacher_attention_mapping.items():
        student_att = student_logits[student_idx]
        teacher_att = teacher_logits[teacher_idx]
        
        loss = kv_loss(student_att, teacher_att)
        
        # Apply weight
        weighted_loss = loss * weights[int(student_idx/2)]
        
        # Accumulate loss
        total_loss += weighted_loss
    
    return total_loss

# здесь пока не обращаем внимания, чуть позже её напишем
def calc_intermediate_layers_feat_loss(student_feat, teacher_feat, weights):
    return None
    
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
    student_teacher_attention_mapping, 
    tb_writer,
    save_dir, 
    id2label):

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
                temperature=train_params.temperature
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

    return