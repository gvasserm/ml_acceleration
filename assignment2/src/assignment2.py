import sys
from pathlib import Path

# Calculate the absolute path to the parent directory (two levels up)
parent_dir = Path(__file__).resolve().parent.parent.parent

# Convert the Path object to a string and add it to sys.path
sys.path.append(str(parent_dir))

import torch
import gc

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

from utils.train import train, TrainParams

from torch import nn
from transformers.models.segformer.modeling_segformer import SegformerLayer, SegformerEfficientSelfAttention, SegformerAttention

import torch_pruning as tp

def validate_state_dicts(model_state_dict_1, model_state_dict_2):

    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
        
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if v_1.shape != v_2.shape:
            print(f"Tensor mismatch: {k_1} vs {k_2}")
        
    return True

def prune_model_l2(model):
    # вот тут надо воспользоваться библиотекой torch pruning

    example_inputs = torch.randn(1, 3, 512, 512, device="cuda")
    #L2 Magnitude prunning
    imp = tp.importance.MagnitudeImportance(p=2)

    # Ignore some layers, e.g., the output layer
    ignored_layers = [model.decode_head]

    num_heads = {}
    for m in model.modules():
        if isinstance(m, SegformerEfficientSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        imp,
        pruning_ratio = 0.75,
        pruning_ratio_dict = {},
        ignored_layers=ignored_layers,
        num_heads=num_heads
    )
    pruner.step()

    for m in model.modules():
        if isinstance(m, SegformerEfficientSelfAttention):
            print(m)
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
    return model

def prune_model_taylor(model):
    # вот тут надо воспользоваться библиотекой torch pruning
    # тут возвращается pruner, а не моделька 

    example_inputs = torch.randn(1, 3, 512, 512, device="cuda")

    # Ignore some layers, e.g., the output layer
    ignored_layers = [model.decode_head]

    #Taylor prunning
    imp = tp.importance.TaylorImportance()


    num_heads = {}
    for m in model.modules():
        if isinstance(m, SegformerEfficientSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    pruner = tp.pruner.MetaPruner(
                model, 
                example_inputs, 
                global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                pruning_ratio=0.75, # target pruning ratio
                ignored_layers=ignored_layers,
                output_transform=lambda out: out.logits.sum(),
                num_heads=num_heads)

    return pruner, num_heads

def calibrate_model(model, train_loader, device):

    model.zero_grad()
    print("Accumulating gradients for taylor pruning...")
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, batch in pbar:
        imgs = batch['pixel_values'].to(device)
        lbls = batch['labels'].to(device)
        loss = model(
                pixel_values=imgs, 
                labels=lbls
            ).loss
        
        loss.backward()
    return model

# Обратите внимание, у вас применение прунинга и его создание разнесены по функциям.
def apply_taylor_pruning(pruner):
    for g in pruner.step(interactive=True):
        g.prune()
    return None


baseline_path = "assignment2/runs/baseline_ckpt.pth"
distilled_ckpt = 'assignment2/runs/baseline_ckpt.pth'
save_dir = 'assignment2/runs/distillation'

# маппинг названия классов и индексов
id2label = {
    0: "background",
    1: "human",
}
label2id = {v: k for k, v in id2label.items()}

dataset_dir = "/home/gvasserm/data/matting_human_dataset/"
train_dataloader, valid_dataloader = init_dataloaders(
    root_dir=dataset_dir,
    batch_size=8,
    num_workers=8,
)

def test2():

    gc.collect()
    torch.cuda.empty_cache()

    baseline_model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path=baseline_path).cuda()
    pruned_model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path=baseline_path).cuda()
    pruner, num_heads = prune_model_taylor(pruned_model)

    input_example = torch.rand(1,3,512,512, device="cuda")

    train_params = TrainParams(
            n_epochs=1,
            lr=12e-5,
            batch_size=24,
            n_workers=8,
            device=torch.device('cuda'),
            loss_weight=0.5,
            last_layer_loss_weight=0.5,
            intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
            intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
        )

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir="/home/gvasserm/data/matting_human_dataset/",
        batch_size=train_params.batch_size,
        num_workers=train_params.n_workers,
    )

    pruned_model = calibrate_model(pruned_model, train_dataloader, "cuda")

    apply_taylor_pruning(pruner)

    for m in pruned_model.modules():
        if isinstance(m, SegformerEfficientSelfAttention):
            print(m)
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()

    pruned_model(input_example)

    ops, params = tp.utils.count_ops_and_params(baseline_model, input_example)
    print(f"Distilled model complexity (Baseline model complexity): {ops/1e6} MMAC, {params/1e6} M params")

    ops, params = tp.utils.count_ops_and_params(pruned_model, input_example)
    print(f"Distilled model complexity (After taylor pruning): {ops/1e6} MMAC, {params/1e6} M params")

    # Print current GPU memory usage
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print(f'Total: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')

    with torch.no_grad():
        teacher_attentions = baseline_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions
        student_attentions = pruned_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions

    teacher_attentions[0].shape

    assert len(teacher_attentions) == 8
    assert len(student_attentions) == 8

    student_teacher_attention_mapping = {2*i: 2*i for i in range(4)}


    tb_writer = SummaryWriter(save_dir)

    train(
        teacher_model=baseline_model,
        student_model=pruned_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping,
        tb_writer=tb_writer,
        save_dir=save_dir, 
        id2label=id2label
    )

    train_params = TrainParams(
            n_epochs=5,
            lr=18e-5,
            batch_size=24,
            n_workers=8,
            device=torch.device('cuda'),
            temperature=3,
            loss_weight=0.5,
            last_layer_loss_weight=0.5,
            intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
            intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
        )

    tb_writer = SummaryWriter(save_dir)
    
    train(
        teacher_model=baseline_model,
        student_model=pruned_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping,
        tb_writer=tb_writer,
        save_dir=save_dir, 
        id2label=id2label
    )

    return


def test1():

    baseline_model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path=baseline_path).cuda()
    distilled_model = init_model_with_pretrain(label2id=label2id, id2label=id2label, pretrain_path=baseline_path).cuda()

    input_example = torch.rand(1,3,512,512, device="cuda")

    if False:
        evaluate_model(baseline_model, valid_dataloader, id2label)
        evaluate_model(distilled_model, valid_dataloader, id2label)

        #distilled_model = init_model_student_with_pretrain(pretrain_path=distilled_ckpt).cuda()

        ops, params = tp.utils.count_ops_and_params(baseline_model, input_example)
        print(f"Baseline model complexity: {ops/1e6} MMAC, {params/1e6} M params")

        ops, params = tp.utils.count_ops_and_params(distilled_model, input_example)
        print(f"Distilled model complexity: {ops/1e6} MMAC, {params/1e6} M params")

        distilled_model

    pruned_model = prune_model_l2(deepcopy(distilled_model))

    #validate_state_dicts(distilled_model.state_dict(), pruned_model.state_dict())

    #diff = DeepDiff(dict(distilled_model.state_dict()), dict(pruned_model.state_dict()))
    #print(diff)

    pruned_model(input_example)

    ops, params = tp.utils.count_ops_and_params(pruned_model, input_example)
    print(f"Distilled model complexity (After magnitude pruning): {ops/1e6} MMAC, {params/1e6} M params")


    # Попробуем уменьшать модель еще сильнее, запрунив головы в attention.
    # Функционал torch pruning это не поддерживает, однако это доступно в transformers
    # Для выбора наименее полезных голов можно воспользоваться L2 нормой весов. 
    # Мы же тут выкинем все, кроме нулевой.

    pruned_model.segformer.encoder.block[1][0].attention.prune_heads([1])
    pruned_model.segformer.encoder.block[2][0].attention.prune_heads([1,2,3,4])
    pruned_model.segformer.encoder.block[3][0].attention.prune_heads([1,2,3,4,5,6,7])

    # Снова оценим вычислительную сложность
    ops, params = tp.utils.count_ops_and_params(pruned_model, input_example)
    print(f"Distilled model complexity (After magnitude pruning): {ops/1e6} MMAC, {params/1e6} M params")

    ops, params = tp.utils.count_ops_and_params(baseline_model, input_example)
    print(f"Distilled model complexity (Baseline model complexity): {ops/1e6} MMAC, {params/1e6} M params")

    train_params = TrainParams(
            n_epochs=1,
            lr=12e-5,
            batch_size=24,
            n_workers=8,
            device=torch.device('cuda'),
            loss_weight=0.5,
            temperature=3,
            last_layer_loss_weight=0.5,
            intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
            intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
        )

    with torch.no_grad():
        teacher_attentions = baseline_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions
        student_attentions = pruned_model(pixel_values=torch.ones(1, 3, 512, 512).to(train_params.device), output_attentions=True).attentions

    teacher_attentions[0].shape

    assert len(teacher_attentions) == 8
    assert len(student_attentions) == 8

    student_teacher_attention_mapping = {2*i: 2*i for i in range(4)}


    tb_writer = SummaryWriter(save_dir)

    # маппинг названия классов и индексов
    id2label = {
        0: "background",
        1: "human",
    }
    label2id = {v: k for k, v in id2label.items()}

    train(
        teacher_model=baseline_model,
        student_model=pruned_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping,
        tb_writer=tb_writer,
        save_dir=save_dir, 
        id2label=id2label
    )

    train_params = TrainParams(
            n_epochs=5,
            lr=18e-5,
            batch_size=24,
            n_workers=8,
            device=torch.device('cuda'),
            loss_weight=0.5,
            temperature=3,
            last_layer_loss_weight=0.5,
            intermediate_attn_layers_weights=(0.5, 0.5, 0.5, 0.5),
            intermediate_feat_layers_weights=(0.5, 0.5, 0.5, 0.5),
        )

    tb_writer = SummaryWriter(save_dir)
    
    train(
        teacher_model=baseline_model,
        student_model=pruned_model,
        train_params=train_params,
        student_teacher_attention_mapping=student_teacher_attention_mapping,
        tb_writer=tb_writer,
        save_dir=save_dir, 
        id2label=id2label
    )

#test1()
test2()