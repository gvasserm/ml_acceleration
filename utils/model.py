from typing import Dict

import torch
from datasets import load_metric
from copy import deepcopy
from torch import nn
from transformers import SegformerForSemanticSegmentation
import torch_pruning as tp


def init_model_student_with_pretrain(pretrain_path):
    
    devices = torch.device("cuda:0")
    params = torch.load(pretrain_path, devices)
    model = SegformerForSemanticSegmentation(params['model'].config)
    model.load_state_dict(params['state_dict'], strict=False)

    return model.cuda()

# def init_model_prunned(pruned_ckpt):
#     devices = torch.device("cuda:0")
#     params = torch.load(pruned_ckpt)
#     model = SegformerForSemanticSegmentation(params['model'].config)
#     tp.load_state_dict(model, state_dict=loaded_state_dict)


def init_model_with_pretrain_(id2label: Dict, label2id: Dict, pretrain_path: str = None):
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    if pretrain_path:
        devices = torch.device("cuda:0")
        state_dict = torch.load(pretrain_path, devices)['state_dict']
        model.load_state_dict(state_dict)
        #model = torch.load(pretrain_path)["model"]
    return model.cuda()


def init_model_with_pretrain(id2label: Dict, label2id: Dict, pretrain_path: str = None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    if pretrain_path:
        model = torch.load(pretrain_path)["model"]

    return model


def evaluate_model(model, valid_dataloader, id2label):
    metric = load_metric("mean_iou")
    predictions = []
    references = []

    with torch.no_grad():
        model.eval()
        for batch in valid_dataloader:
            pixel_values = batch["pixel_values"].cuda()
            labels = batch["labels"].cuda()
            logits = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(
                logits.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)
            predictions.append(predicted.detach().cpu().numpy())
            references.append(labels.detach().cpu().numpy())

        # note that the metric expects predictions + labels as numpy arrays

    metrics = metric._compute(
        predictions=predictions,
        references=references,
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=False,  # we've already reduced the labels before)
    )

    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])

    return metrics
