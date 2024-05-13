

import torch
import tensorflow as tf
from pathlib import Path
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torch.nn as nn
import torch.functional as F

import glob 
import cv2
import numpy as np 

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def representative_dataset():
    paths = glob.glob("/home/gvasserm/data/coco2017/train2017/*.jpg")[:200]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (300, 300))
        img = img[:, :, ::-1]  # Convert BGR to RGB
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = (img / 255.0 - mean) / std
        yield [img]  # Ensure this is a list of numpy arrays

def representative_dataset():
  paths = glob.glob("/home/gvasserm/data/coco2017/train2017/*.jpg")[:10]
  for p in paths:
    img = cv2.resize(cv2.imread(p), (300,300))[np.newaxis,:,:,::-1].astype(np.float32)
    yield [((img/255.0 - mean)/std).astype(np.float32)]


def test1():

    converter = tf.lite.TFLiteConverter.from_saved_model("/home/gvasserm/dev/ml_acceleration/assignment9/dv3_mnv3")
    #converter.target_spec.supported_types = [tf.int8]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_model_quant = converter.convert()
    Path("dv3_mnv3_int8_dynamic.tflite").write_bytes(tflite_model_quant)

    
class ReLU6_Sigmoid(nn.ReLU6):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.ReLU6().forward(input + 3.)* 0.16666667
    
class ReLU6_Swish(ReLU6_Sigmoid):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * ReLU6_Sigmoid().forward(input)


def convert_hardswish_to_relu6(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(model, child_name, ReLU6_Swish())
        else:
            convert_hardswish_to_relu6(child)

def convert_hardsigmoid_to_relu6(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(model, child_name, ReLU6_Sigmoid())
        else:
            convert_hardsigmoid_to_relu6(child)
    
def test():
   
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    convert_hardsigmoid_to_relu6(model)
    convert_hardswish_to_relu6(model)
    
    model.eval()

    #сохраняем и конвертируем модельку
    input_tensor = torch.rand((1, 3, 300, 300))
    with torch.no_grad():
        output_fixed = model(input_tensor)['out']

    torch.onnx.export(model, input_tensor, f=f"dv3_mnv3_fixed.onnx", export_params=True, input_names=['input'], 
                    do_constant_folding=True, opset_version=13, output_names=['output'])
    
    return

    

test()