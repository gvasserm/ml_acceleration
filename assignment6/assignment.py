from transformers import AutoModelForSequenceClassification, AutoTokenizer
#from openvino_tokenizers import convert tokenizers
import openvino as ov
from time import perf_counter
from statistics import median
import torch
import numpy as np
from datasets import load_dataset
import evaluate

import torch
import numpy as np
from datasets import load_dataset
import evaluate


val_dataset = load_dataset("glue", "sst2", split="validation")
accuracy = evaluate.load("accuracy")
model_id = "philschmid/MiniLM-L6-H384-uncased-sst2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

@torch.no_grad
def accuracy_evaluate(model, dataset=val_dataset, accuracy=accuracy):   
    for sample in dataset:
        tokenized = {**tokenizer(sample["sentence"], return_tensors="pt")}
        logits = model(tokenized)
        pred = np.argmax(logits, axis=1)
        accuracy.add(references=sample["label"], predictions=pred)

    return accuracy.compute()


@torch.no_grad
def benchmark(model, dataset, num_warmup=10):
    tokenized_dataset = [{**tokenizer(sample["sentence"], return_tensors="pt")} for sample in dataset]

    # add warmup step
    for i in range(num_warmup):
        warmup_data = tokenized_dataset[i % len(tokenized_dataset)]
        _ = model(warmup_data)
    
    times = []
    for data in tokenized_dataset:
        start = perf_counter()
        model(data)
        end = perf_counter()
        times.append(end - start)

    return (
        f"{sum(times):.5f}s, FPS={(len(dataset) / sum(times)):.3f}, "
        f"latency: {min(times):.5f}s, {median(times):.5f}s, {max(times):.5f}s"
    )

from openvino.runtime import Core
def test():
    #hf_model = AutoModelForSequenceClassification.from_pretrained("/home/gvasserm/dev/ml_acceleration/assignment6/base_model")
    #hf_tokenizer = AutoTokenizer.from_pretrained("/home/gvasserm/dev/ml_acceleration/assignment6/tokenizer")

    
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    #hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    ie = Core()
    ov_model = ie.read_model(model="model_xml", weights="model_bin")
    
    text_input = ["I love accelerating networks"]
    hf_input = tokenizer(text_input, return_tensors="pt")
    #ov_tokenizer = convert_tokenizer(hf_tokenizer)
    ov_model = ov.convert_model(ov_model, example_input=hf_input.data)
    ov.save_model(ov_model, "model.xml")

    compiled_model = ov.compile_model(ov_model)
    print(compiled_model(hf_input.data))

    #val_dataset = load_dataset("glue", "sst2", split="validation")

    #print(f"PyTorch:  {accuracy_evaluate(lambda x: hf_model(**x).logits)}")
    #print(f"OpenVINO: {accuracy_evaluate(lambda x: compiled_model(x)[compiled_model.output()])}")
    #print("Pytorch:  ", benchmark(lambda x: hf_model(**x), tokenizer, val_dataset))
    #print("Openvino: ", benchmark(lambda x: compiled_model(x), tokenizer, val_dataset))
    
    print("Pytorch:  ", benchmark(lambda x: hf_model(**x), val_dataset))
    print("Openvino: ", benchmark(lambda x: compiled_model(x), val_dataset))
    return


def transform_fn(sample):
    return {**tokenizer(sample["sentence"], return_tensors="pt")}

import nncf
from typing import Iterable, Any

def validate(model, dataset):
    accuracy = evaluate.load("accuracy") 
    for sample in dataset:
        tokenized = {**tokenizer(sample["sentence"], return_tensors="pt")}
        logits = model(tokenized)[model.output()]
        pred = np.argmax(logits, axis=1)
        accuracy.add(references=sample["label"], predictions=pred)

    return accuracy.compute()['accuracy']
                             

def test2():

    calibration_dataset = load_dataset("glue", "sst2", split="train[:10%]")
    calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)
    validation_dataset = nncf.Dataset(val_dataset, transform_fn)

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    text_input = ["I love accelerating networks"]
    hf_input = tokenizer(text_input, return_tensors="pt")
    
    #ie = Core()
    #ov_model = ie.read_model(model="model_xml", weights="model_bin")
    ov_model = ov.convert_model(hf_model, example_input=hf_input.data)
    #quantized_model = nncf.quantize_with_accuracy_control(ov_model, calibration_dataset=calibration_dataset, validation_dataset=validation_dataset, validation_fn=validate)
    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    compiled_model = ov.compile_model(quantized_model)
    print(f"OpenVINO: {accuracy_evaluate(lambda x: compiled_model(x)[compiled_model.output()])}")
    ov.save_model(quantized_model, "qbert.xml")
    return

def test3():

    calibration_dataset = load_dataset("glue", "sst2", split="train[:10%]")
    calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)
    validation_dataset = nncf.Dataset(val_dataset, transform_fn)

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    text_input = ["I love accelerating networks"]
    hf_input = tokenizer(text_input, return_tensors="pt")
    
    #ie = Core()
    #ov_model = ie.read_model(model="model_xml", weights="model_bin")
    ov_model = ov.convert_model(hf_model, example_input=hf_input.data)
    quantized_model = nncf.quantize_with_accuracy_control(ov_model, 
                                                          calibration_dataset=calibration_dataset, 
                                                          validation_dataset=validation_dataset, 
                                                          validation_fn=validate,
                                                          )
    #quantized_model = nncf.quantize(ov_model, calibration_dataset)
    compiled_model = ov.compile_model(quantized_model)
    print(f"OpenVINO: {accuracy_evaluate(lambda x: compiled_model(x)[compiled_model.output()])}")
    ov.save_model(quantized_model, "qbert.xml")
    return


test3()
