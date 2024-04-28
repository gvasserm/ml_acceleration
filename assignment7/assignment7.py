from pathlib import Path

import torch

from torch.utils.data.dataloader import DataLoader 
from torchvision import transforms
from torchvision.datasets.imagenette import Imagenette

from torchvision.models import mobilenet_v2
from torchvision.models import MobileNetV2

from polygraphy.backend.trt import CreateConfig, Profile, Calibrator
from polygraphy.comparator import DataLoader as DL
from polygraphy.backend.trt import engine_from_network
from polygraphy.backend.trt import NetworkFromOnnxPath
from polygraphy.backend.trt import save_engine
from polygraphy.backend.trt import TrtRunner, EngineFromBytes



import numpy as np
import time

CLASSES_MAPPING = {
    0: 0,
    1: 217,
    2: 848,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701,
}

         
def imagenette_val_dataloader(batch_size, height, width):
    root_dir = "/home/gvasserm/dev/ml_acceleration/imagenette/"
    
    dataset = Imagenette(
        root=root_dir, 
        split="val", 
        download=False,
        transform=transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True),
        ])
    )

    return DataLoader(dataset, batch_size=batch_size)


def validate(model, batch_size, height, width):
    val_dataloder = imagenette_val_dataloader(batch_size, height, width)

    with torch.no_grad():
        acc = []
        for images, labels in val_dataloder:
            output = model(images.cuda())
            _, predicted_labels = torch.max(output, dim=1)
            predicted_labels = predicted_labels.cpu().tolist()
            
            acc += [predicted_label == CLASSES_MAPPING[label] for predicted_label, label in zip(predicted_labels, labels.tolist())]
            
        print(f"acc = {sum(acc) * 100 / len(acc):.2f}%")

def latency_benchmark(model, test_input, warmup_n=10, benchmark_n=100):
    # model - модель для замеров
    # test_input - тестовый пример
    # warmup_n - кол-во шагов для warmup
    # benchmark_n - кол-во шагов для замера латенси

    # Warm-up phase: run the model several times to stabilize performance
    for _ in range(warmup_n):
        model(test_input)
        torch.cuda.synchronize()  # Wait for CUDA to finish

    bsz = np.float64(test_input.shape[0])

    # Benchmark phase: collect execution times
    timings = []
    for _ in range(benchmark_n):
        start_time = time.time()
        model(test_input)
        torch.cuda.synchronize()  # Ensure model has finished processing
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        timings.append(elapsed_time/bsz)

    # Calculate mean and standard deviation of the timings
    mean_ms = np.mean(timings)
    std_ms = np.std(timings)
    
    print(f"{mean_ms:.3f}ms +- {std_ms:.3f}ms")

    assert (std_ms / mean_ms) < 0.1, "слишком большое отклонение в измерения (> 10%), проверте код, возможно стоит поднять кол-во запусков"

def record_CUDA_graph(model, batch_size, height, width, warmup_n=10):
    # model - модель для записи cuda Graph
    # batch_size - размер батча входных данных
    # height - высота картинки
    # width - ширина картинки
    # warmup_n - кол-во шагов для warmup

    # Move the model to GPU and set it to evaluation mode
    model.to('cuda').eval()

    # Create a random input tensor with the specified dimensions on CUDA
    input_tensor = torch.randn(batch_size, 3, height, width, device='cuda')

    # Create a new CUDA stream for capturing the graph
    stream = torch.cuda.Stream()

    # Warm-up phase: run the model several times on the non-default stream to stabilize performance
    with torch.cuda.stream(stream):
        for _ in range(warmup_n):
            model(input_tensor)
        # Synchronize the stream to ensure all operations are completed
        torch.cuda.synchronize()

    # Instantiate a CUDA graph and start capturing on the specified stream
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        # Perform the computation you want to capture
        output_tensor = model(input_tensor)
        graph.capture_end()
        torch.cuda.synchronize()

    return graph, input_tensor, output_tensor


def test1():

    model = mobilenet_v2(weights=MobileNetV2).eval().cuda()

    bsz, ch, height, width = 1, 3, 224, 224
    validate(model, 16, 224, 224)

    if True:
        # запускаем под no_grad, чтобы минимизировать потребление памяти (исключает выделение памяти под градиенты)
        with torch.no_grad():
            latency_benchmark(
                model, 
                torch.ones(1, 3, 640, 480, device="cuda"), 
                warmup_n=10, 
                benchmark_n=100,
            )

    

    graph, input_placeholder, output_placeholder = record_CUDA_graph(model, bsz, height, width, warmup_n=10)

    test_data = torch.ones(bsz, ch, height, width, device="cuda")
    # запускаем под no_grad, чтобы минимизировать потребление памяти (исключает выделение памяти под градиенты)
    with torch.no_grad():
        # запускаем исходную модель
        model_output = model(test_data)
        
        # запускаем graph
        input_placeholder.copy_(test_data)
        graph.replay()
        graph_output = output_placeholder.clone()
        
        # сравниваем выходы
        assert torch.all(model_output == graph_output), "выход оригинальной модели и CUDA graph не совпадают"

        print("Success cuda graph")

    def graph_runner(input_data):
        input_placeholder.copy_(input_data)
        graph.replay()
        return output_placeholder

    # запускаем под no_grad, чтобы минимизировать потребление памяти (исключает выделение памяти под градиенты)
    with torch.no_grad():
        latency_benchmark(
            model, 
            test_data, 
            warmup_n=10, 
            benchmark_n=100,
        )
        latency_benchmark(
            graph_runner, 
            test_data, 
            warmup_n=10, 
            benchmark_n=100,
        )


def polygraphy_compatible_loader(dataloader):
    for images, _ in dataloader:
        yield  {"input": images.numpy()}  # Assuming the model only requires the image data, not labels.

def test2():

    model = mobilenet_v2(weights=MobileNetV2).eval()
    model.cpu()
    
    if False:
        sample_input = torch.randn(1, 3, 224, 224, device='cpu')
        input_names = ["input"]
        output_names = ["output"]
        output_file = "my-model-ssss.onnx"

        torch.onnx.export(model,               # model being run
                        sample_input,        # model input (or a tuple for multiple inputs)
                        output_file,         # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=12,    # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=input_names,   # the model's input names
                        output_names=output_names)
        
    if False:
        sample_input = torch.randn(1, 3, 224, 224, device='cpu')
        input_names = ["input"]
        output_names = ["output"]
        output_file = "my-model-dsdd.onnx"


        # Define dynamic axes
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # Dynamically adjust batch size, height, and width
            'output': {0: 'batch_size'}  # Assume output has dynamic batch size; adjust depending on model architecture
        }

        torch.onnx.export(model,               # model being run
                        sample_input,        # model input (or a tuple for multiple inputs)
                        output_file,         # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=12,    # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=input_names,   # the model's input names
                        output_names=output_names,
                        dynamic_axes=dynamic_axes)
        
    if True:
        # model_ssss = NetworkFromOnnxPath("my-model-ssss.onnx")
        # config = CreateConfig()

        # engine = engine_from_network(model_ssss, config=config)
        # save_engine(engine, path="my-model-ssss.engine")
    
        with open("my-model-ssss.engine", "rb") as f:
            engine_bytes = f.read()
        
        engine = EngineFromBytes(engine_bytes)

        with TrtRunner(engine) as trt_runner:
            def validation_trt_runner(input_data):
                # пропустим копирование на CPU copy_outputs_to_host=False
                output = trt_runner.infer(feed_dict={"input": input_data}, copy_outputs_to_host=False)
                return output['output']

            validate(validation_trt_runner, batch_size=1, height=224, width=224)
            latency_benchmark(validation_trt_runner, test_input=torch.ones(1, 3, 224, 224), warmup_n=10, benchmark_n=100)

    if False:
        model_dsdd = NetworkFromOnnxPath("my-model-dsdd.onnx")
        profiles=[
            Profile().add('input', min=(1, 3, 224, 224), opt=(32, 3, 224, 224), max=(64, 3, 640, 640))
        ]
        config = CreateConfig(profiles=profiles)
        engine = engine_from_network(model_dsdd, config=config)
        save_engine(engine, path="my-model-dsdd.engine")

        with open("my-model-dsdd.engine", "rb") as f:
            engine_bytes = f.read()
            
        engine = EngineFromBytes(engine_bytes)
        
        with TrtRunner(engine) as trt_runner:
            def validation_trt_runner(input_data):
                # пропустим копирование на CPU copy_outputs_to_host=False
                output = trt_runner.infer(feed_dict={"input": input_data}, copy_outputs_to_host=False)
                return output['output']

            validate(validation_trt_runner, batch_size=1, height=224, width=224)
            validate(validation_trt_runner, batch_size=64, height=224, width=224)


    if False:
        model_int8 = NetworkFromOnnxPath("my-model-dsdd.onnx")
        profiles=[
            Profile().add('input', min=(1, 3, 224, 224), opt=(32, 3, 224, 224), max=(64, 3, 640, 640))
        ]

        data_loader = imagenette_val_dataloader(32, 224, 224)
        
        calibrator = Calibrator(
            data_loader=polygraphy_compatible_loader(data_loader),
            cache='calibration.cache',
        )
        
        config = CreateConfig(
            int8=True,
            calibrator=calibrator,
            profiles=profiles
        )

        engine = engine_from_network(model_int8, config=config)
        save_engine(engine, path="my-model-int8.engine")

    with open("my-model-int8.engine", "rb") as f:
        engine_bytes = f.read()
        
    engine = EngineFromBytes(engine_bytes)

    bsz, ch, height, width = 64, 3, 224, 224
    test_data = torch.ones(bsz, ch, height, width, device="cuda")
    
    with TrtRunner(engine) as trt_runner:
        def validation_trt_runner(input_data):
            # пропустим копирование на CPU copy_outputs_to_host=False
            output = trt_runner.infer(feed_dict={"input": input_data}, copy_outputs_to_host=False)
            return output['output']

        #validate(validation_trt_runner, batch_size=64, height=224, width=224)

        latency_benchmark(
            validation_trt_runner, 
            test_data, 
            warmup_n=10, 
            benchmark_n=100,
        )

#test1()
test2()