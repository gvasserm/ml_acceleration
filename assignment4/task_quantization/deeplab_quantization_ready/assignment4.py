import datetime
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization.quantize_fx import fuse_fx
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large
from tqdm import tqdm
import torch.autograd.profiler as profiler

import utils
from quantization_utils.fake_quantization import fake_quantization
from quantization_utils.static_quantization import quantize_static
from train import evaluate
from train import get_dataset
from train import train_one_epoch, criterion
import timeit

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')


def profile(model, bs=24, device='cpu'):
    # Ensure the model is in evaluation mode to disable layers like dropout and batchnorm during inference
    model.eval()

    input_tensor = torch.randn(bs, 3, 224, 224)
    input_tensor = input_tensor.to(device)
    model.to(device)

    # Warm-up (optional, but recommended for more accurate timing, especially on GPU)
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)

    # Forward pass
    n = 5
    with torch.no_grad():
    #with torch.autograd.profiler.profile() as prof:
        # Timing starts here
        start_time = time.time()
        for _ in range(n):
            output = model(input_tensor)
        # Timing ends here
        end_time = time.time()
        # Calculate and print the elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000/n/bs
    print(f"Elapsed time for the forward pass: {elapsed_time_ms:.2f}ms, batch size: {bs}")
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return


def train(student_model, teacher_model, args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)


    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn
    )

    student_model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()

    model_without_ddp = student_model

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(student_model, teacher_model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(student_model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    
def test1():

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Print current GPU memory usage
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print(f'Total: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')

    # Вытащил дефолтные аргументы, чтобы не упражняться с argparse в ноутбуке
    with Path('assignment4/task_quantization/deeplab_quantization_ready/torch_default_args.pickle').open('rb') as file:
        args = pickle.load(file)

    # Подобирайте под ваше железо
    args.data_path = '/home/gvasserm/data/coco2017/'
    args.epochs = 1
    args.batch_size = 24
    args.workers = 8

    print(args)

    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args, is_train=False)

    dataset_train, num_classes = get_dataset(args, is_train=True)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_sampler = torch.utils.data.SequentialSampler(dataset_train)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=24, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=24, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    model.cuda()

    if True:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)

    
    #model_fused = fuse_fx(deepcopy(model))
    if False:
        confmat = evaluate(model_fused, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)

    quantized_model = quantize_static(model, data_loader_train, num_batches=24, device='cuda:0')
    
    #print_model_size(quantized_model)
    #print_model_size(model)

    #x = torch.randn(24, 3, 1024, 1024)
    
    #profile(model.cpu(), bs=24, device='cpu')
    #profile(quantized_model, bs=24, device='cpu')
    #profile(quantized_model, bs=24, device='cpu')

    if True:
        data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=12, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
        confmat = evaluate(quantized_model, data_loader_test, device='cpu', num_classes=num_classes)
        print(confmat)

    profile(model, bs=24, device='cpu')

    return

def test2():

    from torchvision import transforms, datasets

    from torchvision.models import resnet50
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torch.ao.quantization import get_default_qconfig

    device = "cpu"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    batch_size = 8
   
    #fp32_model = resnet50().eval()
    fp32_model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT).eval()
    #model_fused = fuse_fx(deepcopy(fp32_model))
    #model = deepcopy(fp32_model)
    # `qconfig` means quantization configuration, it specifies how should we
    # observe the activation and weight of an operator
    # `qconfig_dict`, specifies the `qconfig` for each operator in the model
    # we can specify `qconfig` for certain types of modules
    # we can specify `qconfig` for a specific submodule in the model
    # we can specify `qconfig` for some functioanl calls in the model
    # we can also set `qconfig` to None to skip quantization for some operators
    qconfig = get_default_qconfig("x86")
    qconfig_dict = {"": qconfig}
    # `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
    
    trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    
    x = torch.randn(1, 3, 224, 224)
    model_prepared = prepare_fx(fp32_model, qconfig_dict, x)
    #model_quantized = quantize_static(fp32_model, trainloader, num_batches=24, device='cuda:0')
    # calibration runs the model with some sample data, which allows observers to record the statistics of
    # the activation and weigths of the operators
    calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
    for i in range(len(calibration_data)):
        model_prepared(calibration_data[i])
    # `convert_fx` converts a calibrated model to a quantized model, this includes inserting
    # quantize, dequantize operators to the model and swap floating point operators with quantized operators
    model_quantized = convert_fx(deepcopy(model_prepared))
    # benchmark
    profile(fp32_model, bs=32, device='cpu')
    profile(model_quantized, bs=32, device='cpu')
    return

def test3():

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Print current GPU memory usage
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print(f'Total: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')

    # Вытащил дефолтные аргументы, чтобы не упражняться с argparse в ноутбуке
    with Path('assignment4/task_quantization/deeplab_quantization_ready/torch_default_args.pickle').open('rb') as file:
        args = pickle.load(file)

    # Подобирайте под ваше железо
    args.data_path = '/home/gvasserm/data/coco2017/'
    args.epochs = 1
    args.batch_size = 24
    args.workers = 8

    print(args)

    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args, is_train=False)

    dataset_train, num_classes = get_dataset(args, is_train=True)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_sampler = torch.utils.data.SequentialSampler(dataset_train)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=24, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=24, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model.cuda()


    qat_model = fake_quantization(model, data_loader_train)
    qat_model.cuda()

    train(qat_model, args)

    # Инференс делаем на cpu, предварительно конвертируя модельку на CPU
    qat_model.cpu()
    int_qat_model = convert_fx(qat_model)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=12, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    confmat = evaluate(int_qat_model, data_loader_test, device='cpu', num_classes=num_classes)
    print(confmat)
    return

def test4():

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Print current GPU memory usage
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print(f'Total: {t}, Reserved: {r}, Allocated: {a}, Free: {f}')

    # Вытащил дефолтные аргументы, чтобы не упражняться с argparse в ноутбуке
    with Path('assignment4/task_quantization/deeplab_quantization_ready/torch_default_args.pickle').open('rb') as file:
        args = pickle.load(file)

    # Подобирайте под ваше железо
    args.data_path = '/home/gvasserm/data/coco2017/'
    args.epochs = 1
    args.batch_size = 16
    args.workers = 8

    print(args)

    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args, is_train=False)

    dataset_train, num_classes = get_dataset(args, is_train=True)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_sampler = torch.utils.data.SequentialSampler(dataset_train)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=24, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=24, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model.cuda()


    qat_model = fake_quantization(model, data_loader_train)
    qat_model.cuda()

    train(qat_model, model, args)

    # Инференс делаем на cpu, предварительно конвертируя модельку на CPU
    qat_model.cpu()
    int_qat_model = convert_fx(qat_model)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=12, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    confmat = evaluate(int_qat_model, data_loader_test, device='cpu', num_classes=num_classes)
    print(confmat)
    return


#test1()
#test2()
#test3()
test4()