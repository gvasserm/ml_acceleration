
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Union, List
from thop import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifar10 import get_train_transform, get_val_transform
from resnet import resnet18

from supernet import supernet18

import time


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

train_transform = get_train_transform()
val_transform = get_val_transform()

# Change this value if needed.
batch_size = 512

train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform,
)
test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform,
)

train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
test_dataloader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

def train_one_epoch(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        epoch: int,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    wrapped_dataloader = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (inputs, labels) in wrapped_dataloader:
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, predicted_labels = torch.max(logits, 1)
            total_loss += loss.item()
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.shape[0]

        wrapped_dataloader.set_description(
            f'(train) Epoch={epoch}, lr={scheduler.get_last_lr()[0]:.4f} loss={total_loss / (i + 1):.3f}'
        )

    return total_loss / len(dataloader), total_correct / total_samples


# TODO: Copy `train_one_epoch` function here, rename to `pretrain_one_epoch`.
#       Call `model.sample_random_architecture()` before making forward pass on each batch.
def pretrain_one_epoch(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        epoch: int,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    #wrapped_dataloader = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        model.sample_random_architecture()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, predicted_labels = torch.max(logits, 1)
            total_loss += loss.item()
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.shape[0]

        # wrapped_dataloader.set_description(
        #     f'(train) Epoch={epoch}, lr={scheduler.get_last_lr()[0]:.4f} loss={total_loss / (i + 1):.3f}'
        # )

    return total_loss / len(dataloader), total_correct / total_samples


@torch.no_grad()
def validate_one_epoch(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    #wrapped_dataloader = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        logits = model(inputs)
        loss = criterion(logits, labels)
        _, predicted_labels = torch.max(logits, 1)
        total_loss += loss.item()
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.shape[0]

        #wrapped_dataloader.set_description(f'(val) Epoch={epoch}, loss={total_loss / (i + 1):.3f}')

    return total_loss / len(dataloader), total_correct / total_samples


def measure_latency(model: torch.nn.Module, device: torch.device, input_shape=(1, 3, 224, 224), dtype=torch.float32, warmup_iters=10, measure_iters=100) -> float:
    """
    Measure the latency of a forward pass in a PyTorch model.

    Args:
    - model: The PyTorch model to measure.
    - device: The device to run the measurement on (e.g., 'cuda:0', 'cpu').
    - input_shape: The shape of the dummy input tensor for the model.
    - dtype: The data type of the dummy input tensor.
    - warmup_iters: Number of iterations to warm up the model (important for GPUs).
    - measure_iters: Number of iterations to measure the latency.

    Returns:
    - The average latency in milliseconds of a single forward pass.
    """
    
    # Move model to the specified device
    model.to(device)
    
    # Generate a dummy input tensor
    dummy_input = torch.randn(input_shape, dtype=dtype, device=device)
    
    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    
    # Measurement phase
    start_time = time.time()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(dummy_input)
    end_time = time.time()
    
    # Calculate average latency
    avg_latency = (end_time - start_time) / measure_iters * 1000.0  # Convert to milliseconds
    
    return avg_latency

def random_search(
        trained_supernet: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        n_architectures_to_test: int,
        target_latency: float,) -> Union[float, List[float]]:
    # TODO: Implement Random search.
    # This function should evaluate `n_architectures_to_test` architectures
    # using the `trained_supernet` on the whole validation dataset. The resulting
    # architecture should have latency not greater than `target_latency`.
    # Rank architectures by validation accuracy.
    best_accuracy = 0.0
    best_architecture = []
    
    for _ in tqdm(range(n_architectures_to_test)):
        # Sample a random architecture
        trained_supernet.sample_random_architecture()
        current_architecture = [block.active_op_index for block in trained_supernet.search_blocks]
        
        # Measure its latency
        torch.cuda.synchronize() 
        macs, params = profile(trained_supernet, inputs=(torch.zeros(1, 3, 32, 32, device=device),),verbose=False)
        
        # Skip this architecture if it exceeds the target latency
        if macs > target_latency:
            continue
        
        # Evaluate the architecture
        trained_supernet.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = trained_supernet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        
        # Update best architecture if this one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_architecture = current_architecture
    
    return best_accuracy, best_architecture


def test():

    # Select suitable device.
    # You should probably use either cuda (NVidia GPU) or mps (Apple) backend.
    device = torch.device('cuda:0')

    channel_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]

    supernet = supernet18(num_classes=10, zero_init_residual=True, channel_multipliers=channel_multipliers)
    supernet.to(device=device)

    # TODO: define hyperparameters for supernet training.
    criterion = nn.CrossEntropyLoss()
    lr = 0.25
    weight_decay = 5e-4
    momentum = 0.9
    n_epochs = 60  # Longer training gives better results, but let's keep baseline model epochs to 20.

    # TODO: build optimizer and scheduler for supernet training.
    optimizer = optim.SGD(supernet.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * n_epochs)

    for epoch in tqdm(range(n_epochs)):
        loss, accuracy = pretrain_one_epoch(supernet, criterion, train_dataloader, optimizer, scheduler, device, epoch)
        if epoch % 20 ==0:
            print(f'Epoch: {epoch}')
            print(f'train_loss={loss:.4f}, train_accuracy={accuracy:.3%}')
        loss, accuracy = validate_one_epoch(supernet, criterion, test_dataloader, device, epoch)
        if epoch % 20 ==0:
            print(f'test_loss={loss:.4f}, test_accuracy={accuracy:.3%}')

    
    n_architectures_to_test = 100
    target_latency = 30 * 1e6

    accuracy, best_architecture = random_search(supernet, train_dataloader, test_dataloader, device, n_architectures_to_test, target_latency)
    print(f'best architecture: {best_architecture} (test_accuracy={accuracy:.3%})')

    torch.save(supernet.state_dict(), 'supernet.pth')


def test1():

    n_architectures_to_test = 100
    target_latency = 30 * 1e6

    device = torch.device('cuda:0')

    channel_multipliers = (0.5, 0.75, 1.0)
    supernet = supernet18(num_classes=10, zero_init_residual=True, channel_multipliers=channel_multipliers)

    state_dict = torch.load('supernet.pth')
    supernet.load_state_dict(state_dict)
    supernet.to(device=device)
    
    accuracy, best_architecture = random_search(supernet, train_dataloader, test_dataloader, device, n_architectures_to_test, target_latency)
    print(f'best architecture: {best_architecture} (test_accuracy={accuracy:.3%})')

def retrain():

    best_architecture = [3, 3, 1, 0, 0, 0, 1, 0]
    device = torch.device('cuda:0')

    #state_dict = torch.load('supernet.pth')
    channel_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    supernet = supernet18(num_classes=10, zero_init_residual=True, channel_multipliers=channel_multipliers)

    #supernet.load_state_dict(state_dict)
    supernet.sample(best_architecture)
    supernet.to(device=device)

    macs, params = profile(supernet, inputs=(torch.zeros(1, 3, 32, 32, device=device),),verbose=False)
    print(f'Number of macs: {macs / 1e6:.2f}M, number of parameters: {params / 1e6:.2f}M')

    n_epochs = 20
    criterion = nn.CrossEntropyLoss()

    lr = 0.25
    weight_decay = 5e-4
    momentum = 0.9
    n_epochs = 20  # Longer training gives better results, but let's keep baseline model epochs to 20.

    optimizer = optim.SGD(supernet.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * n_epochs)

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')
        loss, accuracy = train_one_epoch(supernet, criterion, train_dataloader, optimizer, scheduler, device, epoch)
        print(f'train_loss={loss:.4f}, train_accuracy={accuracy:.3%}')
        loss, accuracy = validate_one_epoch(supernet, criterion, test_dataloader, device, epoch)
        print(f'test_loss={loss:.4f}, test_accuracy={accuracy:.3%}')


#test()

#test1()

retrain()
    