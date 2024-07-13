import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


from torchvision.models import resnet18

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def load_datasets(dataset_name):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'celeba':
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.CelebA(root='./data', split='test', download=True, transform=transform)
        num_classes = 2  # Assuming binary classification for CelebA attributes
    elif dataset_name == 'mini-fashion':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    return train_dataset, train_loader, test_loader, num_classes


def create_pseudo_labels(num_classes, batch_size):
    return torch.ones(batch_size, num_classes) / num_classes


def hessian_vector_product(model, dataloader, loss_fn, v, num_samples=100):
    model.eval()
    device = next(model.parameters()).device
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    def hvp_fn():
        flat_grad_grad_v = torch.zeros(num_params).to(device)
        count = 0
        for inputs, targets in dataloader:
            if count >= num_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])
            grad_grad_v = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
            flat_grad_grad_v += torch.cat([g.contiguous().view(-1) for g in grad_grad_v]).data
            count += 1
        flat_grad_grad_v /= count
        return flat_grad_grad_v

    return hvp_fn()


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def retrain_baseline(train_dataset, retain_indices, device, num_classes):
    baseline_model = CustomResNet(num_classes=num_classes).to(device)
    baseline_optimizer = optim.SGD(baseline_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    baseline_train_loader = DataLoader(Subset(train_dataset, retain_indices), batch_size=128, shuffle=True)
    
    baseline_model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(100):  # 重新训练100个epoch
        for inputs, targets in baseline_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            baseline_optimizer.zero_grad()
            outputs = baseline_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            baseline_optimizer.step()
    
    return baseline_model
