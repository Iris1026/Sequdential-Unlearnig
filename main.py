import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils import CustomResNet, load_datasets, create_pseudo_labels, hessian_vector_product, evaluate_accuracy, retrain_baseline
from parameters import args_parser



def sequential_unlearning(train_dataset, T, eta, alpha, batch_size, epochs, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomResNet(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    theta_0 = [p.clone().detach() for p in model.parameters()]
    prev_theta_star = [p.clone().detach() for p in model.parameters()]
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)

    for t in range(1, T+1):
        Ft_indices = indices[(t-1)*1000 : t*1000]
        Ft = Subset(train_dataset, Ft_indices)
        Ft_loader = DataLoader(Ft, batch_size=batch_size, shuffle=True)

        sub_model = CustomResNet(num_classes=num_classes).to(device)
        sub_model.load_state_dict(model.state_dict())

        sub_optimizer = optim.SGD(sub_model.parameters(), lr=eta, momentum=0.9, weight_decay=5e-4)
        sub_model.train()
        pseudo_labels = create_pseudo_labels(num_classes, batch_size).to(device)

        for epoch in range(epochs):
            for inputs, _ in Ft_loader:
                inputs = inputs.to(device)
                sub_optimizer.zero_grad()
                outputs = sub_model(inputs)
                loss = loss_fn(outputs, pseudo_labels[:outputs.size(0)])
                loss.backward()
                sub_optimizer.step()

        theta_T_F = [p.clone().detach() for p in sub_model.parameters()]

        sub_model.eval()
        total_loss = torch.tensor(0.0).to(device)
        for inputs, _ in Ft_loader:
            inputs = inputs.to(device)
            outputs = sub_model(inputs)
            batch_loss = loss_fn(outputs, pseudo_labels[:outputs.size(0)])
            total_loss += batch_loss.mean()
        total_loss = total_loss.mean()

        grad_L_t_F = torch.autograd.grad(total_loss, sub_model.parameters(), allow_unused=True, retain_graph=True)
        grad_L_t_F = [g for g in grad_L_t_F if g is not None]
        if not grad_L_t_F:
            raise ValueError("grad_L_t_F is empty. Please check the computation of total_loss and ensure it is a scalar tensor.")
        flat_grad_L_t_F = torch.cat([g.view(-1) for g in grad_L_t_F]).to(device)

        Ft_size = len(Ft_indices)
        R0_size = len(train_dataset)

        remaining_indices = np.setdiff1d(indices, Ft_indices)
        remaining_loader = DataLoader(Subset(train_dataset, remaining_indices), batch_size=batch_size, shuffle=False)
        grad_g_T_R = torch.zeros_like(flat_grad_L_t_F)
        for inputs, targets in remaining_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])
            grad_g_T_R += flat_grads
        grad_g_T_R /= len(remaining_loader)

        term_1 = (Ft_size / R0_size) * hessian_vector_product(model, remaining_loader, loss_fn, grad_g_T_R, num_samples=100)
        term_2 = (len(np.concatenate([indices[(i-1)*1000:i*1000] for i in range(1, t+1)])) / R0_size) * hessian_vector_product(model, remaining_loader, loss_fn, flat_grad_L_t_F, num_samples=100)

        start = 0
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                param_size = param.numel()
                param_term_1 = term_1[start:start + param_size].view_as(param)
                param_term_2 = term_2[start:start + param_size].view_as(param)
                param.copy_(
                    (T / (T+1)) * prev_theta_star[i] +
                    (1 / (T+1)) * theta_T_F[i] +
                    (alpha / (T+1)) * (param_term_1 + param_term_2)
                )
                start += param_size

        prev_theta_star = [p.clone().detach() for p in model.parameters()]

        Rt_indices = np.setdiff1d(indices, np.concatenate([indices[(i-1)*1000:i*1000] for i in range(1, t+1)]))
        Ft_dataloader = DataLoader(Subset(train_dataset, Ft_indices), batch_size=128, shuffle=False)
        Rt_dataloader = DataLoader(Subset(train_dataset, Rt_indices), batch_size=128)

        Acc_FT = evaluate_accuracy(model, Ft_dataloader)
        Acc_RT = evaluate_accuracy(model, Rt_dataloader)
        if t > 1:
            F1_T_1_indices = np.concatenate([indices[(i-1)*1000:i*1000] for i in range(1, t)])
            Acc_F1_T_1 = evaluate_accuracy(model, DataLoader(Subset(train_dataset, F1_T_1_indices), batch_size=128, shuffle=False))
        else:
            Acc_F1_T_1 = 0
        
        print(f"Time {t}: Acc_FT: {Acc_FT:.4f}, Acc_RT: {Acc_RT:.4f}, Acc_F1_T_1: {Acc_F1_T_1:.4f}")

    return model

if __name__ == "__main__":
    args = args_parser()
    train_dataset, train_loader, test_loader, num_classes = load_datasets(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    F_indices = [indices[(t-1)*1000 : t*1000] for t in range(1, args.T+1)]
    R_indices = np.setdiff1d(indices, np.concatenate(F_indices))

    model = sequential_unlearning(train_dataset, args.T, args.eta, args.alpha, args.batch_size, args.epochs, num_classes)
    baseline_model = retrain_baseline(train_dataset, R_indices, device, num_classes)
    Acc_baseline = evaluate_accuracy(baseline_model, DataLoader(Subset(train_dataset, R_indices), batch_size=128, shuffle=False))
    print(f"Baseline accuracy: {Acc_baseline:.4f}")
