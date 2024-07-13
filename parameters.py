import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Sequential Unlearning')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'celeba', 'mini-fashion'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--T', type=int, default=10, help='Number of time points (default: 10)')
    parser.add_argument('--eta', type=float, default=0.01, help='Learning rate for sub-model (default: 0.01)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Coefficient for Hessian vector product (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for sub-model training (default: 5)')
    return parser.parse_args()
