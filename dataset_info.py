from util import get_dataloaders
import torch

def print_dataset_shapes(dataset_name):
    print(f"\n{'-'*50}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'-'*50}")
    
    # Get dataloaders
    if dataset_name in ['cifar10', 'svhn']:
        train_loader, test_loader = get_dataloaders(dataset_name, batch_size=128)
        
        # Print shapes for training set
        data, labels = next(iter(train_loader))
        print(f"Training data shape: {data.shape}")
        print(f"Training labels shape: {labels.shape}")
        print(f"Training data type: {data.dtype}")
        print(f"Training data range: [{data.min():.4f}, {data.max():.4f}]")
        
        # Print shapes for test set
        data, labels = next(iter(test_loader))
        print(f"Test data shape: {data.shape}")
        print(f"Test labels shape: {labels.shape}")
        print(f"Test data type: {data.dtype}")
        print(f"Test data range: [{data.min():.4f}, {data.max():.4f}]")
        
    else:  # noise and eurosat
        loader = get_dataloaders(dataset_name, batch_size=128)
        data, labels = next(iter(loader))
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

def main():
    # Test all datasets
    datasets = ['cifar10', 'svhn', 'noise', 'eurosat']
    
    for dataset_name in datasets:
        print_dataset_shapes(dataset_name)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    main()

