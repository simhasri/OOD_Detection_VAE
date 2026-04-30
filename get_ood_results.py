import warnings
warnings.filterwarnings('ignore')

from util import get_dataloaders
from ood import Likelihood, Mahalanobis, GramMatrix, LikelihoodRegret
import torch
from collections import defaultdict
from pathlib import Path
import json

def get_all_dataloaders(batch_size=64):
    """Get all dataloaders"""
    # Get train and test loaders for ID datasets
    cifar_train, cifar_test = get_dataloaders('cifar10', batch_size=batch_size)
    svhn_train, svhn_test = get_dataloaders('svhn', batch_size=batch_size)
    
    # Get loaders for OOD datasets
    noise_loader = get_dataloaders('noise', batch_size=batch_size)
    eurosat_loader = get_dataloaders('eurosat', batch_size=batch_size)
    
    return {
        'cifar10': {'train': cifar_train, 'test': cifar_test},
        'svhn': {'train': svhn_train, 'test': svhn_test},
        'noise': noise_loader,
        'eurosat': eurosat_loader
    }

def run_ood_detection():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get all dataloaders
    print("Loading datasets...")
    dataloaders = get_all_dataloaders()
    
    # Initialize results dictionary
    results = defaultdict(dict)
    
    # OOD detection methods
    methods = {
        'Likelihood': Likelihood,
        'Mahalanobis': Mahalanobis,
        'GramMatrix': GramMatrix,
        'LikelihoodRegret': LikelihoodRegret
    }
    
    # ID datasets
    id_datasets = ['cifar10', 'svhn']
    # All datasets (for OOD)
    all_datasets = ['cifar10', 'svhn', 'noise', 'eurosat']
    
    # Run all combinations
    for method_name, method_class in methods.items():
        print(f"\n{'-'*50}")
        print(f"Running {method_name}")
        print(f"{'-'*50}")
        
        for id_name in id_datasets:
            for ood_name in all_datasets:
                # Skip if ID and OOD are the same
                if id_name == ood_name:
                    continue
                
                print(f"\nTesting {id_name} (ID) vs {ood_name} (OOD)")
                
                # Create detector instance
                detector = method_class(
                    dataloader1=dataloaders[id_name]['test'],  # Use test set as ID
                    dataloader2=dataloaders[ood_name] if ood_name in ['noise', 'eurosat'] 
                                                    else dataloaders[ood_name]['test'],
                    name1=id_name,
                    name2=ood_name,
                    latent_dim=256,
                    model_dir='saved_models'
                )
                
                # Get scores and calculate AUROC
                try:
                    detector.get_scores()
                    auroc, threshold, accuracy, sample_size = detector.calculate_auroc()
                    
                    # Store results
                    results[method_name][f"{id_name}_vs_{ood_name}"] = {
                        'auroc': float(auroc),
                        'threshold': float(threshold),
                        'accuracy': float(accuracy),
                        'sample_size': sample_size
                    }
                    
                    
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    results[method_name][f"{id_name}_vs_{ood_name}"] = {
                        'error': str(e)
                    }
    
    return results

def save_results(results, filename='ood_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def print_results_table(results):
    # Print results grouped by dataset pairs
    print("\nResults Summary")
    print("="*100)
    print(f"{'Dataset Pair':<30} {'Method':<20} {'AUROC':<10} {'Accuracy':<10} {'Sample Size':<10}")
    print("-"*100)
    
    # Get all unique dataset pairs
    dataset_pairs = set()
    for method in results.values():
        dataset_pairs.update(method.keys())
    dataset_pairs = sorted(dataset_pairs)  # Sort for consistent ordering
    
    # Print results grouped by dataset pairs
    for pair in dataset_pairs:
        first_row = True
        for method in sorted(results.keys()):  # Sort methods for consistent ordering
            metrics = results[method].get(pair, {})
            
            # Print dataset pair only for first method
            pair_name = pair if first_row else ""
            
            if 'error' in metrics:
                print(f"{pair_name:<30} {method:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            else:
                print(f"{pair_name:<30} {method:<20} {metrics['auroc']:.4f} {metrics['accuracy']:.4f} {metrics['sample_size']}")
            
            first_row = False
        print("-"*100)  # Separator between dataset pairs
    
    print("="*100)

def main():
    # Run OOD detection
    results = run_ood_detection()
    
    # Save results
    save_results(results)
    
    # Print results table
    print_results_table(results)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main() 