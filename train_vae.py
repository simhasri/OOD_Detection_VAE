import torch
from model import Encoder, Decoder
from util import get_dataloaders, train_vae, plot_training_history
from pathlib import Path

def train_dataset_vae(dataset_name, batch_size=128, num_epochs=100, 
                     latent_dim=256, learning_rate=1e-3, patience=10):
    
    print(f"\nTraining VAE on {dataset_name.upper()} dataset:")
    print("=" * 50)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size=batch_size)
    print(f"Dataset loaded with {len(train_loader.dataset)} training samples "
          f"and {len(test_loader.dataset)} test samples")
    
    # Initialize models
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    print("Models initialized")
    
    # Create directory for plots
    plots_dir = Path('training_plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Train VAE
    print("\nStarting training...")
    history = train_vae(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=patience,
        dataset_name=dataset_name
    )
    
    # Plot and save training curves
    plot_path = plots_dir / f'{dataset_name}_training_curves.png'
    plot_training_history(history, save_path=str(plot_path))
    print(f"\nTraining curves saved to {plot_path}")
    print("=" * 50)
    
    return encoder, decoder, history

def main():
    # Common parameters
    params = {
        'batch_size': 128,
        'num_epochs': 100,
        'latent_dim': 256,
        'learning_rate': 1e-3,
        'patience': 10
    }
    
    # Datasets to train on
    datasets = ['cifar10', 'svhn']
    
    # Train VAE on each dataset
    for dataset_name in datasets:
        encoder, decoder, history = train_dataset_vae(
            dataset_name=dataset_name,
            **params
        )

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main() 