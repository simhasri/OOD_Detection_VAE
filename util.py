import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# Define NoiseDataset class
class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, transform=None):
        self.size = size
        # Generate random noise as numpy array
        self.data = np.random.rand(size, 32, 32, 3).astype(np.float32)
        self.transform = transform
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Convert to PIL Image
        img = (self.data[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        return img, 0

def get_dataloaders(dataset_name, batch_size=128, num_workers=2, root_dir='./data'):
    
    # Define common transforms for all datasets
    # Resize all images to 32x32 (CIFAR size) and normalize to [0,1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transform)
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
            
    elif dataset_name == 'noise':
        dataset = NoiseDataset(size=10000, transform=transform)
            
    elif dataset_name == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root=root_dir, split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(
            root=root_dir, split='test', download=True, transform=transform)
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
            
    elif dataset_name == 'eurosat':
        dataset = torchvision.datasets.EuroSAT(
            root=root_dir,
            download=True,
            transform=transform
        )
            
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: cifar10, noise, svhn, eurosat")

    # Create single dataloader for noise and eurosat
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for OOD datasets
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=10, min_delta=0, checkpoint_path='checkpoint.pt'):

        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = None
        self.counter = 0
        self.best_model = None
        
    def __call__(self, val_loss, encoder, decoder):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(encoder, decoder)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(encoder, decoder)
            self.counter = 0
        return False
        
    def save_checkpoint(self, encoder, decoder):
        """Save model when validation loss decreases."""
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'best_loss': self.best_loss
        }, self.checkpoint_path)
        
    def load_best_model(self, encoder, decoder):
        """Load the best model."""
        checkpoint = torch.load(self.checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

def plot_training_history(history, save_path=None):

    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_recon_loss'], label='Train')
    plt.plot(history['val_recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot KLD loss
    plt.subplot(1, 3, 3)
    plt.plot(history['train_kld_loss'], label='Train')
    plt.plot(history['val_kld_loss'], label='Validation')
    plt.title('KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train_vae(encoder, decoder, train_loader, test_loader, 
              num_epochs=100, learning_rate=1e-4, device='cuda',
              patience=10, dataset_name='unknown'):
   
    # Create saved_models directory if it doesn't exist
    save_dir = Path('saved_models')
    save_dir.mkdir(exist_ok=True)
    
    # Create checkpoint path with dataset name
    checkpoint_path = save_dir / f'{dataset_name}_checkpoint.pt'
    
    # Move models to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=patience,
        checkpoint_path=str(checkpoint_path)
    )
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'train_kld_loss': [],
        'val_recon_loss': [],
        'val_kld_loss': []
    }
    
    # Optimizers
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                               lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            z, mu, logvar = encoder(data)
            recon_batch = decoder(z)
            
            # Calculate losses
            recon_loss = torch.nn.functional.binary_cross_entropy(
                recon_batch, data, reduction='sum'
            )
            kld_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            
            # Total loss
            loss = recon_loss + kld_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            
            # Accumulate losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld_loss += kld_loss.item()
            
        # Calculate average training losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon = train_recon_loss / len(train_loader.dataset)
        avg_train_kld = train_kld_loss / len(train_loader.dataset)
        
        # Validation phase
        encoder.eval()
        decoder.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                z, mu, logvar = encoder(data)
                recon_batch = decoder(z)
                
                # Calculate losses
                recon_loss = torch.nn.functional.binary_cross_entropy(
                    recon_batch, data, reduction='sum'
                )
                kld_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                
                # Accumulate losses
                val_loss += (recon_loss + kld_loss).item()
                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_recon = val_recon_loss / len(test_loader.dataset)
        avg_val_kld = val_kld_loss / len(test_loader.dataset)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_recon_loss'].append(avg_train_recon)
        history['train_kld_loss'].append(avg_train_kld)
        history['val_recon_loss'].append(avg_val_recon)
        history['val_kld_loss'].append(avg_val_kld)
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f} '
              f'(Recon: {avg_train_recon:.4f}, KLD: {avg_train_kld:.4f})')
        print(f'Val Loss: {avg_val_loss:.4f} '
              f'(Recon: {avg_val_recon:.4f}, KLD: {avg_val_kld:.4f})')
        
        # Early stopping
        if early_stopping(avg_val_loss, encoder, decoder):
            print("Early stopping triggered")
            break
    
    return history