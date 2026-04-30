import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
    def freeze_layers(self):
        """Freeze all layers except reparameterization layers"""
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
    def unfreeze_layers(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
            
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        
        # Initial fully connected layer
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Convolutional transpose layers
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensure output is between [0,1]
        )
        
    def freeze_layers(self):
        """Freeze all layers"""
        for param in self.parameters():
            param.requires_grad = False
        
    def unfreeze_layers(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
            
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 2, 2)
        x = self.conv_transpose(x)
        return x 