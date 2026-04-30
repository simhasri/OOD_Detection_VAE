import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import seaborn as sns
from pathlib import Path
from model import Encoder, Decoder

class OOD_Detector:
    def __init__(self, dataloader1, dataloader2, name1="ID", name2="OOD"):
        
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.name1 = name1
        self.name2 = name2
        
        # Calculate minimum size between dataloaders
        self.min_size = min(len(dataloader1.dataset), len(dataloader2.dataset))
        print(f"Using {self.min_size} samples from each dataset")
        
        # Initialize storage for scores
        self.scores1 = None
        self.scores2 = None
        
    def calculate_score(self, x):
        # Dummy function to be overridden by child classes
        
        raise NotImplementedError("This method should be implemented by child classes")
    
    def get_scores(self, device='cuda'):
        
        self.scores1 = []
        self.scores2 = []
        
        # Calculate scores for first dataloader
        print(f"\nCalculating scores for {self.name1} dataset:")
        count = 0
        with torch.no_grad():
            for data, _ in tqdm(self.dataloader1):
                data = data.to(device)
                batch_scores = self.calculate_score(data)
                self.scores1.extend(batch_scores)
                count += len(batch_scores)
                if count >= self.min_size:
                    break
        
        # Calculate scores for second dataloader
        print(f"\nCalculating scores for {self.name2} dataset:")
        count = 0
        with torch.no_grad():
            for data, _ in tqdm(self.dataloader2):
                data = data.to(device)
                batch_scores = self.calculate_score(data)
                self.scores2.extend(batch_scores)
                count += len(batch_scores)
                if count >= self.min_size:
                    break
        
        # Trim to min_size
        self.scores1 = np.array(self.scores1[:self.min_size])
        self.scores2 = np.array(self.scores2[:self.min_size])
    
    
    def calculate_auroc(self):
      
         # Calculate AUROC score and find best threshold for binary classification
        
        if self.scores1 is None or self.scores2 is None:
            raise ValueError("Scores haven't been calculated yet. Run get_scores first.")
        
        # Create labels (0 for ID, 1 for OOD)
        labels = np.concatenate([np.zeros(len(self.scores1)), 
                               np.ones(len(self.scores2))])
        
        # Concatenate scores
        scores = np.concatenate([self.scores1, self.scores2])
        
        # Find best threshold
        thresholds = np.percentile(scores, np.linspace(0, 100, 1000))
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Get final binary predictions using best threshold
        final_predictions = (scores >= best_threshold).astype(int)
        
        # Calculate AUROC using binary predictions
        auroc = roc_auc_score(labels, final_predictions)
        
        # Print classification results with best threshold
        id_correct = (self.scores1 < best_threshold).mean()
        ood_correct = (self.scores2 >= best_threshold).mean()
        print(f"ID: {self.name1} VS OOD: {self.name2}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Correctly classified {self.name1} samples: {id_correct:.4f}")
        print(f"Correctly classified {self.name2} samples: {ood_correct:.4f}")
        print(f"\nAUROC Score: {auroc:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return auroc, best_threshold, best_accuracy, self.min_size

class Likelihood(OOD_Detector):
    def __init__(self, dataloader1, dataloader2, name1="ID", name2="OOD", 
                 latent_dim=256, model_dir='saved_models'):
       
        super().__init__(dataloader1, dataloader2, name1, name2)
        
        self.model_dir = Path(model_dir)
        self.latent_dim = latent_dim
        
        # Load VAE models for the ID dataset
        self.encoder, self.decoder = self._load_vae_models(name1)
        
    def _load_vae_models(self, dataset_name):
        # Initialize models
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        
        # Load checkpoint
        checkpoint_path = self.model_dir / f'{dataset_name}_checkpoint.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        return encoder, decoder
    
    def calculate_score(self, x):
        
        # Move models to same device as input
        device = x.device
        self.encoder.to(device)
        self.decoder.to(device)
        
        # Set models to eval mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Get latent representation
        z, mu, logvar = self.encoder(x)
        
        # Reconstruct input
        recon_x = self.decoder(z)
        
        # Calculate reconstruction loss (negative log likelihood)
        recon_loss = torch.nn.functional.binary_cross_entropy(
            recon_x, x, reduction='none'
        )
        
        # Sum over all dimensions except batch
        recon_loss = recon_loss.sum(dim=(1,2,3))
        
        # Calculate KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        
        # Total negative ELBO (Evidence Lower BOund)
        neg_elbo = recon_loss + kl_loss
        
        # Convert to numpy array
        return neg_elbo.cpu().numpy()

class Mahalanobis(OOD_Detector):
    def __init__(self, dataloader1, dataloader2, name1="ID", name2="OOD", 
                 latent_dim=256, model_dir='saved_models'):
        
        super().__init__(dataloader1, dataloader2, name1, name2)
        
        self.model_dir = Path(model_dir)
        self.latent_dim = latent_dim
        
        # Load VAE models for the ID dataset
        self.encoder, _ = self._load_vae_models(name1)
        
        # Initialize mean and covariance for ID data
        self.mean = None
        self.inv_cov = None
        
    
    def _load_vae_models(self, dataset_name):
        # Initialize models
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        
        # Load checkpoint
        checkpoint_path = self.model_dir / f'{dataset_name}_checkpoint.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        return encoder, decoder
    
    def _compute_id_statistics(self, device='cuda'):
        print("\nComputing ID statistics...")
        self.encoder.eval()
        
        # Collect latent representations
        latent_vectors = []
        count = 0
        
        with torch.no_grad():
            for data, _ in tqdm(self.dataloader1):
                data = data.to(device)
                # Get latent representation (use mu, not sampled z)
                _, mu, _ = self.encoder(data)
                latent_vectors.extend(mu.cpu().numpy())
                count += len(mu)
                if count >= self.min_size:
                    break
        
        # Convert to numpy array
        latent_vectors = np.array(latent_vectors[:self.min_size])
        
        # Compute mean and covariance
        self.mean = np.mean(latent_vectors, axis=0)
        cov = np.cov(latent_vectors.T)
        
        # Add small constant to diagonal for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # Compute inverse covariance matrix
        self.inv_cov = np.linalg.inv(cov)
        
    def calculate_score(self, x):
        # Move models to same device as input
        device = x.device
        self.encoder.to(device)
        
        # Set model to eval mode
        self.encoder.eval()
        
        # Compute ID statistics if not already computed
        if self.mean is None or self.inv_cov is None:
            self._compute_id_statistics(device)
        
        # Get latent representation (use mu, not sampled z)
        _, mu, _ = self.encoder(x)
        mu = mu.cpu().numpy()
        
        # Calculate Mahalanobis distance for each sample
        diff = mu - self.mean
        scores = np.sum(diff.dot(self.inv_cov) * diff, axis=1)
        
        return scores

class GramMatrix(OOD_Detector):
    def __init__(self, dataloader1, dataloader2, name1="ID", name2="OOD", 
                 latent_dim=256, model_dir='saved_models'):
        
        super().__init__(dataloader1, dataloader2, name1, name2)
        
        self.model_dir = Path(model_dir)
        self.latent_dim = latent_dim
        
        # Load VAE models for the ID dataset
        self.encoder, _ = self._load_vae_models(name1)
        
        # Initialize reference gram matrices
        self.ref_grams = None
        
    def _load_vae_models(self, dataset_name):
        # Initialize models
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        
        # Load checkpoint
        checkpoint_path = self.model_dir / f'{dataset_name}_checkpoint.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        return encoder, decoder
    
    def _compute_gram_matrix(self, features):
        B, C, H, W = features.size()
        features = features.view(B, C, -1)  # [B, C, H*W]
        gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
        return gram / (C * H * W)
    
    def _compute_reference_grams(self, device='cuda'):
        print("\nComputing reference Gram matrices...")
        self.encoder.eval()
        
        # Get the convolutional layers
        conv_layers = self.encoder.conv_layers
        
        # Initialize storage for gram matrices from each layer
        all_grams = []
        batch_grams = []
        count = 0
        
        with torch.no_grad():
            for data, _ in tqdm(self.dataloader1):
                data = data.to(device)
                
                # Get intermediate features
                x = data
                layer_grams = []
                
                for layer in conv_layers:
                    x = layer(x)
                    if isinstance(layer, torch.nn.Conv2d):
                        gram = self._compute_gram_matrix(x)
                        layer_grams.append(gram.cpu())
                
                batch_grams.append(layer_grams)
                count += len(data)
                
                if count >= self.min_size:
                    # Process only complete batches up to min_size
                    num_complete_batches = self.min_size // len(data)
                    batch_grams = batch_grams[:num_complete_batches]
                    break
        
        # Compute mean gram matrices for each layer
        self.ref_grams = []
        num_layers = len(batch_grams[0])
        
        for layer_idx in range(num_layers):
            # Stack only tensors from complete batches
            layer_grams = torch.cat([grams[layer_idx] for grams in batch_grams], dim=0)
            # Take only up to min_size samples
            layer_grams = layer_grams[:self.min_size]
            mean_gram = layer_grams.mean(dim=0)
            self.ref_grams.append(mean_gram)
    
    def calculate_score(self, x):
        # Move models to same device as input
        device = x.device
        self.encoder.to(device)
        
        # Set model to eval mode
        self.encoder.eval()
        
        # Compute reference gram matrices if not already computed
        if self.ref_grams is None:
            self._compute_reference_grams(device)
        
        # Move reference grams to device
        ref_grams = [gram.to(device) for gram in self.ref_grams]
        
        # Get intermediate features and compute gram matrices
        scores = []
        with torch.no_grad():
            # Get intermediate features
            features = x
            layer_idx = 0
            
            for layer in self.encoder.conv_layers:
                features = layer(features)
                if isinstance(layer, torch.nn.Conv2d):
                    # Compute gram matrix
                    gram = self._compute_gram_matrix(features)
                    
                    # Compute Frobenius norm difference with reference
                    diff = gram - ref_grams[layer_idx].unsqueeze(0)
                    layer_score = torch.norm(diff, p='fro', dim=(1,2))
                    scores.append(layer_score)
                    
                    layer_idx += 1
        
        # Combine scores from all layers
        final_scores = torch.stack(scores, dim=0).sum(dim=0)
        
        return final_scores.cpu().numpy()

class LikelihoodRegret(OOD_Detector):
    def __init__(self, dataloader1, dataloader2, name1="ID", name2="OOD", 
                 latent_dim=256, model_dir='saved_models', num_epochs=10,
                 learning_rate=1e-4):
        
        super().__init__(dataloader1, dataloader2, name1, name2)
        
        self.model_dir = Path(model_dir)
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Load VAE models for the ID dataset
        self.encoder, self.decoder = self._load_vae_models(name1)
        
    def _load_vae_models(self, dataset_name):
        # Initialize models
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        
        # Load checkpoint
        checkpoint_path = self.model_dir / f'{dataset_name}_checkpoint.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        return encoder, decoder
    
    def _calculate_likelihood(self, x, encoder, decoder):
        """Calculate negative ELBO (likelihood) for a single sample"""
        # Get latent representation
        z, mu, logvar = encoder(x)
        
        # Reconstruct input
        recon_x = decoder(z)
        
        # Calculate reconstruction loss
        recon_loss = torch.nn.functional.binary_cross_entropy(
            recon_x, x, reduction='sum'
        )
        
        # Calculate KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total negative ELBO
        return recon_loss + kl_loss
    
    def calculate_score(self, x):
        """
        Calculate likelihood regret score for input data
        Process:
        1. Calculate initial likelihood with original model
        2. Create copy of model and train on single sample (only reparametrization layers are trained)
        3. Calculate new likelihood
        4. Score is difference between new and old likelihood
        """
        device = x.device
        batch_size = x.size(0)
        scores = []
        
        # Process each sample individually
        for i in range(batch_size):
            # Get single sample
            sample = x[i:i+1]  # Keep batch dimension
            
            # Create copies of encoder and decoder
            encoder_copy = Encoder(latent_dim=self.latent_dim).to(device)
            decoder_copy = Decoder(latent_dim=self.latent_dim).to(device)
            
            # Load weights
            encoder_copy.load_state_dict(self.encoder.state_dict())
            decoder_copy.load_state_dict(self.decoder.state_dict())
            
            # Calculate initial likelihood with frozen models
            encoder_copy.eval()
            decoder_copy.eval()
            with torch.no_grad():
                initial_likelihood = self._calculate_likelihood(
                    sample, encoder_copy, decoder_copy
                )
            
            # Train mode (only affects unfrozen layers)
            encoder_copy.train()
            decoder_copy.train()
            
            # Freeze all layers
            encoder_copy.freeze_layers()  # Freezes conv layers but keeps fc_mu and fc_logvar trainable
            decoder_copy.freeze_layers()  # Freezes all decoder layers
            
            # Create optimizer only for trainable parameters
            trainable_params = [p for p in encoder_copy.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
            
            # Train on single sample
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                loss = self._calculate_likelihood(sample, encoder_copy, decoder_copy)
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
            
            # Calculate final likelihood
            encoder_copy.eval()
            decoder_copy.eval()
            with torch.no_grad():
                final_likelihood = self._calculate_likelihood(
                    sample, encoder_copy, decoder_copy
                )
            
            # Calculate regret (improvement in likelihood)
            regret = (initial_likelihood - final_likelihood).item()
            scores.append(regret)
            
            # Clean up
            del encoder_copy
            del decoder_copy
        
        return np.array(scores)