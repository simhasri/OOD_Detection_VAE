# VAE-Based Out-of-Distribution Detection

This project implements and compares different VAE-based methods for Out-of-Distribution (OOD) detection. The implemented methods include Likelihood, Mahalanobis Distance, Gram Matrix, and Likelihood Regret approaches.
Authors: Prateek Mehta, Nithyasri Narasimhan, Paras Jain

## Project Structure

<br>
├── model.py           # VAE model architecture <br>
├── util.py           # Utility functions and dataset loaders <br>
├── ood.py            # OOD detection methods <br>
├── train_vae.py      # VAE training script <br>
├── get_ood_results.py # OOD evaluation script <br>
└── dataset_info.py   # Dataset information and validation <br>

## Installation


1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train VAE models on ID datasets:
```bash
python train_vae.py
```

2. Run OOD detection experiments:
```bash
python get_ood_results.py
```

## Implemented Methods

1. **Likelihood-based Detection**
   - Uses negative ELBO as OOD score
   - Implementation in `ood.py:Likelihood`

2. **Mahalanobis Distance**
   - Measures distance in latent space using class-conditional Gaussian distributions
   - Implementation in `ood.py:Mahalanobis`

3. **Gram Matrix**
   - Uses style statistics from intermediate layers
   - Implementation in `ood.py:GramMatrix`

4. **Likelihood Regret**
   - Measures improvement in likelihood after single-sample adaptation
   - Implementation in `ood.py:LikelihoodRegret`

## Datasets

- **In-Distribution (ID)**:
  - CIFAR-10
  - SVHN

- **Out-of-Distribution (OOD)**:
  - Random Noise
  - EuroSAT
  - Cross-dataset (CIFAR-10 vs SVHN)

## Results

Results are saved in:
- `ood_results.json`: Detailed metrics for each method and dataset pair
- `saved_models/`: Trained VAE checkpoints
- `training_plots/`: Training curves for VAE models

## Model Architecture

- **Encoder**: Convolutional neural network with 4 layers
- **Decoder**: Transposed convolutions for reconstruction
- **Latent Space**: 256-dimensional
- **Input Size**: 32x32x3 RGB images

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

