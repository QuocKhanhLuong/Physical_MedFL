"""med: A Flower / PyTorch app for medical image segmentation."""

import logging
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

# Add src to path for imports - use relative path from current file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.RobustMedVFL_UNet import RobustMedVFL_UNet
from data_handling.data_loader import get_federated_dataloaders
from utils.metrics import evaluate_metrics
from utils.losses import CombinedLoss 

# Global config
N_CLASSES = 4
IMG_SIZE = 256
ALPHA = 0.5
NUM_WORKERS = 4
DATA_PATH = "../data/ACDC_preprocessed"
PARTITION_STRATEGY = "iid"  # Default fallback
TRAINING_SOURCES = ["slices"]

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model():
    return RobustMedVFL_UNet(n_channels=1, n_classes=N_CLASSES)

def get_weights(net) -> List[np.ndarray]:
    """Get model weights as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters: List[np.ndarray]) -> None:
    """Set model weights from a list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

def load_data(partition_id: int, num_partitions: int, partition_strategy: str = "iid", alpha: float = 0.5):
    """Load federated data partitions with configurable strategy."""
    logger.info(f"Loading data for {num_partitions} clients using strategy: {partition_strategy}")
    
    trainloaders, valloaders, testloader = get_federated_dataloaders(
        data_path=DATA_PATH,
        num_clients=num_partitions,
        batch_size=8,
        partition_strategy=partition_strategy,  # Now configurable
        val_ratio=0.2,
        alpha=alpha,  # Now configurable
        training_sources=TRAINING_SOURCES,
        partition_by='patient',
        num_workers=NUM_WORKERS
    )
    
    logger.info("Dataloaders cached successfully with lazy loading")
    return trainloaders[partition_id], valloaders[partition_id], testloader

def train(net, trainloader, epochs, device, learning_rate=1e-4, kappa_values=None):
    """Train model with stabilized hybrid loss to prevent client drift."""
    net.to(device)
    net.train()
    
    # STEP 1: Dual loss approach to stabilize training
    criterion_combined = CombinedLoss(num_classes=N_CLASSES, 
                                     in_channels_maxwell=1024,
                                     lambda_val=15.0,
                                     initial_loss_weights=[0.3, 0.5, 0.5, 1.0]
                                     ).to(device)
    
    criterion_ce = nn.CrossEntropyLoss().to(device)  # Stable baseline loss
    
    logger.info(f"Using Hybrid Loss (Combined + CE) with kappa values: {kappa_values}")
    
    # STEP 2: Lower learning rate + regularization against overfitting
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=learning_rate,  # Reduced from 1e-3 to 1e-4
        weight_decay=1e-4  # Increased regularization from 1e-5 to 1e-4
    )
    
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            
            # RobustMedVFL_UNet always returns (logits, maxwell_outputs)
            logits, maxwell_outputs = outputs
            
            # HYBRID LOSS: Physics-informed + Stable CrossEntropy
            loss_combined = criterion_combined(logits, labels.long(), b1=logits, all_es=maxwell_outputs, feat_sm=logits)
            loss_ce = criterion_ce(logits, labels.long())
            
            # 70% physics + 30% stable (prevents client drift)
            loss = 0.7 * loss_combined + 0.3 * loss_ce
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1

        # Log loss weights and class weights for monitoring
        if hasattr(criterion_combined, 'get_current_loss_weights'):
            current_loss_weights = criterion_combined.get_current_loss_weights() 
        if hasattr(criterion_combined, 'get_current_class_weights'):
            current_class_weights = criterion_combined.get_current_class_weights()

    avg_trainloss = running_loss / max(num_batches, 1)
    return avg_trainloss

def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    net.eval()
    
    # Use advanced metrics evaluation
    metrics = evaluate_metrics(net, testloader, device, N_CLASSES)
    
    # Calculate foreground metrics (excluding background class)
    fg_dice_scores = metrics['dice_scores'][1:] if N_CLASSES > 1 else metrics['dice_scores']
    avg_fg_dice = sum(fg_dice_scores) / len(fg_dice_scores) if fg_dice_scores else 0.0
    
    # Calculate loss and accuracy for server compatibility
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_samples = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # RobustMedVFL_UNet always returns (logits, maxwell_outputs)
            logits, _ = outputs  # Take only logits, ignore Maxwell outputs
            
            loss = criterion(logits, labels.long())
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            # Calculate pixel accuracy
            predicted = logits.argmax(dim=1)
            correct_pixels += (predicted == labels).sum().item()
            total_pixels += labels.numel()
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    logger.info(f"Test evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Avg FG Dice: {avg_fg_dice:.4f}")
    return avg_loss, accuracy

def get_testloader():
    """Get test dataloader for server-side evaluation."""
    _, _, testloader = get_federated_dataloaders(
        data_path=DATA_PATH,
        num_clients=1,
        batch_size=8,
        partition_strategy="iid",
        val_ratio=0.2,
        training_sources=TRAINING_SOURCES,
        partition_by='patient',
        num_workers=NUM_WORKERS
    )
    return testloader