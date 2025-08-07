"""med: A Flower / PyTorch app for medical image segmentation."""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

# Add src to path for imports - use absolute path
project_root = "/home/aidev/Physical_MedFL"
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.RobustMedVFL_UNet import RobustMedVFL_UNet
from data_handling.data_loader import get_federated_dataloaders
from utils.metrics import evaluate_metrics
from utils.losses import CombinedLoss 

# Global variables
N_CLASSES = 4
IMG_SIZE = 256
ALPHA = 0.5
NUM_WORKERS = 4
DATA_PATH = "../data/ACDC_preprocessed"
PARTITION_STRATEGY = "iid" 
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

def load_data(partition_id: int, num_partitions: int):
    """Load partition data for federated learning."""
    logger.info(f"Loading federated dataloaders for {num_partitions} clients")
    
    trainloaders, valloaders, testloader = get_federated_dataloaders(
        data_path=DATA_PATH,
        num_clients=num_partitions,
        batch_size=8,
        partition_strategy=PARTITION_STRATEGY,
        val_ratio=0.2,
        alpha=ALPHA,
        training_sources=TRAINING_SOURCES,
        partition_by='patient',
        num_workers=NUM_WORKERS
    )
    
    logger.info("Dataloaders cached successfully with lazy loading")
    return trainloaders[partition_id], valloaders[partition_id], testloader

def train(net, trainloader, epochs, device, learning_rate=1e-3, kappa_values=None):
    """Train the model on the training set with adaptive loss."""
    net.to(device)
    net.train()
    
    criterion = CombinedLoss(num_classes=N_CLASSES, 
                                in_channels_maxwell=1024,
                                lambda_val=15.0,
                                initial_loss_weights=[0.3, 0.5, 0.5, 1.0]
                                ).to(device)
    
    logger.info(f"Using Combined Loss with kappa values: {kappa_values}")
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5  # Regularization to prevent local overfitting
    )
    
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            
            # Handle tuple output from RobustMedVFL_UNet
            if isinstance(outputs, tuple):
                # RobustMedVFL_UNet returns (logits, maxwell_outputs)
                logits, maxwell_outputs = outputs
            else:
                # Fallback for simple models
                logits = outputs
                maxwell_outputs = None
            
            # Use CombinedLoss with physics components
            if maxwell_outputs is not None:
                # Use physics loss with Maxwell outputs
                b1 = logits  # Use logits as B1 field approximation
                all_es = maxwell_outputs  # Maxwell solver outputs from each decoder block
                feat_sm = logits  # Use logits for smoothness regularization
                loss = criterion(logits, labels.long(), b1=b1, all_es=all_es, feat_sm=feat_sm)
            else:
                # Use loss without physics components  
                loss = criterion(logits, labels.long())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1

        # Lấy trọng số cân bằng giữa các loss
        current_loss_weights = criterion.get_current_loss_weights() 
        # Lấy trọng số của các class
        current_class_weights = criterion.get_current_class_weights()

        # In ra để theo dõi
        # print("Current Loss Weights:", loss_weights)
        # print("Current Class Weights:", class_weights)
        print(f"\n   Epoch {epoch+1} - Learned Loss Weights:")
        for name, weight in current_loss_weights.items():
            print(f"     - {name}: {weight:.4f}")
            
        print(f"   Epoch {epoch+1} - Dynamic Class Weights (for CE):")
        class_weights_str = " | ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(current_class_weights.values())])
        print(f"     - {class_weights_str}")
            
        print("-" * 60)
        
    
    avg_trainloss = running_loss / max(num_batches, 1)
    return avg_trainloss

def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    net.eval()
    
    try:
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
                # Handle tuple output from RobustMedVFL_UNet
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Take only logits, ignore Maxwell outputs
                else:
                    logits = outputs
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
    
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        # Fallback to simple evaluation
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        total_pixels = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # Handle tuple output from RobustMedVFL_UNet
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Take only logits
                else:
                    logits = outputs
                loss += criterion(logits, labels.long()).item()
                predicted = logits.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total_pixels += labels.numel()
        
        accuracy = correct / total_pixels if total_pixels > 0 else 0.0
        avg_loss = loss / len(testloader)
        
        logger.info(f"Fallback evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

def get_testloader():
    """Get test dataloader for server-side evaluation."""
    try:
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
    except Exception as e:
        logger.error(f"Error loading test dataloader: {e}")
        return None
