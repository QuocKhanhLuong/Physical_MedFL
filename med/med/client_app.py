"""med: Flower client with PyTorch Lightning multi-GPU support."""

from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as L

# Dán đoạn code này vào đầu file client_app.py 
import torch.nn.functional as F
from typing import Union, Tuple, Any
class SimpleDiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(SimpleDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        total_dice_loss = 0.0
        for i in range(self.num_classes):
            pred_class = probs[:, i, :, :]
            target_class = targets_one_hot[:, i, :, :]
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
            total_dice_loss += (1.0 - dice_score)
            
        return total_dice_loss / self.num_classes

class SimpleDiceCELoss(nn.Module):
    def __init__(self, num_classes, weight_dice=0.5, weight_ce=0.5):
        super(SimpleDiceCELoss, self).__init__()
        self.dice_loss = SimpleDiceLoss(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, logits, targets):
        loss_d = self.dice_loss(logits, targets)
        loss_c = self.ce_loss(logits, targets.long())
        return self.weight_dice * loss_d + self.weight_ce * loss_c

# Add src to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.losses import CombinedLoss
from utils.metrics import evaluate_metrics
from .task import get_model, set_weights, get_weights, load_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Simple Classes for Research ---
class ExperimentConfig:
    """Simple experiment config."""
    def __init__(self, **kwargs):
        self.created_at = datetime.now()
        self.client_id: str = ""
        self.local_epochs: int = 5
        self.experiment_name: str = ""
        self.learning_rate: float = 1e-4
        self.seed: int = 42
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class MetricsTracker:
    """Simple metrics tracking."""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.round_metrics: List[Dict[str, Any]] = []

    def log_round(self, round_num: int, train_metrics: Dict, val_metrics: Dict, compute_metrics: Dict) -> None:
        round_data = {
            "round": round_num, 
            "timestamp": datetime.now().isoformat(), 
            "client_id": self.client_id,
            "training": train_metrics, 
            "validation": val_metrics, 
            "computational": compute_metrics
        }
        self.round_metrics.append(round_data)
        logger.info(
            f"Client {self.client_id} | Round {round_num} | "
            f"Train Loss: {train_metrics.get('avg_train_loss', 0):.4f} | "
            f"Val Dice: {val_metrics.get('avg_foreground_dice', 0):.4f}"
        )

class MedicalModel(L.LightningModule):
    """Lightning module đã được sửa đổi để ổn định hơn."""
    
    def __init__(self, net: nn.Module, learning_rate: float = 1e-4, num_classes: int = 4):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        
        # CHANGED: Thay thế toàn bộ loss phức tạp bằng một loss đơn giản, stateless
        self.loss_fn = SimpleDiceCELoss(num_classes=num_classes)
        
        # Metrics tracking (không đổi)
        self.loss_before = 0.0
        self.loss_after = 0.0
        self.training_step_outputs = []

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        # Forward pass của model gốc trả về nhiều giá trị, chúng ta chỉ cần logits cho loss này
        # Chúng ta sẽ xử lý việc này trong training_step
        return self.net(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        
        # Model của bạn trả về nhiều outputs, chúng ta chỉ lấy logits
        outputs = self(images)
        logits = outputs[0] # Giả định logits là output đầu tiên
        
        # CHANGED: Tính toán loss một cách đơn giản và trực tiếp
        loss = self.loss_fn(logits, labels)
        
        # Store for epoch end (không đổi)
        self.training_step_outputs.append(loss.detach())
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.training_step_outputs:
            epoch_loss = torch.stack(self.training_step_outputs).mean()
            self.loss_after = epoch_loss.item()
            self.training_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Giữ nguyên phần optimizer, weight_decay cũng là một dạng regularization tốt
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )

    def get_loss_before(self, dataloader: DataLoader) -> float:
        """Tính toán loss ban đầu với hàm loss đã được đơn giản hóa."""
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Model của bạn trả về nhiều outputs, chúng ta chỉ lấy logits
                outputs = self(images)
                logits = outputs[0] # Giả định logits là output đầu tiên

                # CHANGED: Tính toán loss một cách đơn giản và trực tiếp
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                num_batches += 1
        
        self.loss_before = total_loss / max(num_batches, 1)
        self.train()
        return self.loss_before

# --- Lightning Trainer Wrapper ---
class MultiGPUTrainer:
    """Simple wrapper around Lightning for multi-GPU training."""
    
    def __init__(self, model: MedicalModel, max_epochs: int = 10):
        self.model = model
        self.max_epochs = max_epochs
        
        # Auto-detect GPU setup
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            accelerator = "gpu"
            devices = num_gpus if num_gpus > 1 else 1
            strategy = "ddp" if num_gpus > 1 else "auto"
            sync_batchnorm = True if num_gpus > 1 else False
            logger.info(f"Using {num_gpus} GPU(s) with strategy: {strategy}")
        else:
            num_gpus = 0
            accelerator = "cpu"
            devices = 1
            strategy = "auto"
            sync_batchnorm = False
            logger.info("Using CPU training")
        
        self.trainer =L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            sync_batchnorm=sync_batchnorm,
            gradient_clip_val=1.0,
        )

    def fit(self, trainloader: DataLoader) -> float:
        """Simple fit function."""
        # Get loss before training
        self.model.get_loss_before(trainloader)
        
        # Train
        self.trainer.fit(self.model, train_dataloaders=trainloader)
        
        return self.model.loss_after

def train_multigpu(net: nn.Module, trainloader: DataLoader, epochs: int, learning_rate: float, config: Dict) -> Tuple[Dict, Dict]:
    """Multi-GPU training function with Lightning."""
    start_time = time.time()
    initial_params = get_weights(net)

    # Create Lightning model
    model = MedicalModel(net, learning_rate)
    trainer = MultiGPUTrainer(model, epochs)
    
    # Train
    final_loss = trainer.fit(trainloader)

    train_metrics = {
        "loss_before": model.loss_before,
        "avg_train_loss": final_loss,
        "loss_after": final_loss,
    }
    
    final_params = get_weights(net)
    delta_params = [old - new for old, new in zip(initial_params, final_params)]
    delta_norm = np.linalg.norm(np.concatenate([p.flatten() for p in delta_params]))
    compute_metrics = {"training_time": time.time() - start_time, "delta_norm": delta_norm}

    return train_metrics, compute_metrics

def evaluate_simple(net: nn.Module, valloader: DataLoader) -> Dict[str, Any]:
    """Simple evaluate function."""
    metrics = evaluate_metrics(net, valloader, DEVICE, NUM_CLASSES)
    metrics['dice_scores'] = [float(s) for s in metrics['dice_scores']]  # type: ignore[index]
    metrics['avg_foreground_dice'] = float(np.mean(metrics['dice_scores'][1:]))  # type: ignore[index]
    return metrics

# --- Main FlowerClient Class ---
class FlowerClient(NumPyClient):
    """Flower client with multi-GPU Lightning support."""
    
    def __init__(self, net: nn.Module, trainloader: DataLoader, valloader: DataLoader, config: ExperimentConfig):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.metrics_tracker = MetricsTracker(config.client_id)
        self._setup_reproducibility()

    def _setup_reproducibility(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_weights(self.net)

    def set_parameters(self, parameters: NDArrays) -> None:
        set_weights(self.net, parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        round_num = int(config.get("server_round", 0))
        self.set_parameters(parameters)

        # Use multi-GPU training
        train_metrics, compute_metrics = train_multigpu(
            self.net, self.trainloader, self.config.local_epochs,
            self.config.learning_rate, config
        )
        
        val_metrics = evaluate_simple(self.net, self.valloader)

        self.metrics_tracker.log_round(round_num, train_metrics, val_metrics, compute_metrics)
        
        client_metrics: Dict[str, Scalar] = {
            "accuracy": val_metrics["avg_foreground_dice"],
            "loss_before": train_metrics["loss_before"],
            "loss_after": train_metrics["loss_after"],
            "delta_norm": compute_metrics["delta_norm"],
        }
        
        num_examples = len(self.trainloader.dataset) if hasattr(self.trainloader, 'dataset') else 0  # type: ignore[arg-type]
        return self.get_parameters({}), num_examples, client_metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        val_metrics = evaluate_simple(self.net, self.valloader)
        
        loss = 1.0 - val_metrics["avg_foreground_dice"]
        
        metrics: Dict[str, Scalar] = {"accuracy": val_metrics["avg_foreground_dice"]}
        for i, score in enumerate(val_metrics["dice_scores"]):
            metrics[f"dice_class_{i}"] = score
            
        num_eval_examples = len(self.valloader.dataset) if hasattr(self.valloader, 'dataset') else 0  # type: ignore[arg-type]
        return loss, num_eval_examples, metrics

# --- Flower App Integration ---
def client_fn(context: Context) -> fl.client.Client:
    """Creates FlowerClient with multi-GPU support."""
    net = get_model()
    
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config.get("num_partitions", context.node_config.get("num-clients", 1)))
    
    # Read config
    partition_strategy = str(context.run_config.get("partition-strategy", "iid"))
    alpha = float(context.run_config.get("alpha", 0.5))
    
    trainloader, valloader, _ = load_data(partition_id, num_partitions, partition_strategy, alpha)
    
    local_epochs = int(context.run_config["local-epochs"])
    experiment_name = str(context.run_config.get("experiment-name", "default-experiment"))
    
    config = ExperimentConfig(
        client_id=str(partition_id),
        local_epochs=local_epochs,
        experiment_name=experiment_name,
        learning_rate=1e-4,
        seed=42
    )

    return FlowerClient(
        net=net, 
        trainloader=trainloader, 
        valloader=valloader, 
        config=config
    ).to_client()

# This defines the ClientApp that Flower will run
app = ClientApp(
    client_fn=client_fn,
)