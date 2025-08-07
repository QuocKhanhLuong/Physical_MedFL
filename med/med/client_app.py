"""med: A Flower / PyTorch app for medical image segmentation."""

import copy
import json
import logging
import math
import numpy as np
import pandas as pd
import sys
import os
import time
import torch
import torch.nn as nn
import uuid

from collections import OrderedDict
from datetime import datetime
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from numpy.typing import NDArray
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from .task import get_model, get_weights, load_data, set_weights, test, train
from utils.losses import Adaptive_tvmf_dice_loss, DynamicLossWeighter, CombinedLoss
from utils.metrics import evaluate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentConfig:
    """Research-grade experiment configuration management."""

    def __init__(
        self,
        experiment_name: str,
        client_id: str,
        learning_rate: float = 1e-4,
        local_epochs: int = 1,
        batch_size: int = 8,
        seed: int = 42,
        data_heterogeneity: str = "iid",
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.data_heterogeneity = data_heterogeneity
        self.created_at = datetime.now()
        self.config_id = str(uuid.uuid4())[:8]

        # Additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            key: str(value) if isinstance(value, datetime) else value
            for key, value in self.__dict__.items()
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsTracker:
    """Advanced metrics tracking for research purposes."""

    def __init__(self, client_id: str, experiment_config: ExperimentConfig):
        self.client_id = client_id
        self.experiment_config = experiment_config
        self.round_metrics: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        self.convergence_data: List[Dict[str, Any]] = []

    def log_round_start(self, server_round: int, global_parameters_norm: float) -> None:
        """Log round start information."""
        round_data = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "global_parameters_norm": global_parameters_norm,
            "client_id": self.client_id
        }
        logger.info(f"Client {self.client_id} - Round {server_round} started")

    def log_training_metrics(
        self,
        server_round: int,
        training_metrics: Dict,
        validation_metrics: Dict,
        computational_metrics: Dict
    ) -> None:
        """Log comprehensive metrics for a round."""
        round_data = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "client_id": self.client_id,
            "training": training_metrics,
            "validation": validation_metrics,
            "computational": computational_metrics
        }
        self.round_metrics.append(round_data)

        # Log summary to console
        avg_train_loss = training_metrics.get("avg_train_loss", 0)
        avg_fg_dice = validation_metrics.get("avg_foreground_dice", 0)
        
        logger.info(
            f"Client {self.client_id} | Round {server_round} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {avg_fg_dice:.4f}"
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        if not self.round_metrics:
            return {}

        # Extract time series data
        train_losses = [r["training"]["avg_train_loss"] for r in self.round_metrics]
        val_dices = [r["validation"]["avg_foreground_dice"] for r in self.round_metrics]
        training_times = [r["computational"]["training_time_seconds"] for r in self.round_metrics]

        summary = {
            "client_id": self.client_id,
            "total_rounds": len(self.round_metrics),
            "experiment_duration_minutes": sum(training_times) / 60.0,
            
            # Training loss statistics
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "best_train_loss": min(train_losses) if train_losses else 0.0,
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "train_loss_std": float(np.std(train_losses)) if train_losses else 0.0,
            
            # Validation dice statistics
            "final_val_dice": val_dices[-1] if val_dices else 0.0,
            "best_val_dice": max(val_dices) if val_dices else 0.0,
            "avg_val_dice": float(np.mean(val_dices)) if val_dices else 0.0,
            "val_dice_std": float(np.std(val_dices)) if val_dices else 0.0,
            
            # Performance metrics
            "avg_training_time": float(np.mean(training_times)) if training_times else 0.0,
            "total_training_time": sum(training_times),
            
            # Convergence metrics
            "convergence_rate": self._calculate_convergence_rate(),
            "stability_score": self._calculate_stability_score()
        }

        return summary

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on loss improvement."""
        train_losses = [r["training"]["avg_train_loss"] for r in self.round_metrics 
                       if "avg_train_loss" in r["training"]]

        if len(train_losses) < 3:
            return 0.0

        # Fit exponential decay to loss progression
        rounds = np.arange(len(train_losses))
        try:
            coeffs = np.polyfit(rounds, np.log(np.array(train_losses) + 1e-8), 1)
            return -coeffs[0]  # Negative slope indicates convergence rate
        except:
            return 0.0

    def _calculate_stability_score(self) -> float:
        """Calculate stability score based on metric variance."""
        val_dices = [r["validation"]["avg_foreground_dice"] for r in self.round_metrics
                     if "avg_foreground_dice" in r["validation"]]

        if len(val_dices) < 2:
            return 0.0

        return float(1.0 / (1.0 + np.var(val_dices)))

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export metrics to pandas DataFrame for analysis."""
        flattened_data = []

        for round_data in self.round_metrics:
            flat_data = {
                "round": round_data["round"],
                "timestamp": round_data["timestamp"],
                "client_id": round_data["client_id"]
            }

            # Flatten training metrics
            for key, value in round_data["training"].items():
                flat_data[f"train_{key}"] = value

            # Flatten validation metrics
            for key, value in round_data["validation"].items():
                if isinstance(value, list):
                    flat_data[f"val_{key}"] = str(value)
                else:
                    flat_data[f"val_{key}"] = value

            # Flatten computational metrics
            for key, value in round_data["computational"].items():
                flat_data[f"comp_{key}"] = value

            flattened_data.append(flat_data)

        return pd.DataFrame(flattened_data)


class FlowerClient(NumPyClient):
    """Research-grade Flower client with comprehensive tracking and FAUP support."""

    def __init__(
        self,
        net,
        trainloader: DataLoader,
        valloader: DataLoader,
        experiment_config: ExperimentConfig,
        enable_detailed_logging: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        self.net = net.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.experiment_config = experiment_config
        self.enable_detailed_logging = enable_detailed_logging
        self.save_checkpoints = save_checkpoints

        # Setup checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("checkpoints") / experiment_config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.metrics_tracker = MetricsTracker(
            experiment_config.client_id,
            experiment_config
        )

        # Setup reproducibility
        self._setup_reproducibility()

        # Initialize state
        self.initial_loss: Optional[float] = None
        self.round_count: int = 0
        self.best_val_metric: float = 0.0
        self.training_start_time: Optional[float] = None

        logger.info(
            f"Client {experiment_config.client_id} initialized "
            f"with {sum(p.numel() for p in net.parameters())} parameters"
        )

    def _setup_reproducibility(self) -> None:
        """Setup reproducible training environment."""
        torch.manual_seed(self.experiment_config.seed)
        np.random.seed(self.experiment_config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment_config.seed)
            torch.cuda.manual_seed_all(self.experiment_config.seed)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Extract model parameters as NDArrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Update model parameters from NDArrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def _calculate_model_delta(self, old_parameters: NDArrays, new_parameters: NDArrays) -> NDArrays:
        """Calculate delta weights (local update)."""
        return [old_param - new_param for old_param, new_param in
                zip(old_parameters, new_parameters)]

    def _calculate_computational_metrics(self, start_time: float, delta_params: NDArrays) -> Dict[str, Any]:
        """Calculate computational and efficiency metrics."""
        training_time = time.time() - start_time

        # Calculate gradient norms per layer
        grad_norms = [np.linalg.norm(param.flatten()).item() for param in delta_params]

        # Calculate parameter statistics
        flat_delta = np.concatenate([param.flatten() for param in delta_params])

        return {
            "training_time_seconds": training_time,
            "delta_norm": np.linalg.norm(flat_delta).item(),
            "grad_norms_per_layer": grad_norms,
            "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
            "param_update_magnitude": float(np.mean(np.abs(flat_delta))),
            "num_parameters": len(flat_delta)
        }

    def _train_local_advanced(self, epochs: int, kappa_values: Optional[List[float]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Advanced local training with comprehensive metrics."""
        start_time = time.time()
        initial_params = self.get_parameters({})

        # Setup loss function - prioritize CombinedLoss for physics-informed training
        # Use CombinedLoss for physics-informed model
        criterion = CombinedLoss(num_classes=NUM_CLASSES, 
                                in_channels_maxwell=1024,
                                NUM_CLASSES=4,
                                lambda_val=15.0,
                                initial_loss_weights=[0.3, 0.5, 0.5, 1.0]
                                ).to(DEVICE)

        def calculate_loss(outputs, labels):
            """Calculate loss based on available loss type."""
            # Handle tuple outputs from RobustMedVFL_UNet
            if isinstance(outputs, tuple):
                logits, maxwell_outputs = outputs
            else:
                logits = outputs
                maxwell_outputs = None
            
            if use_advanced_loss and criterion is not None:
                # Use CombinedLoss with physics components
                if maxwell_outputs is not None:
                    # Use physics loss with Maxwell outputs
                    b1 = logits  # Use logits as B1 field approximation
                    all_es = maxwell_outputs  # Maxwell solver outputs
                    feat_sm = logits  # Use logits for smoothness regularization
                    return criterion(logits, labels, b1=b1, all_es=all_es, feat_sm=feat_sm)
                else:
                    # Use loss without physics components
                    return criterion(logits, labels)
            else:
                # Dynamic weighted loss calculation
                assert base_criterion is not None and dynamic_weighter is not None
                base_losses = base_criterion(logits, labels)
                class_losses = []
                for c in range(NUM_CLASSES):
                    class_mask = (labels == c)
                    if class_mask.sum() > 0:
                        class_losses.append(base_losses[class_mask].mean())
                    else:
                        class_losses.append(torch.tensor(0.0, device=DEVICE))
                return dynamic_weighter(class_losses)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.experiment_config.learning_rate
        )
        
        # Training metrics
        training_metrics = {
            "epochs_completed": 0,
            "avg_train_loss": 0.0,
            "total_batches": 0,
            "loss_per_epoch": [],
            "learning_rate": self.experiment_config.learning_rate,
            "loss_before": 0.0,
            "loss_after": 0.0
        }

        # Calculate loss before training (for FAUP)
        self.net.eval()
        total_loss_before = 0.0
        num_batches_eval = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.net(images)
                loss = calculate_loss(outputs, labels)
                total_loss_before += loss.item()
                num_batches_eval += 1

        loss_before = total_loss_before / num_batches_eval if num_batches_eval > 0 else 0.0
        training_metrics["loss_before"] = loss_before

        # Training loop
        self.net.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = calculate_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                total_loss += loss.item()
                total_batches += 1

            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            training_metrics["loss_per_epoch"].append(avg_epoch_loss)
            training_metrics["epochs_completed"] = epoch + 1

            if self.enable_detailed_logging:
                logger.debug(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

        # Calculate loss after training (for FAUP)
        self.net.eval()
        total_loss_after = 0.0
        num_batches_eval = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.net(images)
                loss = calculate_loss(outputs, labels)
                total_loss_after += loss.item()
                num_batches_eval += 1

        loss_after = total_loss_after / num_batches_eval if num_batches_eval > 0 else 0.0
        training_metrics["loss_after"] = loss_after

        # Final training metrics
        final_epoch_loss = training_metrics["loss_per_epoch"][-1] if training_metrics["loss_per_epoch"] else 0.0
        loss_reduction = self._calculate_loss_reduction(training_metrics["loss_per_epoch"])

        training_metrics.update({
            "avg_train_loss": total_loss / total_batches if total_batches > 0 else 0.0,
            "total_batches": total_batches,
            "final_epoch_loss": final_epoch_loss,
            "loss_reduction": loss_reduction,
            "loss_improvement": max(0.0, loss_before - loss_after)
        })

        # Calculate computational metrics
        final_params = self.get_parameters({})
        delta_params = self._calculate_model_delta(initial_params, final_params)
        computational_metrics = self._calculate_computational_metrics(start_time, delta_params)

        return training_metrics, computational_metrics

    def _calculate_loss_reduction(self, loss_per_epoch: List[float]) -> float:
        """Calculate loss reduction rate across epochs."""
        if len(loss_per_epoch) < 2:
            return 0.0

        initial_loss = loss_per_epoch[0]
        final_loss = loss_per_epoch[-1]

        if initial_loss == 0:
            return 0.0

        return (initial_loss - final_loss) / initial_loss

    def _evaluate_local_advanced(self) -> Dict[str, Any]:
        """Advanced local evaluation with comprehensive metrics."""
        metrics = evaluate_metrics(self.net, self.valloader, DEVICE, NUM_CLASSES)

        # Calculate comprehensive validation metrics
        fg_dice_scores = metrics['dice_scores'][1:] if NUM_CLASSES > 1 else metrics['dice_scores']
        avg_fg_dice = float(np.mean(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0

        validation_metrics = {
            "avg_foreground_dice": avg_fg_dice,
            "dice_scores": [float(x) for x in metrics['dice_scores']],
            "iou_scores": [float(x) for x in metrics['iou']],
            "precision_scores": [float(x) for x in metrics['precision']],
            "recall_scores": [float(x) for x in metrics['recall']],
            "f1_scores": [float(x) for x in metrics['f1_score']],

            # Aggregate metrics
            "avg_foreground_iou": float(np.mean(metrics['iou'][1:]) if NUM_CLASSES > 1 else metrics['iou'][0]),
            "avg_foreground_precision": float(np.mean(metrics['precision'][1:]) if NUM_CLASSES > 1 else metrics['precision'][0]),
            "avg_foreground_recall": float(np.mean(metrics['recall'][1:]) if NUM_CLASSES > 1 else metrics['recall'][0]),
            "avg_foreground_f1": float(np.mean(metrics['f1_score'][1:]) if NUM_CLASSES > 1 else metrics['f1_score'][0]),

            # Statistical metrics
            "dice_variance": float(np.var(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0,
            "dice_std": float(np.std(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0,
            "min_dice": float(np.min(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0,
            "max_dice": float(np.max(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0
        }

        return validation_metrics

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Research-grade fit method with FAUP support."""
        self.round_count += 1
        server_round = int(config.get("server_round", self.round_count))

        # Extract kappa values for adaptive loss
        kappa_values = config.get("kappa_values", None)
        kappa_list = None
        if kappa_values:
            try:
                # Convert JSON string back to list
                if isinstance(kappa_values, str):
                    kappa_list = json.loads(kappa_values)
                elif isinstance(kappa_values, list):
                    kappa_list = kappa_values
                kappa_list = [float(k) for k in kappa_list] if kappa_list else None
            except (json.JSONDecodeError, TypeError, ValueError):
                kappa_list = None

         # Set global parameters
        self.set_parameters(parameters)

         # Perform advanced local training
        training_metrics, computational_metrics = self._train_local_advanced(
            epochs=self.experiment_config.local_epochs,
            kappa_values=kappa_list
        )

        # Perform local evaluation
        validation_metrics = self._evaluate_local_advanced()

        # Track comprehensive metrics
        self.metrics_tracker.log_training_metrics(
            server_round, training_metrics, validation_metrics, computational_metrics
        )

        # Get updated parameters
        new_parameters = self.get_parameters({})

        # Get dataset size safely  
        num_examples = getattr(self.trainloader.dataset, '__len__', lambda: len(self.trainloader) * 8)()

        # Prepare client metrics for server (including FAUP metrics)
        client_metrics: Dict[str, Scalar] = {
            "avg_train_loss": training_metrics["avg_train_loss"],
            "avg_foreground_dice": validation_metrics["avg_foreground_dice"],
            "training_time": computational_metrics["training_time_seconds"],
            "delta_norm": computational_metrics["delta_norm"],
            
            # FAUP metrics: Impact calculation (loss_before - loss_after)
            "loss_before": training_metrics["loss_before"],
            "loss_after": training_metrics["loss_after"]
        }

        return new_parameters, num_examples, client_metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Research-grade evaluate method."""

        # Update adaptive loss parameters from server if provided
        if "kappa_values" in config:
            try:
                kappa_values = config["kappa_values"]
                if isinstance(kappa_values, str):
                    kappa_values = json.loads(kappa_values)
                setattr(self.experiment_config, 'kappa_values', kappa_values)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        self.set_parameters(parameters)
        validation_metrics = self._evaluate_local_advanced()

        loss = 1.0 - validation_metrics["avg_foreground_dice"]

        # Safely get dataset size
        num_examples = getattr(self.valloader.dataset, '__len__', lambda: len(self.valloader) * 8)()

        # Enhanced metrics for server aggregation
        metrics: Dict[str, Scalar] = {
            # Basic metrics
            "accuracy": validation_metrics["avg_foreground_dice"],
            "avg_foreground_dice": validation_metrics["avg_foreground_dice"],
            "avg_foreground_iou": validation_metrics["avg_foreground_iou"],
            "avg_foreground_precision": validation_metrics["avg_foreground_precision"],
            "avg_foreground_recall": validation_metrics["avg_foreground_recall"],
            "avg_foreground_f1": validation_metrics["avg_foreground_f1"],
            
            # Statistical metrics
            "dice_variance": validation_metrics["dice_variance"],
            "dice_std": validation_metrics["dice_std"],
            "min_dice": validation_metrics["min_dice"],
            "max_dice": validation_metrics["max_dice"],
            
            # Performance diversity
            "client_id_numeric": hash(self.experiment_config.client_id) % 1000,  # For tracking
            "validation_samples": num_examples
        }
        
        # Add per-class Dice scores for Adaptive Loss kappa updates
        dice_scores = validation_metrics["dice_scores"]
        for i in range(len(dice_scores)):
            metrics[f"dice_class_{i}"] = float(dice_scores[i])
        
        # Add per-class IoU scores for detailed analysis
        iou_scores = validation_metrics["iou_scores"]
        for i in range(len(iou_scores)):
            metrics[f"iou_class_{i}"] = float(iou_scores[i])
        
        # Add per-class precision/recall for completeness
        precision_scores = validation_metrics["precision_scores"]
        recall_scores = validation_metrics["recall_scores"]
        f1_scores = validation_metrics["f1_scores"]
        
        for i in range(len(precision_scores)):
            metrics[f"precision_class_{i}"] = float(precision_scores[i])
            metrics[f"recall_class_{i}"] = float(recall_scores[i])
            metrics[f"f1_class_{i}"] = float(f1_scores[i])

        return loss, num_examples, metrics

    def export_research_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export comprehensive research data for publication."""
        if output_dir:
            export_dir = Path(output_dir)
        else:
            export_dir = Path("research_exports") / self.experiment_config.experiment_name

        export_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export experiment configuration
        config_path = export_dir / f"config_{self.experiment_config.client_id}.json"
        self.experiment_config.save(config_path)
        exported_files["config"] = str(config_path)

        # Export metrics DataFrame
        df = self.metrics_tracker.export_to_dataframe()
        metrics_path = export_dir / f"metrics_{self.experiment_config.client_id}.csv"
        df.to_csv(metrics_path, index=False)
        exported_files["metrics"] = str(metrics_path)

        # Export summary statistics
        summary = self.metrics_tracker.get_summary_statistics()
        summary_path = export_dir / f"summary_{self.experiment_config.client_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        exported_files["summary"] = str(summary_path)

        # Export convergence data
        convergence_df = pd.DataFrame(self.metrics_tracker.convergence_data)
        convergence_path = export_dir / f"convergence_{self.experiment_config.client_id}.csv"
        convergence_df.to_csv(convergence_path, index=False)
        exported_files["convergence"] = str(convergence_path)

        logger.info(f"Research data exported to {export_dir}")
        return exported_files


def client_fn(context: Context):
    """Create and return a FlowerClient instance."""
    # Load model
    net = get_model()
    
    # Get partition info from context
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    # Load data for this client
    trainloader, valloader, testloader = load_data(partition_id, num_partitions)
    
    # Get training config
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config.get("learning-rate", 1e-3))
    experiment_name = str(context.run_config.get("experiment-name", "ACDC_Medical_FL"))
    
    # Create experiment config
    config = ExperimentConfig(
        experiment_name=experiment_name,
        client_id=str(partition_id),
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        batch_size=8,
        data_heterogeneity="iid"
    )

    # Get dataset size safely
    dataset_size = getattr(trainloader.dataset, '__len__', lambda: 'unknown')()

    print(f"Initializing client {partition_id}/{num_partitions} with {dataset_size} training samples")

    # Return Client instance
    return FlowerClient(
        net=net, 
        trainloader=trainloader, 
        valloader=valloader, 
        experiment_config=config,
        enable_detailed_logging=True,
        save_checkpoints=True
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
