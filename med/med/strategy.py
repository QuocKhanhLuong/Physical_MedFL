"""FL strategy for medical image segmentation."""

import flwr as fl
import json
import logging
import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict
from datetime import datetime
from flwr.common import (
    FitRes,
    EvaluateRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from pathlib import Path

# Add src to path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from utils.metrics import evaluate_metrics
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Theo dõi các chỉ số nâng cao cho mục đích nghiên cứu với comprehensive server evaluation."""
    def __init__(self, experiment_name: str, strategy_name: str):
        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        self.start_time = datetime.now()
        self.round_data: List[Dict[str, Any]] = []
        self.client_data: defaultdict = defaultdict(list)
        self.evaluation_data: List[Dict[str, Any]] = []  # NEW: Server evaluation metrics
        self.convergence_data: List[Dict[str, Any]] = []  # NEW: Convergence tracking

    def log_round_start(self, server_round: int, num_selected_clients: int, current_eta: float) -> None:
        logging.info(f"ROUND {server_round} | Strategy: {self.strategy_name}")
        logging.info(f"Selected clients: {num_selected_clients}, Learning rate: {current_eta:.6f}")

    def log_round_complete(self, server_round: int, metrics: Dict[str, Any], client_fit_res: List[Tuple[ClientProxy, FitRes]]) -> None:
        # Collect comprehensive round metrics
        round_metrics = {
            "round": server_round, 
            "timestamp": datetime.now().isoformat(),
            "fit_phase": "aggregation",
            **metrics
        }
        self.round_data.append(round_metrics)
        
        # Collect client training metrics
        client_metrics_summary = []
        for client, res in client_fit_res:
            client_data = {"round": server_round, "client_id": client.cid, **res.metrics}
            self.client_data[client.cid].append(client_data)
            client_metrics_summary.append({
                "client_id": client.cid,
                "num_examples": res.num_examples,
                "train_loss": res.metrics.get("avg_train_loss", 0.0),
                "val_dice": res.metrics.get("avg_foreground_dice", 0.0)
            })
        
        # Log aggregation results
        logging.info(f"Round {server_round} aggregation complete")
        logging.info(f"Variance: {metrics.get('variance', 0.0):.6f}, "
                    f"Adapted eta: {metrics.get('adapted_eta', 0.0):.6f}, "
                    f"Gradient norm: {metrics.get('pseudo_gradient_norm', 0.0):.4f}")
        
        if client_metrics_summary:
            avg_train_loss = np.mean([c["train_loss"] for c in client_metrics_summary])
            avg_val_dice = np.mean([c["val_dice"] for c in client_metrics_summary])
            logging.info(f"Client averages - Train loss: {avg_train_loss:.4f}, Val dice: {avg_val_dice:.4f}")

    def log_round_evaluation(self, server_round: int, evaluation_metrics: Dict[str, Any], 
                           aggregated_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Track comprehensive server evaluation metrics for each round."""
        
        eval_data = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "phase": "evaluation",
            **evaluation_metrics
        }
        
        if aggregated_metrics:
            eval_data.update({f"agg_{k}": v for k, v in aggregated_metrics.items()})
        
        self.evaluation_data.append(eval_data)
        
        # Convergence analysis
        if len(self.evaluation_data) >= 2:
            prev_eval = self.evaluation_data[-2]
            current_loss = evaluation_metrics.get("centralized_loss", 0.0)
            prev_loss = prev_eval.get("centralized_loss", 0.0)
            current_acc = evaluation_metrics.get("centralized_accuracy", 0.0)
            prev_acc = prev_eval.get("centralized_accuracy", 0.0)
            
            loss_improvement = prev_loss - current_loss
            acc_improvement = current_acc - prev_acc
            
            convergence_metrics = {
                "round": server_round,
                "timestamp": datetime.now().isoformat(),
                "loss_improvement": float(loss_improvement),
                "accuracy_improvement": float(acc_improvement),
                "loss_trend": "improving" if loss_improvement > 0 else "degrading",
                "accuracy_trend": "improving" if acc_improvement > 0 else "degrading",
                "convergence_rate": float(abs(loss_improvement)) if prev_loss > 0 else 0.0
            }
            
            self.convergence_data.append(convergence_metrics)
        
        # Log evaluation results
        logging.info(f"Round {server_round} evaluation - "
                    f"Loss: {evaluation_metrics.get('centralized_loss', 0.0):.4f}, "
                    f"Accuracy: {evaluation_metrics.get('centralized_accuracy', 0.0):.4f}")
        
        if aggregated_metrics:
            fg_dice = aggregated_metrics.get("avg_foreground_dice_clients", 0.0)
            dice_gini = aggregated_metrics.get("dice_gini_coefficient", 0.0)
            logging.info(f"Client metrics - Avg dice: {fg_dice:.4f}, Gini: {dice_gini:.4f}")
        
        # Log convergence status
        if len(self.convergence_data) >= 2:
            latest_conv = self.convergence_data[-1]
            logging.info(f"Trends - Loss: {latest_conv['loss_trend']} ({latest_conv['loss_improvement']:.4f}), "
                        f"Accuracy: {latest_conv['accuracy_trend']} ({latest_conv['accuracy_improvement']:.4f})")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary across all rounds."""
        if not self.evaluation_data:
            return {}
        
        # Extract key metrics
        losses = [e.get("centralized_loss", 0.0) for e in self.evaluation_data]
        accuracies = [e.get("centralized_accuracy", 0.0) for e in self.evaluation_data]
        fg_dices = [e.get("agg_avg_foreground_dice_clients", 0.0) for e in self.evaluation_data if "agg_avg_foreground_dice_clients" in e]
        
        summary = {
            "experiment_name": self.experiment_name,
            "strategy_name": self.strategy_name,
            "total_rounds": len(self.evaluation_data),
            "experiment_duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60.0,
            
            # Performance metrics
            "final_loss": losses[-1] if losses else 0.0,
            "best_loss": min(losses) if losses else 0.0,
            "initial_loss": losses[0] if losses else 0.0,
            "total_loss_improvement": (losses[0] - losses[-1]) if len(losses) >= 2 else 0.0,
            
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "initial_accuracy": accuracies[0] if accuracies else 0.0,
            "total_accuracy_improvement": (accuracies[-1] - accuracies[0]) if len(accuracies) >= 2 else 0.0,
            
            # Convergence analysis
            "convergence_achieved": self._assess_convergence(losses),
            "stability_score": self._calculate_stability(accuracies),
            "improvement_rate": self._calculate_improvement_rate(accuracies),
        }
        
        if fg_dices:
            summary.update({
                "final_fg_dice": fg_dices[-1],
                "best_fg_dice": max(fg_dices),
                "fg_dice_improvement": (fg_dices[-1] - fg_dices[0]) if len(fg_dices) >= 2 else 0.0
            })
        
        return summary

    def _assess_convergence(self, losses: List[float], threshold: float = 0.001) -> bool:
        """Assess if model has converged based on loss stability."""
        if len(losses) < 5:
            return False
        recent_losses = losses[-5:]
        return bool(np.std(recent_losses) < threshold)

    def _calculate_stability(self, accuracies: List[float]) -> float:
        """Calculate stability score based on accuracy variance."""
        if len(accuracies) < 3:
            return 0.0
        return float(1.0 / (1.0 + np.var(accuracies)))

    def _calculate_improvement_rate(self, accuracies: List[float]) -> float:
        """Calculate improvement rate across rounds."""
        if len(accuracies) < 2:
            return 0.0
        return float((accuracies[-1] - accuracies[0]) / len(accuracies))

    def export_research_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        if output_dir:
            export_dir = Path(output_dir)
        else:
            run_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            export_dir = Path("research_exports") / f"{self.experiment_name}_{run_timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # Export aggregation metrics
        if self.round_data:
            round_df = pd.DataFrame(self.round_data)
            round_path = export_dir / "server_aggregation_metrics.csv"
            round_df.to_csv(round_path, index=False)
            exported_files["aggregation_metrics"] = str(round_path)
        
        # Export evaluation metrics (NEW)
        if self.evaluation_data:
            eval_df = pd.DataFrame(self.evaluation_data)
            eval_path = export_dir / "server_evaluation_metrics.csv"
            eval_df.to_csv(eval_path, index=False)
            exported_files["evaluation_metrics"] = str(eval_path)
        
        # Export convergence analysis (NEW)
        if self.convergence_data:
            conv_df = pd.DataFrame(self.convergence_data)
            conv_path = export_dir / "convergence_analysis.csv"
            conv_df.to_csv(conv_path, index=False)
            exported_files["convergence_analysis"] = str(conv_path)
        
        # Export client metrics
        if self.client_data:
            all_client_data = []
            for cid, records in self.client_data.items():
                for record in records:
                    all_client_data.append({"client_id": cid, **record})
            client_df = pd.DataFrame(all_client_data)
            client_path = export_dir / "client_detailed_metrics.csv"
            client_df.to_csv(client_path, index=False)
            exported_files["client_metrics"] = str(client_path)
        
        # Export performance summary (NEW)
        summary = self.get_performance_summary()
        if summary:
            summary_path = export_dir / "experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            exported_files["experiment_summary"] = str(summary_path)
        
        logging.info(f"Research data exported to {export_dir}")
        logging.info(f"Files exported: {list(exported_files.keys())}")
        
        return exported_files

class AdaFedAdamStrategy(FedAvg):
    """
    Chiến lược hợp nhất kết hợp FAUP, AdaFedAdam, và điều phối Adaptive Loss.
    """
    
    def _default_evaluate_metrics_aggregation_fn(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """
        Default implementation of evaluate_metrics_aggregation_fn.
        This function fixes the warning about missing evaluate_metrics_aggregation_fn.
        """
        if not metrics:
            return {}
        
        # Separate metrics by type
        aggregated = {}
        metric_collections = {}
        
        for num_examples, client_metrics in metrics:
            for key, value in client_metrics.items():
                if key not in metric_collections:
                    metric_collections[key] = []
                metric_collections[key].append((num_examples, float(value)))
        
        # Weighted averaging for each metric
        for metric_name, values in metric_collections.items():
            if values:
                # Calculate weighted average
                total_examples = sum(num_examples for num_examples, _ in values)
                if total_examples > 0:
                    weighted_sum = sum(num_examples * value for num_examples, value in values)
                    aggregated[f"weighted_avg_{metric_name}"] = weighted_sum / total_examples
                
                # Calculate simple statistics
                simple_values = [value for _, value in values]
                aggregated[f"avg_{metric_name}"] = float(np.mean(simple_values))
                aggregated[f"std_{metric_name}"] = float(np.std(simple_values))
                aggregated[f"min_{metric_name}"] = float(np.min(simple_values))
                aggregated[f"max_{metric_name}"] = float(np.max(simple_values))
        
        return aggregated
    
    def __init__(
        self,
        *,
        eta: float = 1e-3,
        eta_adapt_rate: float = 1.0,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
        w_impact: float = 0.4,
        w_debt: float = 0.6,
        num_classes: int = 4,
        lambda_val: float = 15.0,
        experiment_name: str = "AdaFedAdam_Experiment",
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        **kwargs,
    ):
        # Set evaluate_metrics_aggregation_fn if not provided - FIX CIRCULAR REFERENCE
        if evaluate_metrics_aggregation_fn is None:
            evaluate_metrics_aggregation_fn = self._default_evaluate_metrics_aggregation_fn
        
        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, **kwargs)
        
        self.eta = eta
        self.initial_eta = eta
        self.eta_adapt_rate = eta_adapt_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None
        self.w_impact = w_impact
        self.w_debt = w_debt
        self.client_debts: Dict[str, int] = defaultdict(int)
        self.client_metrics_history: Dict[str, Dict[str, Any]] = {}
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.kappa_values = np.ones(num_classes) * lambda_val
        self.experiment_tracker = ExperimentTracker(experiment_name, "AdaFedAdamStrategy")
        # FIX: Initialize current_parameters properly after super().__init__()
        self.current_parameters = self.initial_parameters
        logging.info("Unified Fairness Strategy initialized with enhanced metrics aggregation")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        available_clients = client_manager.sample(num_clients=self.min_available_clients, min_num_clients=self.min_available_clients)
        if not available_clients: return []
        
        utilities = np.array([self._calculate_client_utility(client.cid) for client in available_clients])
        probabilities = utilities / np.sum(utilities) if np.sum(utilities) > 0 else None
        
        num_to_select = int(self.fraction_fit * len(available_clients))
        num_to_select = max(self.min_fit_clients, num_to_select)
        
        selected_indices = np.random.choice(len(available_clients), size=num_to_select, replace=False, p=probabilities)
        selected_clients = [available_clients[i] for i in selected_indices]
        
        selected_cids = {c.cid for c in selected_clients}
        for client in available_clients:
            if client.cid in selected_cids:
                # FIX: Reduce debt gradually instead of resetting to 0 for better fairness
                self.client_debts[client.cid] = max(0, self.client_debts[client.cid] - 2)
            else:
                self.client_debts[client.cid] = self.client_debts[client.cid] + 1
            
        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)
        config["kappa_values"] = json.dumps(self.kappa_values.tolist())
        
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]

    def _calculate_client_utility(self, cid: str) -> float:
        metrics = self.client_metrics_history.get(cid, {})
        impact = max(0.0, float(metrics.get("loss_before", 1.0)) - float(metrics.get("loss_after", 1.0)))
        debt = self.client_debts.get(cid, 0)
        return self.w_impact * impact + self.w_debt * debt

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        self.experiment_tracker.log_round_start(server_round, len(results), self.eta)
        
        for client, fit_res in results: self.client_metrics_history[client.cid] = fit_res.metrics
        
        # <<< FIX 1: Thêm kiểm tra None để đảm bảo an toàn kiểu dữ liệu >>>
        if self.current_parameters is None:
            return None, {}
        current_weights = parameters_to_ndarrays(self.current_parameters)
        
        client_updates = []
        total_examples = sum(res.num_examples for _, res in results)
        for _, res in results:
            update = [(cw - sw) * (res.num_examples / total_examples) for cw, sw in zip(parameters_to_ndarrays(res.parameters), current_weights)]
            client_updates.append(update)
            
        pseudo_gradient = [-np.sum([u[i] for u in client_updates], axis=0) for i in range(len(current_weights))]
        flat_pseudo_gradient = self._flatten_ndarrays(pseudo_gradient)
        flat_client_deltas = [self._flatten_ndarrays(u) for u in client_updates]
        variance, similarities = self.get_variance(flat_client_deltas, flat_pseudo_gradient)
        self.eta = self.adapt_lr(variance)
        
        if self.m_t is None or self.v_t is None:
            self.m_t = [np.zeros_like(p) for p in pseudo_gradient]
            self.v_t = [np.zeros_like(p) for p in pseudo_gradient]
            
        # <<< FIX 2: Thêm `assert` để Pylance hiểu rằng m_t và v_t không phải là None ở đây >>>
        assert self.m_t is not None
        assert self.v_t is not None
        
        self.m_t = [self.beta_1 * m + (1 - self.beta_1) * g for m, g in zip(self.m_t, pseudo_gradient)]
        self.v_t = [self.beta_2 * v + (1 - self.beta_2) * np.square(g) for v, g in zip(self.v_t, pseudo_gradient)]
        
        m_hat = [m / (1 - self.beta_1**server_round) for m in self.m_t]
        v_hat = [v / (1 - self.beta_2**server_round) for v in self.v_t]
        
        new_weights = [w - self.eta * m / (np.sqrt(v) + self.tau) for w, m, v in zip(current_weights, m_hat, v_hat)]
        
        self.current_parameters = ndarrays_to_parameters(new_weights)
        
        final_metrics: Dict[str, Scalar] = {"variance": float(variance), "adapted_eta": float(self.eta), "avg_cosine_similarity": float(np.mean(similarities)) if similarities else 0.0, "pseudo_gradient_norm": float(np.linalg.norm(flat_pseudo_gradient))}
        self.experiment_tracker.log_round_complete(server_round, final_metrics, results)
        
        return self.current_parameters, final_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results: return None, {}
        
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_result is None:
            return None, {}
        loss_aggregated, metrics_aggregated = aggregated_result
        
        # Aggregate detailed metrics from clients
        aggregated_metrics = self._aggregate_detailed_metrics(results)
        
        # Update kappa values based on class-specific Dice scores
        class_dice_scores: List[List[float]] = [[] for _ in range(self.num_classes)]
        for _, res in results:
            for i in range(self.num_classes):
                score = res.metrics.get(f"dice_class_{i}", -1.0)
                if score != -1.0 and isinstance(score, (int, float)): 
                    class_dice_scores[i].append(float(score))
                
        avg_class_dice = [float(np.mean(scores)) if scores else 0.0 for scores in class_dice_scores]
        # FIX: Prevent negative kappa values by clipping dice scores to [0, 1]
        clipped_dice = np.clip(avg_class_dice, 0.0, 1.0)
        self.kappa_values = self.lambda_val * (1.0 - clipped_dice)
        
        # Enhanced logging with detailed metrics
        avg_class_dice_floats = [float(x) for x in avg_class_dice]  # Ensure proper typing
        self._log_detailed_evaluation(server_round, aggregated_metrics, avg_class_dice_floats)
        
        # Combine all metrics
        final_metrics = {**metrics_aggregated, **aggregated_metrics}
        final_metrics["avg_fg_dice"] = float(np.mean(np.array(avg_class_dice[1:])))
        
        # NEW: Track server evaluation metrics with experiment tracker
        evaluation_metrics = {
            "centralized_loss": float(loss_aggregated) if loss_aggregated else 0.0,
            "centralized_accuracy": float(metrics_aggregated.get("accuracy", 0.0)),
            "avg_fg_dice": float(final_metrics["avg_fg_dice"]),
            "num_clients_evaluated": len(results),
            "kappa_values": self.kappa_values.tolist(),
            "lambda_val": self.lambda_val
        }
        
        self.experiment_tracker.log_round_evaluation(
            server_round, 
            evaluation_metrics, 
            aggregated_metrics
        )
        
        return loss_aggregated, final_metrics

    def _aggregate_detailed_metrics(self, results: List[Tuple[ClientProxy, EvaluateRes]]) -> Dict[str, Scalar]:
        """Aggregate comprehensive metrics from all clients."""
        if not results:
            return {}
        
        # Initialize metric collectors
        metrics_collector = {
            'dice_scores': [[] for _ in range(self.num_classes)],
            'iou_scores': [[] for _ in range(self.num_classes)],
            'precision_scores': [[] for _ in range(self.num_classes)],
            'recall_scores': [[] for _ in range(self.num_classes)],
            'f1_scores': [[] for _ in range(self.num_classes)],
            'foreground_dice': [],
            'foreground_iou': [],
            'foreground_precision': [],
            'foreground_recall': [],
            'foreground_f1': [],
            'dice_variance': [],
            'dice_std': [],
            'validation_samples': []
        }
        
        # Collect metrics from all clients
        for _, eval_res in results:
            metrics = eval_res.metrics
            
            # Per-class metrics
            for i in range(self.num_classes):
                if f"dice_class_{i}" in metrics:
                    metrics_collector['dice_scores'][i].append(float(metrics[f"dice_class_{i}"]))
                if f"iou_class_{i}" in metrics:
                    metrics_collector['iou_scores'][i].append(float(metrics[f"iou_class_{i}"]))
                if f"precision_class_{i}" in metrics:
                    metrics_collector['precision_scores'][i].append(float(metrics[f"precision_class_{i}"]))
                if f"recall_class_{i}" in metrics:
                    metrics_collector['recall_scores'][i].append(float(metrics[f"recall_class_{i}"]))
                if f"f1_class_{i}" in metrics:
                    metrics_collector['f1_scores'][i].append(float(metrics[f"f1_class_{i}"]))
            
            # Aggregate metrics
            if "avg_foreground_dice" in metrics:
                metrics_collector['foreground_dice'].append(float(metrics["avg_foreground_dice"]))
            if "avg_foreground_iou" in metrics:
                metrics_collector['foreground_iou'].append(float(metrics["avg_foreground_iou"]))
            if "avg_foreground_precision" in metrics:
                metrics_collector['foreground_precision'].append(float(metrics["avg_foreground_precision"]))
            if "avg_foreground_recall" in metrics:
                metrics_collector['foreground_recall'].append(float(metrics["avg_foreground_recall"]))
            if "avg_foreground_f1" in metrics:
                metrics_collector['foreground_f1'].append(float(metrics["avg_foreground_f1"]))
            if "dice_variance" in metrics:
                metrics_collector['dice_variance'].append(float(metrics["dice_variance"]))
            if "dice_std" in metrics:
                metrics_collector['dice_std'].append(float(metrics["dice_std"]))
            if "validation_samples" in metrics:
                metrics_collector['validation_samples'].append(float(metrics["validation_samples"]))
        
        # Calculate aggregated statistics
        aggregated = {}
        
        # Per-class aggregations
        for i in range(self.num_classes):
            if metrics_collector['dice_scores'][i]:
                dice_vals = metrics_collector['dice_scores'][i]
                aggregated[f"avg_dice_class_{i}"] = float(np.mean(dice_vals))
                aggregated[f"std_dice_class_{i}"] = float(np.std(dice_vals))
                aggregated[f"min_dice_class_{i}"] = float(np.min(dice_vals))
                aggregated[f"max_dice_class_{i}"] = float(np.max(dice_vals))
            
            if metrics_collector['iou_scores'][i]:
                iou_vals = metrics_collector['iou_scores'][i]
                aggregated[f"avg_iou_class_{i}"] = float(np.mean(iou_vals))
                aggregated[f"std_iou_class_{i}"] = float(np.std(iou_vals))
            
            if metrics_collector['precision_scores'][i]:
                prec_vals = metrics_collector['precision_scores'][i]
                aggregated[f"avg_precision_class_{i}"] = float(np.mean(prec_vals))
                aggregated[f"std_precision_class_{i}"] = float(np.std(prec_vals))
            
            if metrics_collector['recall_scores'][i]:
                recall_vals = metrics_collector['recall_scores'][i]
                aggregated[f"avg_recall_class_{i}"] = float(np.mean(recall_vals))
                aggregated[f"std_recall_class_{i}"] = float(np.std(recall_vals))
        
        # Overall aggregations with comprehensive statistics
        metric_types = [
            ('foreground_dice', 'Dice'),
            ('foreground_iou', 'IoU'),
            ('foreground_precision', 'Precision'),
            ('foreground_recall', 'Recall'),
            ('foreground_f1', 'F1')
        ]
        
        for metric_key, metric_name in metric_types:
            if metrics_collector[metric_key]:
                values = metrics_collector[metric_key]
                prefix = f"avg_{metric_key}_clients"
                aggregated[prefix] = float(np.mean(values))
                aggregated[f"std_{metric_key}_clients"] = float(np.std(values))
                aggregated[f"min_{metric_key}_clients"] = float(np.min(values))
                aggregated[f"max_{metric_key}_clients"] = float(np.max(values))
                aggregated[f"median_{metric_key}_clients"] = float(np.median(values))
        
        # Client diversity and fairness metrics
        if metrics_collector['dice_variance']:
            dice_var = metrics_collector['dice_variance']
            aggregated["avg_dice_variance"] = float(np.mean(dice_var))
            aggregated["max_dice_variance"] = float(np.max(dice_var))
        
        if metrics_collector['validation_samples']:
            val_samples = metrics_collector['validation_samples']
            aggregated["total_validation_samples"] = float(np.sum(val_samples))
            aggregated["avg_validation_samples_per_client"] = float(np.mean(val_samples))
            aggregated["std_validation_samples_per_client"] = float(np.std(val_samples))
        
        # Performance statistics
        aggregated["num_clients_evaluated"] = len(results)
        
        # Calculate cross-client performance fairness
        if metrics_collector['foreground_dice']:
            fg_dice = metrics_collector['foreground_dice']
            # Gini coefficient for fairness measurement
            aggregated["dice_gini_coefficient"] = self.get_gini(fg_dice)
            # Performance gap (max - min)
            aggregated["dice_performance_gap"] = float(np.max(fg_dice) - np.min(fg_dice))
        
        return aggregated

    def _log_detailed_evaluation(self, server_round: int, aggregated_metrics: Dict[str, Scalar], avg_class_dice: List[float]) -> None:
        """Enhanced logging for evaluation results."""
        logging.info(f"ROUND {server_round} EVALUATION RESULTS")
        
        # Class-specific Performance Table
        logging.info("PER-CLASS PERFORMANCE:")
        class_names = ["Background", "RV", "Myocardium", "LV"]  # ACDC dataset classes
        logging.info(f"{'Class':<12} | {'Dice':<15} | {'IoU':<15} | {'Precision':<15} | {'Recall':<15} | {'Kappa':<10}")
        logging.info("-" * 95)
        
        for i, class_name in enumerate(class_names):
            dice_avg = aggregated_metrics.get(f"avg_dice_class_{i}", 0.0)
            dice_std = aggregated_metrics.get(f"std_dice_class_{i}", 0.0)
            iou_avg = aggregated_metrics.get(f"avg_iou_class_{i}", 0.0)
            iou_std = aggregated_metrics.get(f"std_iou_class_{i}", 0.0)
            prec_avg = aggregated_metrics.get(f"avg_precision_class_{i}", 0.0)
            prec_std = aggregated_metrics.get(f"std_precision_class_{i}", 0.0)
            recall_avg = aggregated_metrics.get(f"avg_recall_class_{i}", 0.0)
            recall_std = aggregated_metrics.get(f"std_recall_class_{i}", 0.0)
            kappa = self.kappa_values[i] if i < len(self.kappa_values) else 0.0
            
            logging.info(
                f"{class_name:<12} | "
                f"{dice_avg:.3f}±{dice_std:.3f}    | "
                f"{iou_avg:.3f}±{iou_std:.3f}    | "
                f"{prec_avg:.3f}±{prec_std:.3f}    | "
                f"{recall_avg:.3f}±{recall_std:.3f}    | "
                f"{kappa:.3f}"
            )
        
        # Overall Performance Summary
        logging.info("FOREGROUND SUMMARY:")
        metrics_summary = [
            ("Dice", "avg_foreground_dice_clients"),
            ("IoU", "avg_foreground_iou_clients"),
            ("Precision", "avg_foreground_precision_clients"),
            ("Recall", "avg_foreground_recall_clients"),
            ("F1", "avg_foreground_f1_clients")
        ]
        
        for metric_name, metric_key in metrics_summary:
            if metric_key in aggregated_metrics:
                avg_val = float(aggregated_metrics[metric_key])
                std_val = float(aggregated_metrics.get(f"std_{metric_key.replace('avg_', '')}", 0.0))
                min_val = float(aggregated_metrics.get(f"min_{metric_key.replace('avg_', '')}", 0.0))
                max_val = float(aggregated_metrics.get(f"max_{metric_key.replace('avg_', '')}", 0.0))
                median_val = float(aggregated_metrics.get(f"median_{metric_key.replace('avg_', '')}", 0.0))
                
                logging.info(
                    f"{metric_name:<10}: {avg_val:.4f} ± {std_val:.4f} "
                    f"| Range: [{min_val:.4f}, {max_val:.4f}] | Median: {median_val:.4f}"
                )
        
        # Fairness and Distribution Analysis
        logging.info("FAIRNESS ANALYSIS:")
        num_clients = int(aggregated_metrics.get("num_clients_evaluated", 0))
        dice_gini = float(aggregated_metrics.get("dice_gini_coefficient", 0.0))
        dice_gap = float(aggregated_metrics.get("dice_performance_gap", 0.0))
        avg_variance = float(aggregated_metrics.get("avg_dice_variance", 0.0))
        max_variance = float(aggregated_metrics.get("max_dice_variance", 0.0))
        
        logging.info(f"Clients: {num_clients}, Gini: {dice_gini:.4f}, Gap: {dice_gap:.4f}")
        logging.info(f"Dice variance - Avg: {avg_variance:.6f}, Max: {max_variance:.6f}")
        
        # Data Distribution
        total_samples = float(aggregated_metrics.get("total_validation_samples", 0))
        avg_samples = float(aggregated_metrics.get("avg_validation_samples_per_client", 0))
        std_samples = float(aggregated_metrics.get("std_validation_samples_per_client", 0))
        
        if total_samples > 0:
            logging.info(f"Data - Total samples: {int(total_samples)}, "
                        f"Avg per client: {avg_samples:.1f} ± {std_samples:.1f}")
        
        # Adaptive System Status
        logging.info("ADAPTIVE PARAMETERS:")
        for i, (class_name, kappa) in enumerate(zip(class_names, self.kappa_values)):
            status = "HIGH" if kappa > 20 else "MED" if kappa > 10 else "LOW"
            logging.info(f"{class_name}: κ={kappa:.3f} ({status})")
        
        # Performance tier
        if "avg_foreground_dice_clients" in aggregated_metrics:
            overall_performance = float(aggregated_metrics["avg_foreground_dice_clients"])
            performance_tier = (
                "Excellent" if overall_performance > 0.8 else
                "Very Good" if overall_performance > 0.6 else
                "Good" if overall_performance > 0.4 else
                "Fair" if overall_performance > 0.2 else
                "Needs Improvement"
            )
            logging.info(f"Round {server_round} performance: {performance_tier} ({overall_performance:.4f})")

    def _flatten_ndarrays(self, ndarrays: NDArrays) -> np.ndarray:
        return np.concatenate([arr.flatten() for arr in ndarrays])

    def get_variance(self, client_deltas: List[np.ndarray], pseudo_gradient: np.ndarray) -> Tuple[float, List[float]]:
        if not client_deltas:
            return 0.0, []
        
        pseudo_norm = np.linalg.norm(pseudo_gradient)
        # FIX: Use smaller threshold and add fallback for very small gradients
        min_threshold = max(self.tau, 1e-12)
        
        if pseudo_norm < min_threshold:
            # When pseudo gradient is very small, return small variance to maintain learning
            return 0.01, []
            
        similarities = []
        for delta in client_deltas:
            delta_norm = np.linalg.norm(delta)
            if delta_norm > min_threshold:
                sim = np.dot(delta, pseudo_gradient) / (delta_norm * pseudo_norm)
                # Clip similarity to [-1, 1] for numerical stability
                sim = np.clip(sim, -1.0, 1.0)
                similarities.append(sim)
        
        if not similarities:
            return 0.01, []  # Small variance instead of 0 to maintain adaptation
            
        variance = float(np.var(similarities))
        return max(variance, 1e-8), similarities  # Ensure minimum variance

    def adapt_lr(self, variance: float) -> float:
        return self.initial_eta / (1.0 + self.eta_adapt_rate * variance)

    def export_research_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        exported_files = self.experiment_tracker.export_research_data(output_dir)
        export_dir = Path(exported_files["server_metrics"]).parent if "server_metrics" in exported_files else Path(output_dir or "research_exports")
        debt_data = [{"client_id": cid, "final_debt": debt} for cid, debt in self.client_debts.items()]
        if debt_data: pd.DataFrame(debt_data).to_csv(export_dir / "faup_final_debts.csv", index=False)
        return exported_files

    def get_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient to measure performance inequality across clients."""
        if len(values) <= 1:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        return float(gini)
