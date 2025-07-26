import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, List

class Adaptive_tvmf_dice_loss(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            lambda_val: float = 15.0,
            kappa_values=None,
            epsilon: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.epsilon = epsilon
        if kappa_values is not None:
            self.register_buffer('kappa_values', torch.tensor(kappa_values, dtype=torch.float32))
        else:
            self.register_buffer('kappa_values', torch.ones(num_classes) * lambda_val)

    def update_kappa_values(self, new_kappa_values) -> None:
        if isinstance(new_kappa_values, (list, np.ndarray)):
            new_kappa_values = torch.tensor(new_kappa_values, dtype=torch.float32)
        device = self.kappa_values.device
        self.kappa_values.data = new_kappa_values.to(device)

    def t_vmf_similarity(self, cos_theta, kappa):
        kappa = F.relu(kappa) + self.epsilon
        return torch.exp(kappa * (cos_theta - 1))
    
    def compute_dice_coefficient(self, pred, target):
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return dice
    
    def forward(self, inputs, targets):
        if inputs.dim() == 4:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.softmax(inputs, dim=-1)
        if targets.dim() == 3:
            targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()
        total_loss = 0.0
        class_losses = []
        for class_idx in range(self.num_classes):
            pred_class = inputs[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            pred_flat = pred_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)
            if torch.sum(target_flat) < self.epsilon:
                class_losses.append(torch.tensor(0.0, device=inputs.device))
                continue
            cos_theta = F.cosine_similarity(pred_flat.unsqueeze(0), target_flat.unsqueeze(0), dim=1, eps=self.epsilon).squeeze()
            kappa_tensor = getattr(self, 'kappa_values')
            kappa = kappa_tensor[class_idx]
            similarity = self.t_vmf_similarity(cos_theta, kappa)
            dice_coeff = self.compute_dice_coefficient(pred_class, target_class)
            tvmf_loss = 1.0 - similarity
            dice_loss = 1.0 - dice_coeff
            class_loss = tvmf_loss + dice_loss
            class_losses.append(class_loss)
            total_loss += class_loss
        avg_loss = total_loss / self.num_classes
        self.last_class_losses = torch.stack(class_losses)
        return avg_loss
    
    def get_class_losses(self) -> Any:
        if hasattr(self, 'last_class_losses'):
            return self.last_class_losses.detach().cpu().numpy()
        return np.zeros(self.num_classes)
    
    def get_adaptive_info(self) -> Any:
        kappa_tensor = getattr(self, 'kappa_values')
        return {'kappa_values': kappa_tensor.detach().cpu().numpy().tolist(), 'lambda_val': self.lambda_val, 'num_classes': self.num_classes}


class PhysicsLoss(nn.Module):
    def __init__(self, in_channels_solver):
        super().__init__()
        self.ms = MaxwellSolver(in_channels_solver)
    def forward(self, b1, eps, sig):
        b, e, s = b1.to(DEVICE), eps.to(DEVICE), sig.to(DEVICE)
        return torch.mean(self.ms.compute_helmholtz_residual(b, e, s))

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(dy) + torch.mean(dx)
    

class DynamicLossWeighter(nn.Module):
    """Dynamically adjusts weights for multiple loss components (e.g., CE, Dice, Physics)."""
    def __init__(self, num_losses: int, tau: float = 1.0, initial_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_losses = num_losses
        self.tau = tau
        if initial_weights:
            assert len(initial_weights) == num_losses, "Number of initial weights must be equal to num_losses"
            weights = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            weights = torch.ones(num_losses, dtype=torch.float32)
        self.log_vars = nn.Parameter(torch.log(weights))

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        """Calculates the total weighted loss."""
        if not isinstance(individual_losses, torch.Tensor):
            individual_losses = torch.stack(individual_losses)
        assert individual_losses.dim() == 1 and individual_losses.size(0) == self.num_losses, \
            f"Input individual_losses must be a 1D tensor of size {self.num_losses}"
        total_loss = 0.0
        for i in range(self.num_losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss_term = precision * individual_losses[i] + self.log_vars[i]
            total_loss += weighted_loss_term
        return total_loss

    def get_current_weights(self) -> Dict[str, float]:
        """Gets the current weights (calculated as exp(-log_var)) for monitoring."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            return {f"weight_{i}": w.item() for i, w in enumerate(weights)}


class ClassWeightUpdater(nn.Module):
    """
    Dynamically adjusts class weights for CrossEntropyLoss based on class-wise Dice scores.
    Uses an Exponential Moving Average (EMA) to stabilize weight updates.
    """
    def __init__(self, num_classes: int, alpha: float = 0.9, epsilon: float = 1e-6):
        """
        Args:
            num_classes (int): Number of segmentation classes.
            alpha (float): Smoothing factor for EMA. Higher alpha means slower updates.
            epsilon (float): Small value to prevent division by zero.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.epsilon = epsilon
        # Register EMA of Dice scores as a buffer to move it with the module (e.g., .to(device))
        self.register_buffer('ema_dice_scores', torch.ones(num_classes))

    def _calculate_per_class_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates Dice score for each class."""
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for i in range(self.num_classes):
            pred_class = probs[:, i, :, :]
            target_class = targets_one_hot[:, i, :, :]
            
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
            dice_scores.append(dice)
            
        return torch.stack(dice_scores)

    def update_and_get_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Updates EMA and returns new class weights based on inverse Dice performance.
        Args:
            logits (torch.Tensor): Raw output from the model (detached).
            targets (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: New weights for CrossEntropyLoss.
        """
        with torch.no_grad():
            current_dice = self._calculate_per_class_dice(logits, targets)
            
            # Update EMA of Dice scores
            self.ema_dice_scores = self.alpha * self.ema_dice_scores + (1 - self.alpha) * current_dice
            
            # Calculate weights as the inverse of the smoothed Dice scores
            inverse_scores = 1.0 / (self.ema_dice_scores + self.epsilon)
            
            # Normalize weights to keep the overall loss magnitude stable
            # Here, we normalize them to sum to num_classes, similar to a default weight of 1 for each class
            normalized_weights = self.num_classes * inverse_scores / torch.sum(inverse_scores)
            
            return normalized_weights


class CombinedLoss(nn.Module):
    """
    Combined loss with two levels of dynamic weighting:
    1. Dynamic weights between different loss functions (CE, Dice, Physics, Smoothness).
    2. Dynamic class weights within the CrossEntropyLoss.
    """
    def __init__(self, in_channels_maxwell=1024, num_classes=4, lambda_val=15.0, initial_loss_weights: Optional[List[float]] = None):
        super().__init__()
        
        # --- Initialize Class Weight Updater ---
        self.class_weighter = ClassWeightUpdater(num_classes=num_classes).to(DEVICE)
        
        # --- Initialize loss components ---
        # 1. Cross Entropy Loss (weights will be set dynamically)
        self.ce = nn.CrossEntropyLoss()
        
        # 2. Adaptive t-vMF Dice Loss
        self.dl = Adaptive_tvmf_dice_loss(num_classes=num_classes, lambda_val=lambda_val)
        
        # 3. Physics Loss
        self.pl = PhysicsLoss(in_channels_maxwell)
        
        # 4. Smoothness Loss
        self.sl = SmoothnessLoss()
        
        # --- Initialize Loss Function Weighter ---
        self.loss_weighter = DynamicLossWeighter(num_losses=4, initial_weights=initial_loss_weights).to(DEVICE)
        
        print("Initialized CombinedLoss with dynamic loss-function weights and dynamic class weights.")

    def forward(self, logits, targets, b1=None, all_es=None, feat_sm=None):
        """Forward pass with two-level dynamic weighting."""
        
        # --- Step 1: Update and set dynamic class weights for CE ---
        # Use logits.detach() so that weight calculation is not part of the main computation graph
        new_class_weights = self.class_weighter.update_and_get_weights(logits.detach(), targets)
        self.ce.weight = new_class_weights
        
        # --- Step 2: Calculate individual loss components ---
        # 1. CrossEntropy Loss (now uses dynamic class weights)
        lce = self.ce(logits, targets.long())
        
        # 2. Adaptive t-vMF Dice Loss
        ldc = self.dl(logits, targets)

        # 3. Physics Loss
        lphy = torch.tensor(0.0, device=logits.device)
        if self.pl is not None and b1 is not None and all_es:
            try:
                e1, s1 = all_es[0]    #, all_es[1] -> all_es[0] is tuple (eps, sig)
                lphy = self.pl(b1, e1, s1)
            except (IndexError, TypeError):
                 print("Warning: Physics loss skipped due to unexpected `all_es` format.")
        
        # 4. Smoothness Loss
        lsm = torch.tensor(0.0, device=logits.device)
        if feat_sm is not None:
            lsm = self.sl(feat_sm)

        # --- Step 3: Combine losses using the dynamic loss weighter ---
        individual_losses = torch.stack([lce, ldc, lphy, lsm])
        total_loss = self.loss_weighter(individual_losses)

        return total_loss

    def get_current_loss_weights(self) -> Dict[str, float]:
        """Helper to monitor the weights between different loss functions."""
        weights = self.loss_weighter.get_current_weights()
        return {
            "weight_CE": weights["weight_0"],
            "weight_Dice": weights["weight_1"],
            "weight_Physics": weights["weight_2"],
            "weight_Smoothness": weights["weight_3"],
        }

    def get_current_class_weights(self) -> Dict[str, float]:
        """Helper to monitor the dynamic class weights for CrossEntropyLoss."""
        with torch.no_grad():
            current_weights = self.ce.weight
            return {f"class_{i}_weight": w.item() for i, w in enumerate(current_weights)}
        
    def get_kappa_values(self):
        """Get current κ values for monitoring"""
        if isinstance(self.dl, Adaptive_tvmf_dice_loss):
            return self.dl.get_adaptive_info()
        return {}