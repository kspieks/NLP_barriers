from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.nn import MSELoss
from transformers import Trainer


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    
    Args:
        model: An nn.Module.
    
    Returns:
        The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class CustomTrainer(Trainer):
    """
    Create custom Trainer class
    """
    def __init__(self, scaler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaler = scaler

    def compute_metrics(self, eval_pred):
        """
        Must take an [`EvalPrediction`] and return
        a dictionary string to metric values.
        """
        predictions, labels = eval_pred
        loss_func = MSELoss()
        scaled_loss = loss_func(logits, labels)

        scaled_mae = mean_absolute_error(labels, predictions)
        scaled_rmse = mean_squared_error(labels, predictions, squared=False)
        scaled_R2 = r2_score(labels, predictions)

        labels_unscaled = self.scaler.inverse_transform(labels)
        preds_unscaled = self.scaler.inverse_transform(predictions)
        
        mae = mean_absolute_error(labels_unscaled, preds_unscaled)
        rmse = mean_squared_error(labels_unscaled, preds_unscaled, squared=False)
        R2 = r2_score(labels_unscaled, preds_unscaled)

        output_dict = {
            'scaled_MSE_loss': scaled_loss,
            'scaled_mae': scaled_mae,
            'scaled_rmse': scaled_rmse,
            'scaled_R2': scaled_R2,
            'MAE': mae,
            'RMSE': rmse,
            'R2': R2,
        }
        return output_dict

