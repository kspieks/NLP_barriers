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
    def __init__(self, scaler=None, targets=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._scaler = scaler
        self._targets = targets
        self.compute_metrics = self._compute_metrics

    def _compute_metrics(self, eval_pred):
        """
        Must take an [`EvalPrediction`] and return
        a dictionary string to metric values.
        """
        # convert numpy arrays to torch tensors
        predictions, labels = eval_pred
        predictions, labels = torch.tensor(predictions), torch.tensor(labels)
        loss_func = MSELoss()
        scaled_loss = loss_func(labels, predictions)

        scaled_mae = mean_absolute_error(labels, predictions)
        scaled_rmse = mean_squared_error(labels, predictions, squared=False)
        scaled_R2 = r2_score(labels, predictions)

        if self._scaler is None:
            output_dict = {
                'MSE_loss': scaled_loss,
                'MAE': scaled_mae,
                'RMSE': scaled_rmse,
                'R2': scaled_R2,
            }
            if len(self._targets) > 1:
                for i, target in enumerate(self._targets):
                    output_dict[f'MAE_{target}'] = mean_absolute_error(labels[:, i], predictions[:, i])
                    output_dict[f'RMSE_{target}'] = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
                    output_dict[f'R2_{target}'] = r2_score(labels[:, i], predictions[:, i])
                    scaled_R2 = r2_score(labels[:, i], predictions[:, i])

            return output_dict
        else:
            labels_unscaled = self._scaler.inverse_transform(labels)
            preds_unscaled = self._scaler.inverse_transform(predictions)
            
            mae = mean_absolute_error(labels_unscaled, preds_unscaled)
            rmse = mean_squared_error(labels_unscaled, preds_unscaled, squared=False)
            R2 = r2_score(labels_unscaled, preds_unscaled)

            output_dict = {
                'scaled_MSE_loss': scaled_loss,
                'scaled_MAE': scaled_mae,
                'scaled_RMSE': scaled_rmse,
                'scaled_R2': scaled_R2,
                'MAE': mae,
                'RMSE': rmse,
                'R2': R2,
            }

            if len(self._targets) > 1:
                for i, target in enumerate(self._targets):
                    output_dict[f'MAE_{target}'] = mean_absolute_error(labels_unscaled[:, i], preds_unscaled[:, i])
                    output_dict[f'RMSE_{target}'] = mean_squared_error(labels_unscaled[:, i], preds_unscaled[:, i], squared=False)
                    output_dict[f'R2_{target}'] = r2_score(labels_unscaled[:, i], preds_unscaled[:, i])
                    scaled_R2 = r2_score(labels_unscaled[:, i], preds_unscaled[:, i])

            return output_dict

