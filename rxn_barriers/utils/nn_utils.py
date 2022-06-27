from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mae = mean_absolute_error(labels, predictions)
    rmse = mean_squared_error(labels, predictions, squared=False)
    R2 = r2_score(labels, predictions)

    return {"MAE": mae, "RMSE": rmse, 'R2': R2}


class CustomTrainer(Trainer):
    """Create custom loss function to scale or weight the targets"""
    def __init__(self, scaler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaler = scaler

    def compute_loss(self, model, inputs, return_outputs=False):
        # feed inputs to model and extract logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # extract target Labels
        labels = inputs.get("labels")

        # define custom loss function with class weights
        if self.scaler:
            logits = self.scaler.inverse_transform(logits)

        loss_func = MSELoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss
