

from omegaconf import DictConfig
from collections import OrderedDict
from flwr.common import Scalar, NDArrays
from models.factory import test_server, test
from models.factory import get_model
import numpy as np
import torch
from utils import plot_confusion_matrix, set_seed

def get_on_fit_config(config: DictConfig):
    """Return a function which returns training configurations."""

    def fit_conf_fn(server_round: int):
        """Return a configuration with static lr, momentum and (local) epochs and round."""

        return {'lr': config.lr,
                'momentum': config.momentum,
                'local_epochs': config.local_epochs,
                'round': server_round}
    
    
    return fit_conf_fn

def get_evaluate_fn(model_name:str,dataset_name:str, num_classes: int, testloader, trigger_testloader=None, max_rounds=None, save_path=None, seed=42):
    """Return an evaluation function for server-side evaluation."""

    def evaluate_fn( server_round: int, parameters, config):
        # The `evaluate_fn` function will be called after every round

        set_seed(seed)

        model = get_model(model_name, dataset_name, seed)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy ,class_report, all_labels, all_preds  = test_server(model, testloader, num_classes,  device)
        
        metrics = {
            "accuracy": accuracy,
        }
        
        for cls_idx in range(num_classes):
            cls_str = str(cls_idx)
            if cls_str in class_report:
                metrics[f"recall_class_{cls_idx}"] = class_report[cls_str]["recall"]
                metrics[f"precision_class_{cls_idx}"] = class_report[cls_str]["precision"]
                metrics[f"f1_class_{cls_idx}"] = class_report[cls_str]["f1-score"]

        if trigger_testloader is not None:
            _, asr, _, _, _ = test(model, trigger_testloader, num_classes, device)
            metrics["asr"] = asr

        if max_rounds is not None and server_round == max_rounds:
            plot_confusion_matrix(all_labels, all_preds, num_classes, save_path=save_path)
        return loss, metrics
        
    return evaluate_fn



