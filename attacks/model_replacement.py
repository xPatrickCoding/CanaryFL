from attacks.attack_registry import register_attack

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

@register_attack("model_replacement")
class ModelReplacementAttack:
    """
    Class for the implementation of the Model Replacement Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters
    ------
    manipulate_parameters
        gets old and new parameters as input, performs attack to replace the global model with attack models
        returns scaled manipulated parameters
    ------
    apply
        return dataset without any manipulation 
    """
    
    affected_stages = {"fit"}  # Tells the AttackManager when the attack is active

    def __init__(self, config, malicious_ids, seed=42, save_path=None):
        self.malicious_ids = malicious_ids
        self.total_clients = config.get("num_clients", 20)
        if len(malicious_ids) > 0:
            default_scaling = self.total_clients / len(malicious_ids)
        else:
            default_scaling = 1.0  # Fallback
            
        self.scaling_factor = config.get("scaling_factor", default_scaling)
        self.save_path = save_path
        self.seed = seed


    def manipulate_parameters(self, global_model, target_model, client_id):
        
        if client_id not in self.malicious_ids:
            return target_model
        
        if global_model is None:
            print("[WARN] Global parameters not set, no attack")
            return target_model

        print(f"[!] Model Replacement Attack on client {client_id}")

        # Elementweise skalierte Updates:
        scaled_params = []
        for glob_p, targ_p in zip(global_model, target_model):
            update = targ_p - glob_p
            scaled_update = self.scaling_factor * update
            scaled_param = glob_p + scaled_update
            scaled_params.append(scaled_param)

        return scaled_params

    def apply(self, dataset, client_id):
        # No data manipulation with this attack

        return dataset
