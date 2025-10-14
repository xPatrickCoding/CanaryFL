import torch
from .attack_registry import register_attack

@register_attack("model_scaling")
class ModelScalingAttack:
    """
    Class for the implementation of the Model Scaling Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters
    ------
    manipulate_parameters
        gets old and new parameters as input, multiplies the new parameters with a scaling factor
        returns scaled manipulated parameters
    ------
    manipulate_gradients
        gets gradient as input and multiplies the gradient with a scaling factor
        returns scaled gradient
    ------
    apply
        return dataset without any manipulation 
    """
    
    affected_stages = {"fit"} # Tells the AttackManager when the attack is active
    
    def __init__(self, config: dict, malicious_ids: set, seed: int):
        self.malicious_ids = malicious_ids
        self.factor = config.get("explosion_factor", 2)  # Faktor für die Manipulation

    def apply(self, dataset, client_id):
        # No data manipulation with this attack
        return dataset

    def manipulate_gradients(self, gradients, client_id):
        print("Manipulating gradients for client:", client_id)
        if client_id in self.malicious_ids:
            # Gradient Scaling
            return [g * self.factor for g in gradients]
        return gradients

    def manipulate_parameters(self, old_parameters, new_parameters, client_id):
        print("Manipulating gradients for client:", client_id)

        if client_id in self.malicious_ids:
            # Scaling of Modellparameters 
            return [p * self.factor for p in new_parameters]
        return new_parameters
