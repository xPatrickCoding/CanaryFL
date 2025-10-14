# attacks/sign_flipping_attack.py

import numpy as np
from attacks.attack_registry import register_attack

@register_attack("sign_flipping")
class SignFlippingAttack:
    """
    Class for the implementation of the Sign Flipping Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters
    ------
    param_diff : 
        returns difference between model parameters to see the effect of an attack
    -------
    manipulate_parameters
        gets old and new parameters as input, calcultates the update and changes the sign of the update
        returns manipulated parameters
    ------
    apply
        return dataset without any manipulation 
    """
    
    affected_stages = {"fit"}  # Tells the AttackManager when the attack is active

    def __init__(self, config, malicious_ids, seed=42, save_path=None):
        self.malicious_ids = malicious_ids
        self.scale = config.get("scale", -1.0)  # -1 = Flip, < -1 = Flip & Amplify
        self.save_path = save_path
        self.seed = seed
        np.random.seed(seed)
        
    def param_diff(self,params1, params2):
        return sum(np.max(np.abs(p1 - p2)) for p1, p2 in zip(params1, params2))

    def manipulate_parameters(self, old_parameters, new_parameters, client_id):
        if client_id not in self.malicious_ids:
            return new_parameters

        print(f"[!] Client {client_id} performs Sign-Flipping Attack")
        print(self.scale)
        flipped_parameters = [self.scale * (new - old) + old for old, new in zip(old_parameters, new_parameters)]
        #diff = self.param_diff(new_parameters, flipped_parameters)
        return flipped_parameters

    def apply(self, dataset, client_id):
        # No data manipulation with sign flipping
        return dataset
