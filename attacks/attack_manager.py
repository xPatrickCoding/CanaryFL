from .attack_registry import get_attack
from .model_scaling import ModelScalingAttack
from .model_replacement import ModelReplacementAttack
from .centralized_backdoor import CBAttack
from .distributed_backdoor import DBAttack
from .sign_flipping import SignFlippingAttack
from omegaconf import ListConfig, DictConfig

class AttackManager:
    """
    Class for the generating the attacks based on the config file and managing them

    Methods
    ----------
    __init__
        generates the attac instances based on the config file
        
    ------
    _is_attack_active
        returns True if the attack should be active in the current round, else False
    ------
    apply_attacks
        applies all active attacks on the dataset of a client, calls the apply method of each attack
        relevant for data poisoning attacks like backdoor attacks
    ------
    manipulate_parameters
        applies all active attacks on the model parameters of a client, calls the manipulate_parameters method of each attack
        relevant for model poisoning attacks like sign flipping, model scaling, model replacement
    ------
    get_triggered_testloader
        returns triggered testset for backdoor attacks if available
    """
    def __init__(self, config: dict, total_clients: int, seed: int, save_path: str = None, testset=None, num_classes:int = 10):
        self.active_attacks = []
        self.save_path = save_path
        self.malicious_clients = set()
        
        if not config:
            print("[INFO] No attacks configured.")
            return
        for attack_name, attack_conf in config.items():
            if not attack_conf.get("enabled", False):
                continue
            print(attack_name)
            attack_cls = get_attack(attack_name)
            if not attack_cls:
                raise ValueError(f"Unknown attack: {attack_name}")
            
            fraction = attack_conf.get("malicious_client_fraction", 0.0)
            num_malicious = int(total_clients * fraction)
            malicious_ids = set(range(num_malicious))  # z.B. Client 0, 1, ...
            if attack_name == "cba" or attack_name=="dba":
                attack_instance = attack_cls(config=attack_conf, malicious_ids=malicious_ids, seed=seed, save_path=self.save_path, testset=testset)
            elif attack_name == "gaussian_attack" or attack_name == "model_replacement": 
                attack_instance = attack_cls(config=attack_conf, malicious_ids=malicious_ids, seed=seed, save_path=self.save_path)
            elif attack_name == "label_flipping":
                attack_instance = attack_cls(config=attack_conf, malicious_ids=malicious_ids, seed=seed, num_classes=num_classes)
            else:
                attack_instance = attack_cls(config=attack_conf, malicious_ids=malicious_ids, seed=seed)
            
            attack_instance.active_rounds = attack_conf.get("active_rounds", None)
            
            self.active_attacks.append(attack_instance)
            self.malicious_clients.update(malicious_ids)



    def _is_attack_active(self, attack, current_round: int) -> bool:
        """Check if attack is scheduled for the current round"""
        
        rounds = getattr(attack, "active_rounds", None)
        if rounds is None:  # default: always active
            return True
        if isinstance(rounds, (list, ListConfig)):
            return current_round in list(rounds)

        if isinstance(rounds, (dict, DictConfig)):
            start = rounds.get("start", 0)
            end = rounds.get("end", float("inf"))
            step = rounds.get("every", 1)
            return start <= current_round <= end and (current_round - start) % step == 0

        return False
    
    
    def apply_attacks(self, dataset, client_id, current_round: int = 0):
        for attack in self.active_attacks:
            dataset = attack.apply(dataset, client_id)
        return dataset
    
    def manipulate_parameters(self, old_parameters, new_parameters, client_id, stage, current_round: int = 0):
        for attack in self.active_attacks:
            if not self._is_attack_active(attack, current_round):
                continue
            if hasattr(attack, "manipulate_parameters") and stage in getattr(attack, "affected_stages", set()):
                print(f"Manipulating parameters for attack: {attack.__class__.__name__} at stage: {stage}")
                new_parameters = attack.manipulate_parameters(old_parameters, new_parameters, client_id)
        return new_parameters


    
    def get_triggered_testloader(self):
        for attack in self.active_attacks:
            if hasattr(attack, "generate_backdoor_testset") and attack.testset is not None:
                print("Attribut gefunden")
                if attack.backdoor_testset is not None:
                    return attack.backdoor_testset
        return None

