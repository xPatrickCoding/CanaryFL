# attacks/gaussian_random.py
import numpy as np
from attacks.attack_registry import register_attack
import matplotlib.pyplot as plt
import os


@register_attack("gaussian_attack")  
class GaussianRandomAttack:
    """
    Class for the implementation of the Gaussian Random Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters
    ------
    manipulate_parameters
        gets old and new parameters as input, calcultates the update and adds noise to the update
        returns manipulated parameters
    ------
    apply
        return dataset without any manipulation 
    ------
    visualize_perturbation
        creates and saves visualizations of the perturbations introduced by the attack
    """
    affected_stages = {"fit"}  # Tells the AttackManager when the attack is active

    def __init__(self, config, malicious_ids, seed=42, save_path=None):
        self.malicious_ids = malicious_ids
        self.mean = config.get("mean", 0.0)
        self.std = config.get("std", 0.05)
        self.seed = seed
        np.random.seed(seed)
        self.save_path = save_path
        self.has_plotted = False

    def manipulate_parameters(self, old_parameters, new_parameters, client_id):
        np.random.seed(self.seed+client_id)

        if client_id not in self.malicious_ids:
            return new_parameters  # normal

        # Ersetze die Änderung (Delta) durch Gauß-Rauschen
        print(f"[!] Client {client_id} führt GRA aus")
        noisy_parameters = []
        for old, new in zip(old_parameters, new_parameters):
            shape = old.shape
            dtype = old.dtype
            delta = new - old
            # Gauß-Rauschen im selben Format
            noise = np.random.normal(loc=self.mean, scale=self.std, size=shape).astype(dtype)

            attacked_delta = delta + noise
            noisy_parameters.append(old + attacked_delta)
        


        visualize_perturbation(old_parameters, noisy_parameters, client_id, save_path=self.save_path)
            
        return noisy_parameters
    
    def apply(self, dataset, client_id):
        # No data manipulation with this attack
        return dataset
    


def visualize_perturbation(old_parameters, new_parameters, client_id, save_path):
    if not save_path:
        print(f"[!] save_path is None – Visualisierung skipped")
        return

    os.makedirs(save_path, exist_ok=True)

    # (1) Histogram of absolute changes
    diffs = [np.abs(new - old).flatten() for old, new in zip(old_parameters, new_parameters)]
    diffs = np.concatenate(diffs)

    plt.figure(figsize=(8, 4))
    plt.hist(diffs, bins=100, color="skyblue", edgecolor="black")
    plt.title(f"[Client {client_id}] Absolute parameter change through Gaussian Attack")
    plt.xlabel("Δ Parameter")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"gaussian_hist_client_{client_id}.png"))
    plt.close()

    # (2) Line plot for an example parameter array (e.g., weight matrix of a layer)
    example_idx = 0  # Erstes Layer
    old_flat = old_parameters[example_idx].flatten()
    new_flat = new_parameters[example_idx].flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(old_flat, label="Before", alpha=0.7)
    plt.plot(new_flat, label="After", alpha=0.7)
    plt.title(f"[Client {client_id}] Example parameter comparison (Layer {example_idx})")
    plt.xlabel("Index")
    plt.ylabel("Wert")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"gaussian_lineplot_client_{client_id}.png"))
    plt.close()








