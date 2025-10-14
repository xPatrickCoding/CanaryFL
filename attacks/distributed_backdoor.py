from attacks.attack_registry import register_attack
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision.utils import save_image
from pathlib import Path
from functools import partial
from utils import set_seed

class DTriggeredDataset(Dataset):
    """
    Class for the generating a triggered dataset

    Methods
    ----------
    __init__
        initializes the attack with config parameters and generates backdoor testset
        
    ------
    get_item
        return (possibly triggered) sample and label at given index
    ------
    __len__
        returns length of dataset
    """
    def __init__(self, dataset, indices_to_poison, insert_trigger_fn, target_label):
        self.dataset = dataset
        self.indices_to_poison = set(indices_to_poison)
        self.insert_trigger_fn = insert_trigger_fn
        self.target_label = target_label
        self.testloader = None
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.indices_to_poison:
            x = self.insert_trigger_fn(x.clone())
            y = self.target_label
        return x, y
    
    def __len__(self):
        return len(self.dataset)


def get_backdoor_test_set(original_testset, source_label, insert_trigger_fn, target_label):
    indices_to_poison = [i for i, (_, label) in enumerate(original_testset) if label == source_label]
    filtered_data = [original_testset[i] for i in indices_to_poison]
    indices_to_poison = list(range(len(filtered_data)))
    print(len(filtered_data), "Backdoor Testset Länge")
    return DTriggeredDataset(filtered_data, indices_to_poison, insert_trigger_fn, target_label)

@register_attack("dba")  # Registration for AttackManager 
class DBAttack:
    """
    Class for the implementation of the Distributed Backdoor Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters and generates backdoor testset
    -------
    
    setup_trigger_position
        calculates the position of the trigger based on the trigger index
    ------
    insert_combined_trigger
        inserts multiple triggers into the image at predefined positions
        
    ------
    generate_backdoor_testset
        return testset that only contains triggered samples from source_label to target_label
    ------
    ------
    apply
        return triggered trainset for specific client if it is malicious, else original dataset
    ------
    insert_trigger
        inserts a part dependent on the client_id of the complete trigger into the image at the specified position
    ------
    visualize_trigger
        saves an example image with the trigger to the specified path
    """
    def __init__(self, config: dict, malicious_ids: set, seed: int, testset:Dataset, save_path: str = None):
        self.config = config
        self.malicious_ids = malicious_ids
        self.seed = seed
        self.testset= testset
        set_seed(seed)

        self.source_label = config.get("source_label", 2)
        self.target_label = config.get("target_label", 7)
        self.poisoning_ratio = config.get("poisoning_ratio", 0.3)
        self.trigger_value = config.get("trigger_value", 255)

        # Trigger distribution
        self.num_triggers = config.get("num_triggers", 2)
        self.trigger_size = config.get("trigger_size", 2)
        self.gap = config.get("gap", 2)
        self.shift = config.get("shift", 0)
        
        self.trigger = torch.ones((self.num_triggers, 1, self.trigger_size, self.trigger_size)) * self.trigger_value
        self.backdoor_testset= self.generate_backdoor_testset()

        if save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None
         
    def setup_trigger_position(self, trigger_idx: int):
        row = (trigger_idx // 2) * (self.trigger_size + self.gap) + self.shift
        col = (trigger_idx % 2) * (self.trigger_size + self.gap) + self.shift
        return row, col
    
    def insert_combined_trigger(self, x: torch.Tensor) -> torch.Tensor:
        for trigger_idx in range(self.num_triggers):
            position = self.setup_trigger_position(trigger_idx)
            segment = self.trigger[trigger_idx].unsqueeze(0)
            x = self.insert_trigger(x, position, segment)
        return x
    
    def generate_backdoor_testset(self):
        if self.testset is None:
            return None

        indices_to_poison = [i for i, (_, label) in enumerate(self.testset) if label == self.source_label]
        filtered_data = [self.testset[i] for i in indices_to_poison]

        indices_to_poison = list(range(len(filtered_data)))
        print(len(filtered_data), "Backdoor Testset Länge")
        return DTriggeredDataset(filtered_data, indices_to_poison, self.insert_combined_trigger, self.target_label)


    def apply(self, dataset: Dataset, client_id: int) -> Dataset:
        if client_id not in self.malicious_ids:
            return dataset

        poisoned_indices = [idx for idx in range(len(dataset)) if dataset[idx][1] == self.source_label]
        num_poison = int(self.poisoning_ratio * len(poisoned_indices))
        poisoned_indices = np.random.choice(poisoned_indices, num_poison, replace=False).tolist()

        trigger_idx = client_id % self.num_triggers
        trigger_position = self.setup_trigger_position(trigger_idx)
        trigger_segment = self.trigger[trigger_idx].unsqueeze(0)  # 1x1xSxS Tensor

        insert_fn = partial(self.insert_trigger, position=trigger_position, trigger_segment=trigger_segment)

        bd_dataset = DTriggeredDataset(
            dataset, poisoned_indices,
            insert_fn,
            self.target_label
        )

        if self.save_path:
                self.visualize_trigger(bd_dataset, client_id)

        return bd_dataset

    
    def insert_trigger(self, img: torch.Tensor, position: tuple, trigger_segment: torch.Tensor) -> torch.Tensor:
        row, col = position
        c, h, w = img.shape
        _, _, th, tw = trigger_segment.shape  # trigger size
        
        for i in range(th):
            for j in range(tw):
                if c == 1:
                    img[0, row + i, col + j] = trigger_segment[0, 0, i, j]
                else:
                    img[:, row + i, col + j] = trigger_segment[0, 0, i, j]
        return img






    def visualize_trigger(self, dataset: Dataset, client_id: int):
        if not self.save_path:
            return  # Do nothing if no storage path is set
        for idx in range(len(dataset)):
            if idx in dataset.indices_to_poison:
                x, _ = dataset[idx]  
                filename = self.save_path / f"trigger_client_{client_id}.png"
                save_image(x, filename)
                print(f"[DBAttack] Trigger-Bild gespeichert: {filename}")
                break
