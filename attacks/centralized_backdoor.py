from attacks.attack_registry import register_attack
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.utils import save_image
from pathlib import Path
from utils import set_seed


class CTriggeredDataset(Dataset):
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



@register_attack("cba")  
class CBAttack:
    """
    Class for the implementation of the Centralized Backdoor Attack

    Methods
    ----------
    __init__
        initializes the attack with config parameters and generates backdoor testset
        
    ------
    generate_backdoor_testset
        return testset that only contains triggered samples from source_label to target_label
    ------
    ------
    apply
        return triggered trainset for specific client if it is malicious, else original dataset
    ------
    insert_trigger
        inserts the trigger into the image at the specified position
    ------
    visualize_trigger
        saves an example image with the trigger to the specified path
    """
    def __init__(self, config: dict, malicious_ids: set, seed: int, save_path: str = None, testset=Dataset):
        self.config = config
        self.malicious_ids = malicious_ids
        self.seed = seed
        self.testset=testset
        
        set_seed(seed)
        self.source_label = config.get("source_label", 2)
        self.target_label = config.get("target_label", 7)
        self.poisoning_ratio = config.get("poisoning_ratio", 0.3)
        self.trigger_value = config.get("trigger_value", 255)  # color of trigger
        self.trigger_position = config.get("trigger_position", [(26, 26)])  # coordinates of trigger position
        
        self.backdoor_testset= self.generate_backdoor_testset()

        if save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None
                
                
    def generate_backdoor_testset(self):
        if self.testset is None:
            return None
        print("erzeuge backdor set")
        indices_to_poison = [i for i, (_, label) in enumerate(self.testset) if int(label) == self.source_label]
        filtered_data = [self.testset[i] for i in indices_to_poison]
        indices_to_poison = list(range(len(filtered_data)))
        print(len(filtered_data), "Backdoor Testset Länge")
        bd_testset = CTriggeredDataset(filtered_data, indices_to_poison, self.insert_trigger, self.target_label)
        
        return bd_testset
    

    def apply(self, dataset: Dataset, client_id: int) -> Dataset:
        if client_id not in self.malicious_ids:
            return dataset  # Do not manipulate if client is not malicious

        poisoned_indices = []
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            if y == self.source_label:
                poisoned_indices.append(idx)

        num_poison = int(self.poisoning_ratio * len(poisoned_indices))
        poisoned_indices = np.random.choice(poisoned_indices, num_poison, replace=False).tolist()

        bd_dataset =  CTriggeredDataset(dataset, poisoned_indices, self.insert_trigger, self.target_label)
    
        if self.save_path:
            self.visualize_trigger(bd_dataset, client_id)

        return bd_dataset
    
    def insert_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """Places trigger pixels at specific locations."""
        for x, y in self.trigger_position:
            if img.shape[0] == 1:  # 1 Kanal
                img[0, x, y] = self.trigger_value
            else:
                img[:, x, y] = self.trigger_value  # Set all channels to the same level
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