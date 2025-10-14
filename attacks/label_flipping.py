from torch.utils.data import Dataset
from .attack_registry import register_attack
import random

# Wrapper Dataset mit manipulierten Labels
class LabelFlippedDataset(Dataset):
    """
    Class for the construction of the label flipped dataset

    Methods
    ----------
    __init__
        input: original dataset, mapping of label flips
        output: initialized LabelFlippedDataset
    -------
    __getitem__ : 
        input: index of item
        output: (data, manipulated label)
    -------
    __len__
        returns length of dataset
    ------

    """
    def __init__(self, original_dataset, mapping):
        self.dataset = original_dataset
        self.mapping = mapping

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y_flipped = self.mapping.get(int(y), y)
        return x, y_flipped

    def __len__(self):
        return len(self.dataset)
            
            
@register_attack("label_flipping")
class LabelFlippingAttack:
    """
    Class for the implementation of the Label Flipping Attack

    Methods
    ----------
    __init__
        reads attack mode from config and generates flip mapping, initializes further parameters
    ------
    apply
        return flipped dataset if client is malicious, else original dataset 
    """
    def __init__(self, config, malicious_ids, seed, num_classes):
        self.malicious_ids = malicious_ids
        self.flip_mapping = {}
        self.num_classes = num_classes
        random.seed(seed)

        # Check whether the mode is “random” or “custom”
        mode = config.get("mode", "custom")
        if mode == "random":
            num_flips = config.get("num_flips", 1)  # standard 1 Flip
            flipped_classes = random.sample(range(self.num_classes), num_flips)  # choose random classes to flip
            
            available_targets = list(range(num_classes))
            for cls in flipped_classes:
                available_targets.remove(cls)  # Remove the source class from the possible targets
                target = random.choice(available_targets)
                self.flip_mapping[cls] = target
                available_targets.append(cls)  # Add the source class back to make it available to other classes.
            
            config["flip_mapping"] = self.flip_mapping

        elif mode == "custom":
            self.flip_mapping = config.get("flip_mapping", {})
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def apply(self, dataset: Dataset, client_id: int):
        if client_id not in self.malicious_ids:
            return dataset
        print(f"[!] Client {client_id} performs LabelFlipping attack")

        return LabelFlippedDataset(dataset, self.flip_mapping)
