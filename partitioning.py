import numpy as np
import inspect
from torch.utils.data import Subset
import torch
from collections import defaultdict
from torchvision import transforms
from collections import defaultdict
import random
import numpy as np
import torch

"""Pathological Non-IID Partitioning Strategy"""
def pathological_partition(dataset, num_clients, num_classes, classes_per_client, seed=42):
    """
    Assignment of `classes_per_client` classes per client (non-exclusive), with random distribution.
    """
    #random.seed(seed)
    #np.random.seed(seed)

    # Map: label -> list of indices
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        label_to_indices[int(label)].append(idx)

    # Shuffle the indices per class
    for indices in label_to_indices.values():
        random.shuffle(indices)

    # Initialize clients
    client_indices = [[] for _ in range(num_clients)]


    # Assign `classes_per_client` classes to each client (not exclusive)
    client_classes = [random.sample(range(num_classes), classes_per_client) for _ in range(num_clients)]

    class_ptr = {cls: 0 for cls in range(num_classes)}  # Pointer for index removal

    for client_id, assigned_classes in enumerate(client_classes):
        for cls in assigned_classes:
            indices = label_to_indices[cls]
            ptr = class_ptr[cls]

            # Distribute data evenly among clients
            samples_per_client = len(indices) // num_clients
            if samples_per_client == 0:
                samples_per_client = 1  # Fallback

            selected = indices[ptr:ptr + samples_per_client]
            client_indices[client_id].extend(selected)
            class_ptr[cls] += samples_per_client

    # Create subsets per client
    from torch.utils.data import Subset
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]

    return client_datasets



"""NonIID Dirichlet Partitioning Strategy"""
def noniid_dirichlet_partition(dataset, num_clients,num_classes,  alpha, seed):

    labels = labels = np.asarray(dataset.targets)  
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Distribution based on Dirichlet
    client_indices = [[] for _ in range(num_clients)]
    for cls_idx in class_indices:
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        split = np.split(cls_idx, proportions)
        for client_id, idx in enumerate(split):
            client_indices[client_id].extend(idx.tolist())

    return [Subset(dataset, idxs) for idxs in client_indices]


"""Feature Skew Partitioning Strategy"""
def feature_skew_partition(dataset, num_clients, variations):
    subsets = []

    # Check which parameters the constructor of the dataset accepts
    constructor_params = inspect.signature(dataset.__class__).parameters

    for cid in range(num_clients):
        brightness = variations.get("brightness", [1.0])[cid % len(variations.get("brightness", [1.0]))]
        rotation = variations.get("rotation", [-5, 5])

        transform = transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.ColorJitter(brightness=brightness),
            transforms.ToTensor()
        ])

        # Prepare the arguments dynamically
        init_args = {}
        if "root" in constructor_params:
            init_args["root"] = getattr(dataset, "root", "./data")
        if "train" in constructor_params:
            init_args["train"] = True
        if "split" in constructor_params:
            init_args["split"] = "train"
        if "download" in constructor_params:
            init_args["download"] = True
        if "transform" in constructor_params:
            init_args["transform"] = transform

        # Instantiate dataset with the appropriate parameters
        skewed_dataset = dataset.__class__(**init_args)

        # Partitioning for this client
        indices = list(range(cid, len(dataset), num_clients))
        subsets.append(Subset(skewed_dataset, indices))

    return subsets
