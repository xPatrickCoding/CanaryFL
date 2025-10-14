from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA
from utils import DATASET_INFO
from partitioning import pathological_partition, noniid_dirichlet_partition, feature_skew_partition
from utils import save_malicious_clients
from torchvision import transforms
import torch
from attacks.attack_manager import AttackManager
from torchvision.transforms import Lambda
from medmnist import PathMNIST
from medmnist import INFO as MEDMNIST_INFO
from WrappedPathMNIST import WrappedPathMNIST
from WrappedDermaMNIST import WrappedDermaMNIST
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pathlib import Path


def seed_worker(worker_id):
    """sets the seed for data loader workers for reproducibility"""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(dataset_name: str, model_name: str, datapath: str = './data'):
    """gets the dataset name as input, takes related informationen, downloads the dataset and returns a train and test set"""

    print(dataset_name)
    dataset_name = dataset_name.lower()
    info = DATASET_INFO[dataset_name]

    if dataset_name == "mnist":
        transform = get_transform(dataset_name, model_name, info["input_size"], info["input_channels"])
        trainset = MNIST(datapath, train=True, download=True, transform=transform)
        testset = MNIST(datapath, train=False, download=True, transform=transform)

    elif dataset_name in ["fashion-mnist", "fmnist"]:
        transform = get_transform(dataset_name, model_name, info["input_size"], info["input_channels"])
        trainset = FashionMNIST(datapath, train=True, download=True, transform=transform)
        testset = FashionMNIST(datapath, train=False, download=True, transform=transform)

    elif dataset_name == "cifar10":
        train_transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-7, 7)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])
        
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])
        
        trainset = CIFAR10(datapath, train=True, download=True, transform=train_transformations)
        testset = CIFAR10(datapath, train=False, download=True, transform=test_transforms)
    
    elif dataset_name == "pathmnist":
        transform = get_transform(dataset_name, model_name, info["input_size"], info["input_channels"])
        trainset = WrappedPathMNIST(split="train", transform=transform, download=True)
        testset = WrappedPathMNIST(split="test", transform=transform, download=True)
        
    elif dataset_name == "dermamnist":
        transform = get_transform(dataset_name, model_name, info["input_size"], info["input_channels"])
        trainset = WrappedDermaMNIST(split="train", transform=transform, download=True)
        testset = WrappedDermaMNIST(split="test", transform=transform, download=True)


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainset, testset



def prepare_dataset(dataset_name: str,
                    model_name: str,
                    num_partitions: int, 
                    batch_size: int,
                    val_ratio: float = 0.1, 
                    partitioning: dict = None, attacks: dict = None, seed:int = 2023, save_path: str = None):
    
    """Prepares the dataset for federated learning by partitioning it and applying attacks if specified."""

    info = DATASET_INFO[dataset_name.lower()]
    trainset, testset = get_dataset(dataset_name, model_name)
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    # Partition the dataset to the specified partitioning method (like noniid, iid)
    if partitioning and partitioning.get("non_iid", False):
        partition_type = partitioning.get("type", "label-skew")

        if partition_type == "label-skew":
            method = partitioning["label-skew"]["method"]
            if method == "pathological":
                classes_per_client = partitioning["label-skew"].get("classes_per_client", 2)
                datasets = pathological_partition(trainset, num_partitions, info["num_classes"], classes_per_client, seed=seed)
            elif method == "dirichlet":
                alpha = partitioning["label-skew"].get("alpha", 0.5)
                datasets = noniid_dirichlet_partition(trainset, num_partitions,info["num_classes"],  alpha, seed)
            else:
                raise ValueError(f"Unknown label-skew method: {method}")

        elif partition_type == "feature-skew":
            variations = partitioning["feature-skew"].get("variations", {})
            datasets = feature_skew_partition(trainset, num_partitions, variations)
        
        else:
            raise ValueError(f"Unknown non-IID type: {partition_type}")
    
    else:
    
        num_total = len(trainset)
        base_size = num_total // num_partitions
        remainder = num_total % num_partitions

        partition_len = [base_size + 1 if i < remainder else base_size for i in range(num_partitions)]
        datasets = random_split(trainset, partition_len, torch.Generator().manual_seed(seed))

    
    # if attacks are active apply them to the datasets of the malicious clients
    malicious_clients = set()
    print(attacks)
    bd_testloader = None
    if attacks:
        print("Num partitions:", num_partitions)
        attack_manager = AttackManager(attacks, total_clients=num_partitions, seed=seed, save_path=save_path, testset=testset, num_classes = info["num_classes"])
        malicious_clients = attack_manager.malicious_clients
        for i in range(len(datasets)):
            datasets[i] = attack_manager.apply_attacks(datasets[i], i)
        bd_testset = attack_manager.get_triggered_testloader()
        #print("Length of Tesloader:", len(testset))
        
        # Function to save some trigger samples from the backdoor testset
        def save_trigger_samples(dataset, save_path, client_id, num_samples=5):
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            saved = 0
            for idx in range(len(dataset)):
                if hasattr(dataset, "indices_to_poison") and idx in dataset.indices_to_poison:
                    x, y = dataset[idx]
                    filename = save_path / f"trigger_client_{client_id}_{saved}.png"
                    save_image(x, filename)
                    print(f"[DBAttack] Trigger-Bild gespeichert: {filename}")
                    saved += 1
                    if saved >= num_samples:
                        break

        if bd_testset:
            save_trigger_samples(bd_testset, num_samples=5,client_id=0,  save_path=save_path)
            bd_testloader = DataLoader(
                bd_testset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2, 
                worker_init_fn=seed_worker, 
                generator=g
            )
            
            
    # Create train, validation and test dataloaders for each client with two workers
    trainloaders = []
    valloaders = []
    for trainset_ in datasets: 
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(seed))
        trainloaders.append(DataLoader(
            for_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            worker_init_fn=seed_worker,
            generator=g
        ))

        valloaders.append(DataLoader(
            for_val, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g
        ))
        
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Save the IDs of the malicious clients as csv
    save_malicious_clients(malicious_clients, total_clients=num_partitions, results_path=save_path)
    return trainloaders, valloaders, testloader, bd_testloader, info["num_classes"], malicious_clients

def repeat_channels(x):
    return x.repeat(3, 1, 1)

def get_transform(dataset_name: str, model_name: str, target_shape: tuple, target_channels: int):
    """returns the transformations needed for the specific dataset and model"""

    base_transform = [ToTensor()]

    if model_name.lower() in ["efficientnet", "vgg11", "vgg16","mobilenetv3-small","mobilenetv3-large", "mobilenetv2", "inception", "vgg", "alexnet"]:
        if dataset_name in ["mnist", "fmnist"]:
            base_transform.append(Resize((224, 224)))
        else:
            base_transform.append(Resize((224, 224)))
    elif model_name.lower() == "lenet":
        base_transform.append(Resize((32, 32)))  

    # Normalize depending on the dataset
    mean_std_map = {
        "mnist": ((0.1307,), (0.3081,)),
        "fmnist": ((0.2860,), (0.3530,)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "pathmnist": ((0.7405, 0.5330, 0.7058), (0.0723, 0.1038, 0.0731)), 
        "dermamnist": ((0.7082, 0.5021, 0.5668), (0.0988, 0.1421, 0.1523)), 

    }
    mean, std = mean_std_map[dataset_name]
    base_transform.append(Normalize(mean, std))

    return Compose(base_transform)






