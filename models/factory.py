# models/factory.py

from models.LeNet5 import LeNet5
from models.ResNet18 import ResNet18
from models.MobileNetV3 import MobileNetV3
from models.VGG import VGG
import torch
from sklearn.metrics import classification_report, confusion_matrix
from models.EfficientNet import EfficientNetWrapper
from utils import set_seed

def get_model(model_name: str, dataset_name: str, seed: int = 42):
    """"Generates the model specific to the specified dataset by name and returns it."""
    set_seed(seed)
    from utils import DATASET_INFO
    info = DATASET_INFO[dataset_name.lower()]
    input_channels = info["input_channels"]
    num_classes = info["num_classes"]
    model_name = model_name.lower()
    if model_name == "lenet5":
        return LeNet5(num_classes=num_classes, in_channels=input_channels)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif model_name == "mobilenetv3-small":
        return MobileNetV3(num_classes=num_classes, in_channels=input_channels, version='small')
    elif model_name == "mobilenetv3-large":
        return MobileNetV3(num_classes=num_classes, in_channels=input_channels, version='large')
    elif model_name == "vgg11":
        return VGG(num_classes=num_classes, in_channels=input_channels, version="vgg11")
    elif model_name == "vgg13":
        return VGG(num_classes=num_classes, in_channels=input_channels, version="vgg13")
    elif model_name == "vgg16":
        return VGG(num_classes=num_classes, in_channels=input_channels, version="vgg16")
    elif model_name == "vgg19":
        return VGG(num_classes=num_classes, in_channels=input_channels, version="vgg19")
    elif model_name == "efficientnet":
        return EfficientNetWrapper(num_classes=num_classes, in_channels=input_channels, version="efficientnet_b0")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(net, trainloader, optimizer,epochs, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, num_classes, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds = []
    all_labels = []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / len(testloader.dataset)
    
        # Classification report
    class_report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0, labels=list(range(num_classes))
    )
    return loss, accuracy, class_report,  all_labels, all_preds




def test_client(net, testloader, num_classes, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    
    return loss, accuracy




def test_server(net, testloader, num_classes, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds = []
    all_labels = []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / len(testloader.dataset)
    
        # Classification report
    class_report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0, labels=list(range(num_classes))
    )
    return loss, accuracy, class_report,  all_labels, all_preds