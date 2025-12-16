# CanaryFL: Benchmark Framework for evaluating poisoning attacks under realistic scenarios
Reproducible benchmarks for poisoning attacks in federated learning — datasets, attack implementations, and evaluation pipelines


![Python Versions](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


This text provides an overview of the features of our framework and all the information you need to set up the framework as easily as possible. 

## Main Features

| Feature                                                   | Our Framework |
|-----------------------------------------------------------|---------------|
| **Security Capabilities**                                 |               |
| High attack diversity (≥ 8, modular)                      | ✅             |
| Defenses integrated                                       | ✅             |
| Realistic special/medical datasets (e.g., PathMNIST)      | ✅             |
| **Flexibility & Models**                                  |               |
| High model diversity                                      | ✅             |
| State-of-the-art IoT models (e.g., EfficientNet)          | ✅             |
| Hydra organization, configs, logging                      | ✅             |
| Reproducibility                                           | ✅             |
| Automatic aggregation of repeated experiments             | ✅             |
| **Usability & Automation**                                |               |
| Fully automated evaluation pipeline                       | ✅             |
| Ntfy integration: automatic alerts upon completion        | ✅             |
| Comprehensive evaluation results                          | ✅             |
| Very easy entry / usability                               | ✅             |
| Active community / maintenance                            | ✅             |
| **Production Readiness**                                  |               |
| Flower-based                                              | ✅             |
| Scalable (cloud, edge, local)                             | ✅             |
| Easy Extendability for Real Device Deployment             | ✅             |

## Configuration Management

   **Category**      |                                                                                      **Details**                                                                                       |
| :-------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   **FL Algorithms**   |                                                                FedAvg,Krum, MultiKrum, FedAdam, FedOpt, FedAvgM, FedTrimmedAvg, NormClipFedAvg, FedMedian, SimpleClustering                                                          |
| **Data Distribution** |  IID, Pathological Label-Skew, Dirichlet Label-Skew, Feature-Skew  |
|     **Datasets**      |                                    MNIST, FashionMNIST, CIFAR10, PathMNIST, DermaMNIST                                    |
|      **Models**       | Lenet5,Resnet18, MobilenetV3-small, MobilenetV3-large, Vgg11, Vgg13, Vgg16, Vgg19, Efficientnet-b0, Efficientnet-b1, Efficientnet-b2 |
|      **Attacks**       | Label Flipping, Sign Flipping Attack, Gaussian Random Attack, Centralized Backdoor Attack, Decentralized Backdoor Attack, Model Replacement Attack, Direct Model Scaling Attack, Model Initialization Attack |



## Getting started


1. Install Miniconda: 
```
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
```
2. After Install refresh terminal
```
source ~/miniconda3/bin/activate
```
3. Create a new conda environment
```
conda create --name <my-env> python=3.10
```
4. Activate new environment
```conda activate <my-env>```

5. Maybe you need to install pip for conda with ```conda install pip```
6. Install pytorch libraries with ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu``` (visit https://pytorch.org/get-started/locally/ to find your´e version)
6. Install requirements.txt with ```pip3 install -r requirements.txt```
7. Clone repository 

Now you are able to specify configs in the configs folder and run ```python3 main.py``` to start the experiment. One configuration must be named ```base.yaml```, the others can be named arbitrarily.

To get an overview of some possible parameters, take a look at the “Example Configurations” folder. Here you will find a sample configuration for each attack. In addition, the file “some_possible_parameters.yaml” shows the possible values for the most important parameters. 

If you choose VGG as the model you need to turn off gpu training, because these models do not support cuda deterministic training. The second option would be to unset "torch.use_deterministic_algorithms(True)', but then you dont get reproducible results anymore. 
## Citation

If you are using FLPoison for your work, please cite our paper with:


## Licenses and Third-Party Resources

### Project License
The code in this repository is released under the Apache 2.0 License (see LICENSE file).

### Datasets
This benchmark uses the following publicly available datasets:

- [**MNIST**](http://yann.lecun.com/exdb/mnist/) (LeCun et al.), licensed under the Creative Commons Attribution-Share Alike 3.0  
- [**Fashion-MNIST**](https://arxiv.org/abs/1708.07747) (Han Xiao, Kashif Rasul, Roland Vollgraf
), licensed under the MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com
- [**CIFAR-10**](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Krizhevsky et al.), available for academic research use  
- [**PathMNIST**](https://medmnist.com/) (Yang et al.), licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0)
- [**DermaMNIST**](https://medmnist.com/) (Yang et al.), licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

The datasets are not redistributed in this repository and are obtained from their official sources.

## Third-Party Frameworks

- [Flower](https://arxiv.org/abs/2007.14390), Apache License 2.0 — used as included, code not modified
- [PyTorch](https://pytorch.org/), BSD-3-Clause
- [Hydra](https://hydra.cc/), MIT License

