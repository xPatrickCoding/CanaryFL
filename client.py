
import flwr as fl
import torch
from collections import OrderedDict
from models.factory import  train, test_client, test_server, test
from flwr.common import Scalar, NDArrays
from typing import Dict
from flwr.common import Context
from models.factory import get_model
import numpy as np
from attacks.attack_manager import AttackManager
from utils import visualize_parameters
import hashlib
from utils import set_seed

class FlowerClient (fl.client.NumPyClient):
    """
    Class for the client implementation in Flower framework

    Methods
    ----------
    __init__
        initializes the client with config parameters, id and dataloaders
        
    ------
    hash_model_params
        prints the sha256 hash of the model parameters for easier identification of models
    ------
    get_parameters
        returns model parameters to the server and activates model poisoning attacks (model initialization attack)
    ------
    set_parameters
        sets model parameters from the client model state dict
    ------
    fit
        method for training on the client side and activating model poisoning attacks
    ------
    evaluate
        method for evaluation on the client side
        
    """
    def __init__(self,
                 cid,
                 model,
                 trainloader,
                 valloader,
                 num_classes,
                 save_path, 
                 seed, 
                 attack_manager=None,
                 malicious_clients=None,
                ) -> None:
        super().__init__()
        self.cid = int(cid)
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.attack_manager = attack_manager
        self.malicious_clients = malicious_clients if malicious_clients is not None else set()
        self.save_path= save_path
        set_seed(seed)
        print(self.device)


    def hash_model_params(self, parameters):
        """prints the sha256 hash of the model parameters for easier identification of models"""
        combined = np.concatenate([p.flatten() for p in parameters])
        print( hashlib.sha256(combined.tobytes()).hexdigest())
        
    def get_parameters(self, config:Dict[str, Scalar]):
        """returns model parameters to the server and activates model poisoning attacks (model initialization attack)"""
        print("CID: ", self.cid)

        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        #self.hash_model_params()
        if self.attack_manager and self.cid in self.malicious_clients:
        # get_parameters -> Stage "get_parameters"
            parameters = self.attack_manager.manipulate_parameters(
                old_parameters=parameters,
                new_parameters=parameters,
                client_id=self.cid,
                stage="get_parameters"
        )
        return parameters
    
    def set_parameters(self, parameters):
        """sets model parameters from the client model state dict"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """method for training on the client side and activating model poisoning attacks"""
        print("Client ID:", self.cid)
        
        self.set_parameters(parameters)
        
        old_parameters = [val.detach().cpu().clone().numpy().copy() for _, val in self.model.state_dict().items()]

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        current_round = config.get("round", None)
        
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)
        
        updated_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]  # Originale Modellparameter


        if self.attack_manager and self.cid in self.malicious_clients:
            updated_parameters = self.attack_manager.manipulate_parameters(
                old_parameters=old_parameters,
                new_parameters=updated_parameters,
                client_id=self.cid, 
                stage="fit",
                current_round=current_round
            )
        return updated_parameters, len(self.trainloader), {"client_id": self.cid}
    

    def evaluate(self, parameters: NDArrays, config):
        """method for evaluation on the client side"""
        self.set_parameters(parameters)
        loss, accuracy = test_client(net=self.model, testloader=self.valloader, num_classes=self.num_classes, device=self.device)
        
        return float(loss), len(self.valloader), {"accuracy": accuracy}
    
    
def generate_client_fn(trainloader, valloader, num_classes, model_name, dataset_name, malicious_clients,attack_manager,  save_path, seed=42):
    """generates clients with specific id, model, trainloader and valloader"""
    def client_fn(context: Context):
        set_seed(seed)
        cid = context.node_config["partition-id"]
        model = get_model(model_name, dataset_name, seed=seed)
        client = FlowerClient(cid=cid, model=model, trainloader=trainloader[int(cid)],
                            valloader=valloader[int(cid)],
                            num_classes=num_classes, 
                            save_path=save_path,
                            seed=seed,
                            malicious_clients=malicious_clients,
                            attack_manager=attack_manager)
        return client.to_client()
    
    return client_fn
