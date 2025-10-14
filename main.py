
import hydra
import pickle
import flwr as fl
import requests
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from utils import  handle_label_distribution_plots, plot_metrics, save_results_as_csv, save_aggregated_metrics_to_csv, plot_final_results, aggregate_histories, plot_aggregated_metrics_over_rounds,plot_aggregated_metrics_grouped, plot_aggregated_final_results
from utils import DATASET_INFO
from models.factory import get_model
from flwr.common import ndarrays_to_parameters
from attacks.model_scaling import ModelScalingAttack
from attacks.distributed_backdoor import DTriggeredDataset
from attacks.centralized_backdoor import CTriggeredDataset 
from attacks.gaussian_random_attack import GaussianRandomAttack 
from aggregation_wrapper import DeterministicAggregationWrapper
from aggregators.NormClipping import NormClippingFedAvg
from aggregators.Clustering import SimpleClusteringFedAvg
from omegaconf import ListConfig


from attacks.attack_manager import AttackManager
from utils import set_seed

def get_initial_parameters(cfg):
    """generates initial model parameters for FedOpt and FedAdam strategies"""

    model = get_model(cfg.model, cfg.dataset)
    model.eval()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(weights)

def get_malicious_client_fraction(cfg):
    """returns the fraction of malicious clients from the attack config, if no attack is defined, returns 0.0"""

    if not hasattr(cfg, 'attacks'):
        return 0.0
    for attack_cfg in cfg.attacks.values():
        fraction = getattr(attack_cfg, 'malicious_client_fraction', None)
        if fraction is not None:
            return fraction
    return 0.0


def build_common_args(cfg, num_classes, testloader, trigger_testloader=None, save_path=None, seed=42):
    """sets and returns static parameters for Flower strategies and handles special parameters for certain strategies"""

    args = {
        "fraction_fit": 0.0001,
        "min_fit_clients": cfg.num_clients_per_round_fit,
        "fraction_evaluate": 0.0001,
        "min_evaluate_clients": cfg.num_clients_per_round_eval,
        "min_available_clients": cfg.num_clients,
        "on_fit_config_fn": get_on_fit_config(cfg.config_fit),
        "evaluate_fn": get_evaluate_fn(cfg.model, cfg.dataset, num_classes, testloader, trigger_testloader,  max_rounds=cfg.num_rounds, save_path=save_path, seed=seed),
    }
    
    if cfg.aggregation_strategy in ["Krum", "MultiKrum"]:
        num_clients = getattr(cfg, 'num_clients', 20)
        malicious_client_fraction = get_malicious_client_fraction(cfg)

        num_malicious_clients = int(malicious_client_fraction * num_clients)
        args["num_malicious_clients"] = num_malicious_clients
        print(f"Malicious Client Fraction: {malicious_client_fraction}, Num Malicious Clients: {args['num_malicious_clients']}")
        
        if cfg.aggregation_strategy == "MultiKrum":
            num_clients_to_keep = num_clients - num_malicious_clients
            args["num_clients_to_keep"] = num_clients_to_keep
            print(f"[BUILD] -> Num Clients To Keep (Multi-Krum): {num_clients_to_keep}")
            

    if cfg.aggregation_strategy in ["FedAdam", "FedOpt", "FedAvgM"]:
        args["initial_parameters"] = get_initial_parameters(cfg)
    
    if cfg.aggregation_strategy == "NormClipFedAvg":
        args["clip_norm"] = getattr(cfg, "clip_norm", 5.0)
        
    if cfg.aggregation_strategy == "SimpleClustering":
        args["clustering"] = getattr(cfg, "clustering", "DBSCAN")  # default DBSCAN
        print(f"[BUILD] -> SimpleClustering using {args['clustering']}")
        
    if cfg.aggregation_strategy == "FedAvgM":
        args["server_momentum"] = getattr(cfg, "server_momentum", 0.9)  # default 0.9

    return args

def get_strategy(cfg, common_args):
    """returns the selected Flower strategy based on the config file and initializes it with the args parameters"""

    strategy_map = {
        "FedAvg": fl.server.strategy.FedAvg,
        "Krum": fl.server.strategy.Krum,
        "MultiKrum": fl.server.strategy.Krum,
        "FedAdam": fl.server.strategy.FedAdam,
        "FedOpt": fl.server.strategy.FedOpt,
        "FedAvgM": fl.server.strategy.FedAvgM,
        "FedTrimmedAvg": fl.server.strategy.FedTrimmedAvg,
        "NormClipFedAvg": NormClippingFedAvg,
        "FedMedian": fl.server.strategy.FedMedian,
        "SimpleClustering": SimpleClusteringFedAvg,
    }

    strategy_class = strategy_map.get(cfg.aggregation_strategy)
    if strategy_class is None:
        raise ValueError(f"Unknown Strategy: {cfg.aggregation_strategy}")

    return strategy_class(**common_args)


def send_ntfy_message(topic, message):
    """sends a notification message to ntfy.sh"""

    url = f"https://ntfy.sh/{topic}"
    headers = {'Title': 'Experiment', 'Priority': 'urgent'}  
    requests.post(url, data=message.encode('utf-8'), headers=headers)
    
def run_experiment(cfg: DictConfig, save_path: Path):
    """runs a single experiment based on the provided config and saves results to the specified path"""

    print(OmegaConf.to_yaml(cfg))
    save_path.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    trainloaders, validationloaders, testloader, bdtestloader, num_classes, malicious_clients = prepare_dataset(cfg.dataset,cfg.model,cfg.num_clients,
                                                                   cfg.batch_size, 
                                                                   partitioning=cfg.partitioning, attacks=cfg.attacks,
                                                                   seed = cfg.seed,    
                                                                   save_path = save_path)
    
    config_path = save_path / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    handle_label_distribution_plots(trainloaders, num_classes, save_path)
    
    model_scaling_attack=None

    
    attack_manager = AttackManager(
    config=cfg.attacks,
    total_clients=cfg.num_clients,
    seed=cfg.seed,
    save_path=save_path,
    testset=None,  # only relevant for backdoor
    num_classes=num_classes
    )
    

    client_fn = generate_client_fn(
    trainloader=trainloaders,
    valloader=validationloaders,
    num_classes=num_classes,
    model_name=cfg.model,
    dataset_name=cfg.dataset,
    malicious_clients=malicious_clients,
    attack_manager=attack_manager,
    save_path=save_path,
    seed=cfg.seed
    ) 

    print("Build common args")
    common_args = build_common_args(cfg, num_classes, testloader, trigger_testloader=bdtestloader, save_path= save_path, seed=cfg.seed)
    base_strategy = get_strategy(cfg, common_args)
    strategy = DeterministicAggregationWrapper(base_strategy)

    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),strategy=strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0},
    )
    

    results_path = Path(save_path) / "results.pkl"
    
    results = {'history': history.metrics_centralized }
    

    save_results_as_csv(results["history"], save_path)
    plot_metrics(history.metrics_centralized, num_classes, results_path.parent)
    
    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
        
        

# Hydra main function to run experiments based on config files
# Based on each config file, run multiple experiments with different seeds

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(_cfg: DictConfig):
    from omegaconf import OmegaConf

    original_cwd = Path(hydra.utils.get_original_cwd())
    config_dir = original_cwd / "configs"
    config_files = sorted(config_dir.glob("*.yaml"))

    for config_file in config_files:
        print(f"\n🔧 Load Configuration: {config_file.name}")
        cfg = OmegaConf.load(config_file)
        if isinstance(cfg.seed, int):
            base_seed = cfg.seed
            num_repeats = cfg.get("repeat_experiment", 1)
            seeds = [base_seed + i for i in range(num_repeats)]
        elif isinstance(cfg.seed, (ListConfig)):
            seeds = list(cfg.seed)
        else:
            raise ValueError(f"Unsupported type for seeds: {type(cfg.seeds)}")

        print(f"Seeds for experiments: {seeds}")
        output_dir = Path(HydraConfig.get().runtime.output_dir)

        for seed in seeds:
            print(f"\n🚀 Start experiment with Seed {seed} for config: {config_file.name}")
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            run_cfg.seed = seed

            save_path = output_dir / config_file.stem / f"seed_{seed}"
            run_experiment(run_cfg, save_path)

        # After the run: aggregate
        histories = []
        for seed in seeds:
            path = output_dir / config_file.stem / f"seed_{seed}" / "results.pkl"
            with open(path, "rb") as f:
                result = pickle.load(f)
                histories.append(result["history"])

        aggregated_metrics = aggregate_histories(histories)
        # Save and plot aggregated results
        save_aggregated_metrics_to_csv(aggregated_metrics, output_dir / config_file.stem)
        plot_aggregated_metrics_over_rounds(aggregated_metrics, output_dir / config_file.stem)
        plot_aggregated_final_results(aggregated_metrics, DATASET_INFO[cfg.dataset]["num_classes"], output_dir / config_file.stem)
        plot_aggregated_metrics_grouped(aggregated_metrics, output_dir / config_file.stem)
        send_ntfy_message("master_thesis2025", f"Experiment mit {config_file.name} fertig! ✅")


if __name__ == "__main__":
    main()
