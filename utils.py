

import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import numpy as np
import torch
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import numpy as np
import random
import os


def set_seed(seed: int):
    """sets the seed for all used libraries and deterministic algorithms to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)



DATASET_INFO = {
    "mnist": {
        "input_channels": 1,
        "input_size": (28, 28),
        "num_classes": 10,
        "greyscale": True,
    },
    "fmnist": {
        "input_channels": 1,
        "input_size": (28, 28),
        "num_classes": 10,
        "greyscale": True,
    },
    "cifar10": {
        "input_channels": 3,
        "input_size": (32, 32),
        "num_classes": 10,
        "greyscale": False,
    },
    "pathmnist": {
        "input_channels": 3,
        "input_size": (28, 28),
        "num_classes": 9,
        "greyscale": False,
    },
    
    "dermamnist": {
        "input_channels": 3,
        "input_size": (28, 28),
        "num_classes": 7,
        "greyscale": False,
    },
    
    
}


def save_results_as_csv(metrics, results_path):
    """generates csv file for one meausrement with alle relevant metrics in one file"""
    output_dir = Path(results_path)
    metrics_csv_path = output_dir / "federated_metrics.csv"
    df = pd.DataFrame()
    for metric_name, values in metrics.items():
        rounds, metric_values = zip(*values)
        df[metric_name] = metric_values

    df.index = rounds
    df.index.name = "Round"

    # In CSV speichern
    df.to_csv(metrics_csv_path)

def plot_precision_over_rounds(history, num_classes, results_path):
    """plots precision for each class over the rounds"""
    output_dir = Path(results_path) / "precision_over_rounds.png"
    rounds = sorted(set(r for r, _ in history['accuracy']))

    colors = sns.color_palette("colorblind", num_classes)
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H'] 

    plt.figure(figsize=(10, 6))
    for cls_idx in range(num_classes):
        precision_key = f'precision_class_{cls_idx}'
        if precision_key in history:
            values = [v for r, v in history[precision_key]]
            plt.plot(rounds, values, label=f'Class {cls_idx}', 
                     color=colors[cls_idx], marker=markers[cls_idx % len(markers)], 
                     markersize=8, linewidth=2)


    plt.xlabel('Round', fontsize=10, fontweight='bold')
    plt.ylabel('Precision', fontsize=10, fontweight='bold')
    plt.grid(True)

    leg = plt.legend(frameon=True, fontsize=11, loc='best', facecolor='white', edgecolor='black')
    for legobj in leg.get_lines():
        legobj.set_linewidth(3)  
        legobj.set_markersize(10)  

    plt.tight_layout()
    plt.savefig(output_dir, bbox_inches='tight')
    plt.close()



def plot_metrics_over_rounds(history, num_classes, results_path):
    """plots F1 Score and Recall for each class over the rounds in two subplots"""
    colors = sns.color_palette("colorblind", num_classes)
    output_dir = Path(results_path) / "f1_recall_over_rounds.png"
    rounds = sorted(set(r for r, _ in history['accuracy']))
    step = max(1, len(rounds) // 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H']  # verschiedene Markerformen

    # F1 Score
    for cls_idx in range(num_classes):
        f1_key = f'f1_class_{cls_idx}'
        if f1_key in history:
            values = [v for r, v in history[f1_key]]
            axes[0].plot(rounds, values, label=f'Class {cls_idx}', 
                         color=colors[cls_idx], marker=markers[cls_idx % len(markers)], markersize=8, linewidth=2)
    axes[0].set_xlabel('Round', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontsize=10, fontweight='bold')
    axes[0].set_xticks(range(min(rounds), max(rounds) + 1, step))
    axes[0].grid(True)

    # Recall
    for cls_idx in range(num_classes):
        recall_key = f'recall_class_{cls_idx}'
        if recall_key in history:
            values = [v for r, v in history[recall_key]]
            axes[1].plot(rounds, values, label=f'Class {cls_idx}', 
                         color=colors[cls_idx], marker=markers[cls_idx % len(markers)], markersize=8, linewidth=2)
    axes[1].set_xlabel('Round', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Recall', fontsize=10, fontweight='bold')
    axes[1].set_xticks(range(min(rounds), max(rounds) + 1, step))
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(0.92, 0.5),
               borderaxespad=0,
               frameon=True,
               framealpha=0.9,
               facecolor='white',
               fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_dir, bbox_inches='tight')
    plt.close()




def plot_final_results(history, num_classes, results_path):
    """plots final F1 Score and Recall for each class of the final training round as bar chart"""
    output_dir = Path(results_path) / "final_f1_recall_over_rounds.png"
    last_round = max(r for r, _ in history['accuracy'])

    f1_scores = []
    recalls = []
    classes = list(range(num_classes))

    for cls_idx in classes:
        f1_key = f'f1_class_{cls_idx}'
        recall_key = f'recall_class_{cls_idx}'

        # F1 
        if f1_key in history:
            f1_val = next(v for r, v in history[f1_key] if r == last_round)
        else:
            f1_val = 0
        f1_scores.append(f1_val)

        # Recall 
        if recall_key in history:
            recall_val = next(v for r, v in history[recall_key] if r == last_round)
        else:
            recall_val = 0
        recalls.append(recall_val)

    x = range(num_classes)

    plt.figure(figsize=(10, 6))
    width = 0.35

    plt.bar(x, f1_scores, width=width, label='F1 Score', alpha=0.7)
    plt.bar([i + width for i in x], recalls, width=width, label='Recall', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Final F1 Score and Recall per Class (Round {last_round})')
    plt.xticks([i + width / 2 for i in x], classes)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(output_dir)
    plt.close()

def plot_accuracy(metrics, results_path):
    """generates a line plot for accuracy over the rounds"""
    accuracy = metrics["accuracy"]
    rounds, acc_values = zip(*accuracy)
    
    output_dir = Path(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, acc_values, marker="o", label="Accuracy", color="blue")
    plt.xlabel("Rounds", fontsize=10, fontweight='bold')
    plt.legend()
    plt.grid(True)

    accuracy_plot_path = output_dir / "accuracy_plot.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"Accuracy plot saved to {accuracy_plot_path}")
    
    
    
def plot_metrics(metrics, num_classes, result_path):
    """plotter function which calls all other plot functions"""
    label_dist_dir = result_path / "metric_plots"
    label_dist_dir.mkdir(parents=True, exist_ok=True)
    
    plot_accuracy(metrics,label_dist_dir)
    plot_final_results(metrics, num_classes, label_dist_dir)
    plot_metrics_over_rounds(metrics, num_classes, label_dist_dir)
    plot_precision_over_rounds(metrics, num_classes, label_dist_dir)


def plot_label_distribution(datasets, num_classes, output_dir, prefix="client"):
    """plots bar charts for label distribution of the clients"""
    
    number_of_plots = 5 #number of clients to plot (to avoid too many plots)
    os.makedirs(output_dir, exist_ok=True)
    for i, subset in enumerate(datasets[:number_of_plots]):
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
        label_counts = [0] * num_classes

        for x, y in loader:
            for label in y:
                label_counts[label.item()] += 1

        plt.figure()
        bars = plt.bar(range(num_classes), label_counts, color="skyblue")
        plt.xlabel("Class")
        plt.ylabel("Number")
        plt.title(f"Label Distribution- Client {i}")
        plt.xticks(range(num_classes))
        plt.tight_layout()
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(height),
                     ha='center', va='bottom', fontsize=8)
        path = os.path.join(output_dir, f"{prefix}_client_{i}.png")
        plt.savefig(path)
        plt.close()


def get_label_distributions_from_loaders(trainloaders, num_classes):
    """ Returns a list of Dicts – each Dict contains label -> count for a client"""
    client_label_counts = []

    for loader in trainloaders:
        label_count = defaultdict(int)
        for _, labels in loader:
            for label in labels:
                label_count[int(label)] += 1
        for label in range(num_classes):
            if label not in label_count:
                label_count[label] = 0
        client_label_counts.append(dict(label_count))

    return client_label_counts



def plot_label_distribution_heatmap(client_label_counts, save_path=None):
    """plots a heatmap for label distribution of all clients"""
    num_clients = len(client_label_counts)
    num_classes = len(client_label_counts[0])

    matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_idx, label_count in enumerate(client_label_counts):
        for label, count in label_count.items():
            matrix[client_idx][int(label)] = count

    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, annot=False, cmap="Blues", cbar=True)
    plt.xlabel("Classes")
    plt.ylabel("Clients")
    plt.title("Heatmap: Class Distribution per Client")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()


def plot_class_presence_histogram(label_distributions, num_classes, save_path):
    """plots a histogram showing in how many clients each class is present"""
    presence = [0] * num_classes
    for label_count in label_distributions:
        for label in label_count:
            if label_count[label] > 0:
                presence[int(label)] += 1

    plt.figure(figsize=(8, 5))
    plt.bar(range(num_classes), presence, color="coral")
    plt.xlabel("Class")
    plt.ylabel("Number of Clients with Samples")
    plt.title("Class Presence Across Clients")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def handle_label_distribution_plots(trainloaders, num_classes, save_path):
    """Handles all label distribution plots and saves them to the given path"""
    label_dist_dir = save_path / "label_distributions"
    label_dist_dir.mkdir(parents=True, exist_ok=True)

    # Einzelne Clients (Balkendiagramme)
    train_subsets = [loader.dataset for loader in trainloaders]
    plot_label_distribution(
        datasets=train_subsets,
        num_classes=num_classes,
        output_dir=label_dist_dir,
        prefix="train"
    )
    
    label_distributions = get_label_distributions_from_loaders(trainloaders, num_classes)
    heatmap_path = save_path / "label_distributions" / "label_distribution_heatmap.png"
    plot_label_distribution_heatmap(label_distributions, save_path=heatmap_path)
    
    presence_hist_path = label_dist_dir / "class_presence_histogram.png"
    plot_class_presence_histogram(label_distributions, num_classes, presence_hist_path)



def save_malicious_clients(malicious_clients, total_clients, results_path):
    """generates a csv file listing all clients and whether they are malicious or not"""
    output_dir = Path(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "malicious_clients.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "is_malicious"])
        for client_id in range(total_clients):
            writer.writerow([client_id, client_id in malicious_clients])



def aggregate_histories(histories):
    """
    Aggregates multiple metric_histories (each a dict of {metric_name: List[values]})
    to the mean and standard deviation across repetitions.
    """
    aggregated = {}

    metric_keys = histories[0].keys()
    for key in metric_keys:
        values_per_round = [hist[key] for hist in histories]  # List of lists
        values_per_round = np.array(values_per_round)  # shape: (num_runs, num_rounds)
        mean = np.mean(values_per_round, axis=0)
        std = np.std(values_per_round, axis=0)
        aggregated[key] = {"mean": mean, "std": std}

    return aggregated


def plot_aggregated_metrics_over_rounds(aggregated_metrics, save_path: Path):
    """plots aggregated metrics (mean & std) over the rounds for each metric"""
    
    for metric_name, data in aggregated_metrics.items():
        mean = data["mean"][:, 1]  
        std = data["std"][:, 1]    
        rounds = data["mean"][:, 0]  

        plt.figure()
        plt.plot(rounds, mean, label=f"{metric_name} (mean)")
        plt.fill_between(rounds, mean - std, mean + std, alpha=0.3, label="±1 std")
        plt.xlabel("Round",fontsize=10, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)  
        plt.tight_layout()
        plt.savefig(save_path / f"{metric_name}_aggregated.png")
        plt.close()





def plot_aggregated_final_results(aggregated_metrics,num_classes,  save_path: Path):
    """plots final aggregated results of the last epoch (mean & std) for each class as bar chart"""
    class_ids = list(range(num_classes))
    metrics = ['f1', 'recall', 'precision']
    width = 0.2
    x = np.arange(len(class_ids))  

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, metric in enumerate(metrics):
        means = []
        stds = []

        for cls_id in class_ids:
            key = f"{metric}_class_{cls_id}"
            mean_value = aggregated_metrics[key]["mean"][-1][1]
            std_value = aggregated_metrics[key]["std"][-1][1]
            means.append(mean_value)
            stds.append(std_value)

        
        ax.bar(x + i * width, means, width, yerr=stds, capsize=5, label=metric.capitalize())

    ax.set_xticks(x + width) 
    ax.set_xticklabels([f"Class {cls}" for cls in class_ids])
    ax.set_ylabel("Score")
    ax.set_title("Final Aggregated Scores per Class (Mean ± Std)")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path / "final_aggregated_results.png")
    plt.close()



def save_aggregated_metrics_to_csv(aggregated_metrics, save_path: Path):
    """
    Saves all aggregated metrics (mean & std) collected in **a CSV file**.
    Each row is one round, columns are e.g.:
    Round, accuracy_mean, accuracy_std, recall_class_0_mean, recall_class_0_std, ...
    """
    all_rounds = None
    result_dict = {}

    for metric_name, values in aggregated_metrics.items():
        mean = values["mean"]
        std = values["std"]

        if mean.ndim == 2 and mean.shape[1] == 2:
            rounds = mean[:, 0].astype(int)
            mean_values = mean[:, 1]
            std_values = std[:, 1]
        else:
            rounds = np.arange(len(mean))
            mean_values = mean
            std_values = std

        if all_rounds is None:
            all_rounds = rounds
            result_dict["Round"] = all_rounds

        result_dict[f"{metric_name}_mean"] = mean_values
        result_dict[f"{metric_name}_std"] = std_values

    df = pd.DataFrame(result_dict)
    df.to_csv(save_path / "aggregated_metrics_all_in_one.csv", index=False)
    
    

def visualize_parameters(original_parameters, manipulated_parameters, save_path):
    """visualizes the change in model parameters before and after an attack"""

    plt.figure(figsize=(12, 6))
    
    # Calculation of the mean values of the parameters    
    orig_means = [p.mean() for p in original_parameters]
    man_means = [p.mean() for p in manipulated_parameters]
    
    print(f"Original Means: {orig_means}")
    print(f"Manipulated Means: {man_means}")
    
    # Scatter plot for direct comparison    
    indices = range(len(orig_means)) 
    plt.scatter(indices, orig_means, label="Original Parameters", color="blue", alpha=0.7)
    plt.scatter(indices, man_means, label="Manipulated Parameters", color="orange", alpha=0.7)
    
    for i, (orig, man) in enumerate(zip(orig_means, man_means)):
        plt.plot([i, i], [orig, man], color="gray", linestyle="--", alpha=0.5)
    
    plt.legend()
    plt.title("Parameter Comparison: Original vs Manipulated")
    plt.xlabel("Parameter Index")
    plt.ylabel("Mean Value")
    plt.grid(True)
    plt.savefig(save_path / f"parameter_attack.png", bbox_inches='tight')
    plt.close()
    
    


def plot_aggregated_metrics_grouped(aggregated_metrics: dict, save_path: Path):
    """plots aggregated metrics (mean & std) over the rounds grouped with all classes for each metric,"""
    grouped_metrics = defaultdict(dict)

    # Group by metric type (precision, recall, f1)    
    for metric_key, data in aggregated_metrics.items():
        if "class_" in metric_key:
            metric_type, class_key = metric_key.split("_class_")
            grouped_metrics[metric_type][f"class_{class_key}"] = data


    class_keys = sorted({key for metrics in grouped_metrics.values() for key in metrics})
    num_classes = len(class_keys)
    colors = sns.color_palette("colorblind", num_classes)
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H']

    # For each metric one plot
    for metric_type, class_dict in grouped_metrics.items():
        plt.figure(figsize=(10, 6))

        for idx, (class_key, data) in enumerate(sorted(class_dict.items())):
            rounds = data["mean"][:, 0]
            mean = data["mean"][:, 1]
            std = data["std"][:, 1]

            plt.plot(rounds, mean,
                     label=class_key,
                     color=colors[idx],
                     marker=markers[idx % len(markers)],
                     markersize=8,
                     linewidth=2)
            plt.fill_between(rounds, mean - std, mean + std,
                             color=colors[idx], alpha=0.2)

        plt.xlabel("Round", fontsize=10, fontweight='bold')
        plt.ylabel(metric_type.capitalize(), fontsize=10, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True)

        plt.legend(title="Class", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left',
                   borderaxespad=0., frameon=True, framealpha=0.9, facecolor='white')

        plt.tight_layout(rect=[0, 0, 0.85, 1])  
        plt.savefig(save_path / f"{metric_type}_aggregated.png", bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path=None):
    """Plots and optionally saves the confusion matrix."""
    output_dir = Path(save_path)
    cm_path = output_dir / "confusion_matrix.png"
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)))
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if cm_path:
        plt.savefig(cm_path)
        plt.close()
