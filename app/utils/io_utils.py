import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def extract_data(path):
    """
    Extract data from a json file to a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_gantt_to_csv(gantt_history, clients_per_step_list, save_path):
    arrivals = {}
    for arrival_time, clients in enumerate(clients_per_step_list):
        for client in clients:
            client_id = client[0]
            arrivals[client_id] = arrival_time

    data = []
    for entry in gantt_history:
        client_id = entry['client_id']
        arrival_time = arrivals.get(client_id)
        start_time = int(round(entry['start']))
        end_time = int(round(entry['end']))
        c = entry['class']
        server_id = entry['server_id']
        estimated_proc = entry.get('estimated_proc_time', None)
        real_proc = entry.get('real_proc_time', None)

        data.append([
            client_id, arrival_time, start_time, end_time,
            c, server_id, estimated_proc, real_proc
        ])

    df = pd.DataFrame(data, columns=[
        'client_id', 'arrival_time', 'start_time', 'end_time',
        'class', 'server_id', 'estimated_proc_time', 'real_proc_time'
    ])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, sep=';')
    print(f"CSV sauvegardé: {save_path}")

def plot_gantt(gantt_data, n_servers, title="Diagramme de Gantt"):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab20").colors

    for task in gantt_data:
        start = task['start']
        duration = task['end'] - task['start']
        server = task['server']
        client = task['client']

        ax.broken_barh([(start, duration)], (10 * server, 9),
                       facecolors=colors[server % len(colors)], edgecolor='black')
        ax.text(start + duration / 2, 10 * server + 4.5, f"C{client}",
                ha='center', va='center', color='white',
                fontsize=9, fontweight='bold', clip_on=False)

    ax.set_yticks([10 * i + 4.5 for i in range(n_servers)])
    ax.set_yticklabels([f"Serveur {i}" for i in range(n_servers)])
    ax.set_xlabel("Temps")
    ax.set_title(title)
    ax.grid(True)
    plt.margins(x=0.05, y=0.05)
    plt.tight_layout()
    plt.show()


def plot_clients_per_time(clients_per_dt, dt, title="Nombre de clients en attente par pas de temps"):
    times = np.arange(len(clients_per_dt)) * dt
    plt.figure(figsize=(12, 4))
    plt.plot(times, clients_per_dt, marker='o', linestyle='-')
    plt.xlabel("Temps")
    plt.ylabel("Nombre de clients en attente")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_client_history_to_csv(client_history: dict, path: str, file_name: str) -> None:
    df = pd.DataFrame(client_history)
    os.makedirs(path, exist_ok=True)
    df.to_csv(path + '/' + file_name, index=False, sep=';')

def save_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: str,
    figsize: tuple = (10, 6),
    rotation: int = 45,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Generic function to create and save a boxplot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    x : str
        Column name for x-axis (categories)
    y : str
        Column name for y-axis (values)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    output_path : str
        Path where the plot will be saved
    figsize : tuple
        Figure size
    rotation : int
        Rotation of x-axis labels
    dpi : int
        Resolution of the saved figure
    show : bool
        Whether to display the plot
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot
    plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, data=df)
    plt.xticks(rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close()