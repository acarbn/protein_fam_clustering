"""
Protein Sequence Clustering and Network Visualization Helper Module

Author: Burcin Acar

Description:
    This module provides functions to cluster protein sequences into families
    based on embeddings from ESM protein language models and visualize the
    resulting clusters as network graphs.

Key Functions:
    - load_data: Load a CSV file of protein sequences into a DataFrame.
    - check_data: Print basic info and null values of the DataFrame.
    - check_sequences: Validate sequences for natural amino acids and plot length distribution.
    - clean_data: Remove nulls and duplicate sequences.
    - get_embeddings_batched: Generate ESM embeddings in batches for memory efficiency.
    - distance_matrix: Compute pairwise distances between embeddings and plot a heatmap.
    - choose_threshold: Select a dynamic distance threshold for clustering based on histogram analysis.
    - visualize_network_tsne: Create a network graph of sequences, detect communities using Louvain clustering, 
      and visualize the clusters (note: uses spring_layout).

Usage:
    Import this module in your main clustering script and call the functions as needed.
    Example:
        import helper
        df = helper.load_data("data/sequences.csv")
        embeddings = helper.get_embeddings_batched(df["original_sequence"].tolist())
        dist_mat = helper.distance_matrix(df)
        threshold = helper.choose_threshold(dist_mat)
        df = helper.visualize_network_tsne(df, dist_mat, threshold)
"""

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import networkx as nx
from networkx.algorithms.community import louvain_communities
import torch
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.colors as mcolors


pio.renderers.default = "browser"   # open interactive graph in default browser

aa_dict = set("ARNDCEQGHILKMFPSTWYV")

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def check_data(df):
    print("First five rows of the data: \n", df.head(),"\n")
    print("Shape of the data: ", df.shape,"\n")
    print("Null values in the data: \n", df.isnull().sum(),"\n")

def check_sequences(df):
    print("Data on sequence lengths:",df["original_sequence"].apply(len).describe(),"\n")
    sequence_lengths = df["original_sequence"].apply(len)

    plt.hist(sequence_lengths, bins=30, edgecolor='black')
    plt.xlabel("Sequence length")
    plt.ylabel("Counts")
    plt.title("Distribution of Sequence Lengths")
    plt.grid(True)
    plt.savefig("results/aalen.png", dpi=300, bbox_inches="tight")

    plt.show()

    df["only_natural_aas"] = df["original_sequence"].apply(lambda s: set(s) <= aa_dict)
    
    if (~df["only_natural_aas"]).any():  # checks if any value is False
        print("Unnatural amino acid exists in:", df[~df["only_natural_aas"]],"\n")
    else:
        print("No unnatural amino acid found in data.","\n")

    return df 


def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates(subset=["original_sequence"])
    return df

def get_embeddings_batched(sequences, batch_size=16, local_model_dir="./model/esm2_t33_650M_local"):
    """Return embeddings for a list of sequences in batches to save memory."""
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModel.from_pretrained(local_model_dir)
    model.eval()
    model.to(device)
    
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)  # shape: (num_sequences, hidden_dim)


def distance_matrix(df, method="cosine",model_name="esm2_t33_650M_local"):
    """
    Compute a distance matrix from sequence embeddings in a DataFrame.

    Parameters:
    - df: pd.DataFrame, must contain a column "embedding" with lists/arrays
    - method: str, distance metric ("cosine", "euclidean", etc.)

    Returns:
    - dist_mat: np.ndarray, shape (n_sequences, n_sequences)
    """
    # Convert embeddings column to 2D array
    emb_array = np.vstack(df["embedding"].values)

    # Compute pairwise distances
    dist_vec = pdist(emb_array, metric=method)
    dist_mat = squareform(dist_vec)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
    dist_mat,
    cmap="viridis",
    annot=False,     # Turn off annotation for large matrices
    cbar=True
    )
    plt.title("Heatmap of Cosine Distance Matrix")
    plt.xlabel("Sequence ID")
    plt.ylabel("Sequence ID")

 # Force ticks and labels at step=50
    plt.xticks(
    ticks=range(0, dist_mat.shape[1], 50),
    labels=range(0, dist_mat.shape[1], 50),
    rotation=90
    )
    plt.yticks(
    ticks=range(0, dist_mat.shape[0], 50),
    labels=range(0, dist_mat.shape[0], 50),
    rotation=0
    )

    plt.savefig("results/"+model_name+"_distmat.png", dpi=300, bbox_inches="tight")
    plt.show()


    return dist_mat

def choose_threshold(dist_mat,model_name="esm2_t33_650M_local"):
# Example: dist_mat is your 411x411 distance matrix
    dist_values = dist_mat[np.triu_indices_from(dist_mat, k=1)]  # Upper triangle (excludes diagonal)
    plt.figure(figsize=(10, 6))
    plt.hist(dist_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pairwise Distances (411 Nodes)')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add vertical lines for thresholds (e.g., mean, median, percentiles)
    plt.axvline(np.mean(dist_values), color='red', linestyle='--', label=f'Mean: {np.mean(dist_values):.2f}')
    plt.axvline(np.mean(dist_values)-np.std(dist_values), color='green', linestyle='--', label=f'Threshold: {(np.mean(dist_values)-np.std(dist_values)):.2f}')
    plt.legend()
    plt.savefig("results/"+model_name+"_threshold.png", dpi=300, bbox_inches="tight")

    plt.show()

    #threshold = np.mean(dist_values)
    #threshold=np.percentile(dist_values,20)
    threshold=np.mean(dist_values)-np.std(dist_values)
    #percentile = np.sum(dist_values <= 0.075) / len(dist_values) * 100  
    #print("percentile: ",percentile)    
    return threshold


def visualize_network_tsne(df, dist_mat, threshold,model_name="esm2_t33_650M_local"):

# Simulate the data and create the graph (as in previous response) ---

    np.fill_diagonal(dist_mat, 0)
    dist_mat = (dist_mat + dist_mat.T) / 2
    num_proteins=len(df)
    G = nx.Graph()
    G.add_nodes_from(range(num_proteins))

    for i in range(num_proteins):
        for j in range(i + 1, num_proteins):
            if dist_mat[i, j] < threshold:
                similarity = 1 - dist_mat[i, j]
                G.add_edge(i, j, weight=similarity)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

    communities = louvain_communities(G, weight='weight',seed=42)
    print(f"Found {len(communities)} communities with sizes: {[len(c) for c in communities]}")

# Create a consistent mapping from node to community index
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}

    df["community"] = df.index.map(node_to_community)

# Get nodes in consistent order
    nodes = list(G.nodes())

# Create color list matching node order - updated colormap syntax
    cmap = plt.colormaps['gist_ncar'].resampled(len(communities))
    node_colors = [cmap(node_to_community[node]) for node in nodes]

# Get positions ---
    pos = nx.spring_layout(G, k=0.5, iterations=50,seed=42,weight='weight')

# Plot with explicit ordering ---
    plt.figure(figsize=(15, 15))
    ax = plt.gca()  # Get current Axes instance

# Draw network
    nx.draw_networkx_nodes(
                        G, pos, 
                        nodelist=nodes,
                        node_color=node_colors,
                        edgecolors="black",
                        linewidths=0.8,  node_size=50, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)

    plt.title("Protein Sequence Network - Correct Community Coloring")
    plt.axis('off')

# Create colorbar with proper Axes reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(communities)-1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, ticks=range(len(communities)), label='Community', shrink=0.7)
    plt.savefig("results/"+model_name+"_clusters.png", dpi=300, bbox_inches="tight")

    plt.show()

    return df


def visualize_network_plotly(df, dist_mat, threshold, model_name="esm2_t33_650M_local"):

    # Build Graph
    np.fill_diagonal(dist_mat, 0)
    dist_mat = (dist_mat + dist_mat.T) / 2
    num_proteins = len(df)

    G = nx.Graph()
    G.add_nodes_from(range(num_proteins))

    for i in range(num_proteins):
        for j in range(i + 1, num_proteins):
            if dist_mat[i, j] < threshold:
                similarity = 1 - dist_mat[i, j]
                G.add_edge(i, j, weight=similarity)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

    # Communities
    communities = louvain_communities(G, weight='weight', seed=42)
    print(f"Found {len(communities)} communities with sizes: {[len(c) for c in communities]}")

    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    df["community"] = df.index.map(node_to_community)

    # Positions 
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42, weight='weight')

    # Extract edges for Plotly 
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(node_to_community[node])
        # hover text: index + sequence snippet (if present)
        seq_info = df.loc[node, "original_sequence"][:15] + "..." if "original_sequence" in df else ""
        node_text.append(f"ID: {df.index[node]}<br>Community: {node_to_community[node]}<br>{seq_info}")

    cmap = plt.cm.gist_ncar
    # Convert to hex colors for Plotly
    colorscale = [[i/(len(communities)-1), mcolors.to_hex(cmap(i/(len(communities)-1)))] for i in range(len(communities))]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=node_color,
            size=8,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                title=dict(
                    text='Community',
                    side='right'   # ✅ valid here
                    )
            ),
            line=dict(width=0.5, color='black')
        )
    )

    # Build Figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Protein Sequence Network - {model_name}",
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.write_html(f"results/{model_name}_clusters_interactive.html")
    print(f"The interactive fig is saved as results/{model_name}_clusters_interactive.html")
    fig.show()

