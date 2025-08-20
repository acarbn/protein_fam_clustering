"""
Cluster Protein Sequences into Families and Visualize as Graph Networks

Author: Burcin Acar

Description:
    This script takes a CSV file of protein sequences, computes embeddings using 
    a specified ESM model, clusters the sequences into families based on embedding
    similarity, and visualizes the resulting clusters as network graphs using spring
    layout.
    
Features:
    - Validates and cleans input sequences
    - Generates or loads precomputed ESM embeddings
    - Computes pairwise distances using cosine similarity of the embeddings
    - Dynamically selects a clustering threshold
    - Visualizes sequence clusters as network graphs
    - Saves the cluster/community assignments alongside the original sequences
    - Displays network graphs interactively

Usage:
    python3 -m src.main <csv_file> [--model_name MODEL_NAME]

Example:
    python3 -m src.main data/seqs.csv --model_name esm2_t33_650M_local
"""

import pandas as pd
from .helper import load_data, check_data, check_sequences, clean_data, get_embeddings_batched
from .helper import distance_matrix, visualize_network_tsne, choose_threshold, visualize_network_plotly
import numpy as np
import os
import argparse

def main():
    # Load sequence data
    parser = argparse.ArgumentParser(description="Process a CSV file.")
    parser.add_argument(
        "csv_file", 
        type=str, 
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./model/esm2_t33_650M_local",
        help="Model name (default: ./model/esm2_t33_650M_local)"
    )
    args = parser.parse_args()
    print("Selected model is:  ", args.model_name)
    input_data = load_data(args.csv_file)
    print("Protein sequences are loaded.\n")

    # Check and clean input data
    os.makedirs("results", exist_ok=True)
    check_data(input_data)
    df=check_sequences(input_data)
    df.drop("only_natural_aas",axis=1, inplace=True)
    df = clean_data(df)
    print("Null data and replicates are removed.\n")
    check_data(df)

    # Calculate or load embeddings for sequences
    embed_path="data/"+args.model_name+"_embeddings.npy"

    if os.path.exists(embed_path):
        print("Embeddings were already saved. Loading the npy file...\n")
        embeddings = np.load(embed_path)
        df["embedding"] = [emb.tolist() for emb in embeddings]

    else:
        sequences = df["original_sequence"].tolist()
        embeddings = get_embeddings_batched(sequences, batch_size=16,local_model_dir ="./model/"+args.model_name)
        df["embedding"] = [emb.tolist() for emb in embeddings]
        np.save(embed_path, embeddings)
    
    # Calculate pairwise protein distances based on cosine sim of embeddings
    dist_mat=distance_matrix(df,model_name=args.model_name)   

    # Choose a threshold distance for clustering
    threshold = choose_threshold(dist_mat,model_name=args.model_name)
    print("Chosen dynamic threshold:", threshold,"\n")

    # Visualize networks
    df=visualize_network_tsne(df, dist_mat, threshold,model_name=args.model_name)
    str_threshold=str(threshold)

    # Save community numbers next to protein seq
    df.drop("embedding",axis=1, inplace=True)
    df.to_csv("results/"+args.model_name+"_seqs_with_communities.csv", index=False)

    # Display networks interactivelt
    visualize_network_plotly(df, dist_mat, threshold,model_name=args.model_name)

if __name__ == "__main__":
    main()
