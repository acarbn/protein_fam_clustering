"""
Compare Computed Communities with PFAM Families

Author: Burcin Acar

Description:
    This script merges computed community assignments from a protein clustering
    analysis with known PFAM family annotations. It extracts PFAM IDs from TSV
    files, associates them with the corresponding protein sequences, and produces
    a combined CSV file containing each protein's ID, sequence, assigned community,
    and PFAM families.

Workflow:
    1. Extract PFAM IDs from all TSV files in 'eval/pfam/'.
    2. Expand multiple PFAMs per sequence into separate rows.
    3. Group by sequence ID to create a comma-separated list of PFAMs.
    4. Merge PFAM information with the clustering results CSV.
    5. Save the merged results to 'eval/<model_name>_proteins_with_pfam.csv'.

Usage:
    python3 -m src.compare_communities_pfam <model_name>

Example:
    python -m src.compare_communities_pfam esm2_t33_650M_local
"""

import pandas as pd
import glob
import re
import argparse

parser = argparse.ArgumentParser(description="Extract model name.")

parser.add_argument(
        "model_name",
        type=str,
        help="Model name given to download "
    )
args = parser.parse_args()


# === Step 1: Extract Pfam IDs from all TSV files ===
tsv_files = glob.glob("eval/pfam/*.tsv")  # Adjust path if needed
all_data = []

for file in tsv_files:
    df = pd.read_csv(file, sep="\t", comment="#", header=None)
    for _, row in df.iterrows():
        sequence_id = row[0]  # assuming first column is the same as 'id' in CSV
        row_str = " ".join(map(str, row))
        pfams = re.findall(r"PF\d{5}", row_str)
        if pfams:
            all_data.append({"id": sequence_id, "pfam_id": pfams})

df_all = pd.DataFrame(all_data)

# Expand to one Pfam per row
df_expanded = df_all.explode("pfam_id")

# Group by ID → join unique Pfams as comma-separated string
df_grouped = (
    df_expanded.groupby("id")["pfam_id"]
    .apply(lambda x: ",".join(sorted(set(x))))
    .reset_index()
)

# === Step 2: Read your protein CSV ===
df_proteins = pd.read_csv("results/"+args.model_name+"_seqs_with_communities.csv")  # has id, original_sequence, community

# === Step 3: Merge Pfam IDs into protein file ===
df_merged = df_proteins.merge(df_grouped, on="id", how="left")

# === Step 4: Save final output ===
df_merged.to_csv("eval/"+args.model_name+"_proteins_with_pfam_final.csv", index=False)

print("Done! Output saved to eval/"+args.model_name+"_proteins_with_pfam.csv")
