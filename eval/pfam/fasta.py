# Convert seq.csv file into fasta format for InterPro searchh
import pandas as pd

# Load CSV
df = pd.read_csv("data/seqs.csv")

# Write to FASTA
with open("eval/sequences.fasta", "w") as fasta_file:
    for _, row in df.iterrows():
        fasta_file.write(f">{row['id']}\n")
        fasta_file.write(f"{row['original_sequence']}\n")
