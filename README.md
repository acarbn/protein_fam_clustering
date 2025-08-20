# Protein Sequence Clustering and Family Analysis (protein_fam_clustering)
This repository provides tools to cluster protein sequences into families using ESM embeddings, visualize the networks, and compare computed communities with known PFAM families.

**Author:** Burcin Acar  

---
## **Workflow Overview**

1. Load protein sequences from a CSV file.
2. Compute ESM embeddings using a locally downloaded model.
3. Compute pairwise distances between sequences based on embedding similarities.
4. Dynamically select a clustering threshold distance.
5. Construct a graph network and detect communities using Louvain clustering.
6. Save the output CSV with corresponding community assignments.
7. Compare with PFAM families and save the final annotated CSV.
8. Interactively inspect graph networks 

## **Setup**

1. **Clone the repository:**
```bash
git clone https://github.com/acarbn/protein_pfam_clustering.git
cd protein_fam_clustering
```

2. **Create and activate a virtual environment (optional but recommended):**
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```
3. **Install dependencies from requirements.txt:**
```bash
pip install -r requirements.txt
```
## **Usage**

**Step 1: Download ESM Model**

Downloads the tokenizer and the model weights.
```bash
python3 -m src.download_esm_model esm2_t33_650M_UR50D
```
**Step 2: Cluster Protein Sequences**

Outputs embeddings, distance matrix, communities, and network visualization in results/.
```bash
python3 -m src.main data/seqs.csv --model_name esm2_t33_650M_local
```
**Step 3: Compare Communities with PFAM**

Produces eval/<model_name>_proteins_with_pfam.csv with PFAM annotations.
```bash
python3 -m src.compare_communities_pfam esm2_t33_650M_local
```
## **Folder Structure**
```text
.
├── data/                  # Input CSVs and later generated embeddings
├── model/                 # Local ESM models
├── results/               # Clustering outputs and plots
├── eval/                  # PFAM annotations and comparison outputs
├── main.py                # Main clustering script
├── helper.py              # Helper functions
├── download_esm_model.py  # ESM model downloader
├── compare_communities_pfam.py  # PFAM comparison script
├── requirements.txt       # Python dependencies
└── README.md
