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


