# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Visualisation
----------------------------
- similarity_matrix() : calcule la matrice N x N de similarité entre
  tous les documents, pour une métrique donnée.
- plot_similarity_heatmap() : heatmap matplotlib de cette matrice.
- plot_dendrogram() : regroupe les documents par similarité (clustering
  hiérarchique) pour visualiser les "grappes" de documents proches
  (utile pour repérer visuellement des groupes de plagiat).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif nécessaire (sauvegarde en fichier)
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from data import DOCUMENTS
from similarity_metrics import METRICS


def similarity_matrix(documents: dict, metric_name: str) -> tuple[np.ndarray, list[str]]:
    metric_fn = METRICS[metric_name]
    names = list(documents.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1.0 if i == j else metric_fn(documents[names[i]], documents[names[j]])
    return matrix, names


def plot_similarity_heatmap(matrix: np.ndarray, names: list[str], title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Similarité")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_dendrogram(matrix: np.ndarray, names: list[str], output_path: str):
    """Clustering hiérarchique à partir de la matrice de distance
    (1 - similarité)."""
    distance_matrix = 1 - matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=names, ax=ax, leaf_rotation=45)
    ax.set_title("Regroupement des documents par similarité")
    ax.set_ylabel("Distance (1 - similarité)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    matrix, names = similarity_matrix(DOCUMENTS, "Jaccard (n-grammes)")
    plot_similarity_heatmap(matrix, names, "Similarité Jaccard entre documents", "heatmap_demo.png")
    plot_dendrogram(matrix, names, "dendrogram_demo.png")
    print("Visualisations générées : heatmap_demo.png, dendrogram_demo.png")
