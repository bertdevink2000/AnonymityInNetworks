import pandas as pd
import networkx as nx
import numpy as np
import random
from collections import Counter
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    preferential_attachment,
)
import gzip
from pathlib import Path
import json
import time

def load_enron_email_graph(path: str,min_edge_weight: int = 1,) -> nx.Graph:
    """
    Load the SNAP Enron email graph (email-Enron.txt.gz).

    File format:
        Each line: u v
        Lines starting with # are comments.

    Returns:
        Undirected weighted NetworkX graph.
    """
    edge_counts = Counter()

    # Read compressed SNAP file
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.strip().split()
            edge_counts[(u, v)] += 1

    G = nx.Graph()

    # Build weighted undirected graph
    for (u, v), w in edge_counts.items():
        if w < min_edge_weight:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G

def remove_random_edges(G: nx.Graph, n: int, seed: int | None = None):
    """Randomly remove n edges from G."""
    if n > G.number_of_edges():
        raise ValueError("n cannot be larger than number of edges in G")

    rng = random.Random(seed)
    G_removed = G.copy()
    edges = list(G_removed.edges())
    removed_edges = rng.sample(edges, n)
    G_removed.remove_edges_from(removed_edges)
    return G_removed, removed_edges


def add_random_edges(G: nx.Graph, n: int, seed: int | None = None):
    """Randomly insert n edges from the set of non-edges."""
    G_new = G.copy()
    rng = random.Random(seed)

    non_edges = list(nx.non_edges(G_new))
    if n > len(non_edges):
        raise ValueError("Not enough non-edges to insert.")

    added = rng.sample(non_edges, n)
    G_new.add_edges_from(added)
    return G_new, added


def add_predicted_edges(G: nx.Graph, predictions: list[tuple], n: int):
    """Add up to n predicted edges (u, v, score) to G (if not already present)."""
    G_new = G.copy()
    added_edges = []
    for u, v, score in predictions:
        if len(added_edges) >= n:
            break
        if not G_new.has_edge(u, v) and u != v:
            G_new.add_edge(u, v, weight=1.0, predicted_score=score)
            added_edges.append((u, v))
    return G_new, added_edges


def _score_candidates(G: nx.Graph, method: str, candidates):
    """Score candidate non-edges with a given link prediction method."""
    method = method.lower()
    if method == "jaccard":
        gen = jaccard_coefficient(G, candidates)
    elif method == "adamic_adar":
        gen = adamic_adar_index(G, candidates)
    elif method == "preferential_attachment":
        gen = preferential_attachment(G, candidates)
    else:
        raise ValueError(f"Unknown link prediction method: {method}")
    return [(u, v, score) for (u, v, score) in gen]


def predict_topn_links(
    G: nx.Graph,
    n: int,
    method: str = "jaccard",
    candidate_sample_factor: int = 10,
    seed: int | None = None,
):
    """Sample candidate non-edges, score them, and return top-n (u, v, score)."""
    rng = random.Random(seed)
    non_edges = list(nx.non_edges(G))
    if not non_edges:
        return []

    sample_size = min(len(non_edges), candidate_sample_factor * n)
    candidates = rng.sample(non_edges, sample_size)

    scored = _score_candidates(G, method, candidates)
    scored = [t for t in scored if t[2] is not None]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:n]


def perturb_with_link_prediction(
    G: nx.Graph,
    n: int,
    method: str,
    candidate_sample_factor: int = 10,
    seed: int | None = None,
):
    """
    1) Remove n random edges.
    2) Insert n edges chosen via `method` link prediction on the post-deletion graph.
    """
    G_removed, removed = remove_random_edges(G, n, seed=seed)
    preds = predict_topn_links(
        G_removed,
        n=n,
        method=method,
        candidate_sample_factor=candidate_sample_factor,
        seed=seed,
    )
    G_new, added = add_predicted_edges(G_removed, preds, n)
    return G_new, removed, added


def _largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as an undirected graph."""
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_nodes() == 0:
        return H
    comp = max(nx.connected_components(H), key=len)
    return H.subgraph(comp).copy()


def compute_hay_measures(G: nx.Graph) -> dict:
    """
    Compute the graph measures from 'Anonymizing Social Networks'
    on the largest connected component.
    """
    t_total = time.time()
    H = _largest_component_subgraph(G)

    # Degree sequence & median degree
    t0 = time.time()
    degrees = np.array([d for _, d in H.degree()], dtype=float)
    median_degree = float(np.median(degrees)) if len(degrees) > 0 else float("nan")
    t1 = time.time()
    print(f"[Hay] Degree stats: {t1 - t0:.2f}s")

    # Diameter (exact on GCC)
    t0 = time.time()
    diameter = float(nx.diameter(H)) if H.number_of_edges() > 0 else float("nan")
    t1 = time.time()
    print(f"[Hay] Diameter: {t1 - t0:.2f}s")

    # Approximate median path length
    def approximate_median_path_length(H, num_sources=500, seed=42):
        rng = random.Random(seed)
        nodes = list(H.nodes())
        if not nodes:
            return float("nan")
        sources = rng.sample(nodes, min(num_sources, len(nodes)))
        lengths = []

        for s in sources:
            sp = nx.single_source_shortest_path_length(H, s, cutoff=None)
            lengths.extend(v for v in sp.values() if v > 0)

        return float(np.median(lengths)) if lengths else float("nan")

    t0 = time.time()
    median_path_len = approximate_median_path_length(H)
    t1 = time.time()
    print(f"[Hay] Approx. median path length: {t1 - t0:.2f}s")

    # Closeness (median)
    t0 = time.time()
    closeness = nx.closeness_centrality(H)
    median_closeness = float(np.median(list(closeness.values()))) if closeness else float("nan")
    t1 = time.time()
    print(f"[Hay] Closeness centrality: {t1 - t0:.2f}s")

    # Betweenness (median) – approximate via k=500 sampled nodes
    t0 = time.time()
    betweenness = nx.betweenness_centrality(H, k=min(500, H.number_of_nodes()), normalized=True, seed=42)
    median_betweenness = float(np.median(list(betweenness.values()))) if betweenness else float("nan")
    t1 = time.time()
    print(f"[Hay] Approx. betweenness (k=500): {t1 - t0:.2f}s")

    # Average clustering coefficient
    t0 = time.time()
    avg_clustering = float(nx.average_clustering(H)) if H.number_of_nodes() > 0 else float("nan")
    t1 = time.time()
    print(f"[Hay] Avg clustering: {t1 - t0:.2f}s")

    print(f"[Hay] TOTAL: {time.time() - t_total:.2f}s\n")

    return {
        "num_nodes": H.number_of_nodes(),
        "num_edges": H.number_of_edges(),
        "median_degree": median_degree,
        "diameter": diameter,
        "median_path_length": median_path_len,
        "median_closeness": median_closeness,
        "median_betweenness": median_betweenness,
        "avg_clustering": avg_clustering,
    }



def print_measure_comparison(
    measures_original: dict,
    measures_perturbed: dict,
    label_perturbed: str,
    measures_random: dict | None = None,
    label_random: str = "Random (100%)",
):
    keys = [
        "median_degree",
        "diameter",
        "median_path_length",
        "median_closeness",
        "median_betweenness",
        "avg_clustering",
    ]

    header = ["Measure", "Original", label_perturbed]
    if measures_random is not None:
        header.append(label_random)

    print("\t".join(header))
    for k in keys:
        row = [
            k,
            f"{measures_original[k]:.3f}",
            f"{measures_perturbed[k]:.3f}",
        ]
        if measures_random is not None:
            row.append(f"{measures_random[k]:.3f}")
        print("\t".join(row))


def _cache_filename(
    cache_dir: Path,
    csv_path: str,
    perturb_fraction: float,
    seed: int,
    tag: str,
) -> Path:
    base = Path(csv_path).stem
    pf = int(round(perturb_fraction * 1000))  # e.g. 0.05 -> 50
    fname = f"{base}_pf{pf}_seed{seed}_{tag}.json"
    return cache_dir / fname


def _load_or_compute_measures(
    cache_dir: Path,
    csv_path: str,
    perturb_fraction: float,
    seed: int,
    tag: str,
    compute_fn,
) -> dict:
    """
    If cached JSON exists, load and return it.
    Otherwise, call compute_fn(), cache result, and return it.
    """
    cache_dir.mkdir(exist_ok=True)
    path = _cache_filename(cache_dir, csv_path, perturb_fraction, seed, tag)

    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    measures = compute_fn()
    with open(path, "w") as f:
        json.dump(measures, f)
    return measures


def run_experiment(
    csv_path: str,
    perturb_fraction: float = 0.05,
    # keep only methods actually implemented in _score_candidates:
    link_methods=("jaccard", "adamic_adar", "preferential_attachment"),
    seed: int = 42,
):
    cache_dir = Path("results_cache")

    # Load original Enron graph
    G_original = load_enron_email_graph(csv_path)
    E = G_original.number_of_edges()
    n = int(round(perturb_fraction * E))
    print(f"Original: {G_original.number_of_nodes()} nodes, {E} edges")
    print(f"Perturbation: delete {n}, add {n} edges ({perturb_fraction*100:.1f}% of edges)\n")

    # Original measures (no caching needed; fast-ish compared to perturbations)
    measures_orig = compute_hay_measures(G_original)

    # Random 100% perturbation baseline (tag: "random_full")
    measures_rand_full = _load_or_compute_measures(
        cache_dir,
        csv_path,
        perturb_fraction,  # harmless even though full random ignores it
        seed,
        tag="random_full",
        compute_fn=lambda: compute_hay_measures(
            add_random_edges(
                *remove_random_edges(G_original, E, seed=seed),
                seed=seed
            )[0]
        ),
    )

    # Random partial perturbation baseline (tag: "random_partial")
    def _compute_random_partial():
        G_random, _ = remove_random_edges(G_original, n, seed=seed)
        G_random, _ = add_random_edges(G_random, n, seed=seed)
        return compute_hay_measures(G_random)

    measures_random = _load_or_compute_measures(
        cache_dir,
        csv_path,
        perturb_fraction,
        seed,
        tag="random_partial",
        compute_fn=_compute_random_partial,
    )

    print("=== Random perturbation (uniform deletions + insertions) ===")
    print_measure_comparison(
        measures_orig,
        measures_random,
        label_perturbed=f"Random {perturb_fraction*100:.0f}%",
        measures_random=measures_rand_full,
        label_random="Random 100%",
    )
    print()

    # Model-based perturbation using different link prediction methods
    for m in link_methods:
        tag = f"lp_{m}"
        print(f"=== Link-prediction perturbation: {m} ===")

        def _compute_lp():
            G_lp, _, _ = perturb_with_link_prediction(
                G_original,
                n=n,
                method=m,
                candidate_sample_factor=10,
                seed=seed,
            )
            return compute_hay_measures(G_lp)

        measures_lp = _load_or_compute_measures(
            cache_dir,
            csv_path,
            perturb_fraction,
            seed,
            tag=tag,
            compute_fn=_compute_lp,
        )

        print_measure_comparison(
            measures_orig,
            measures_lp,
            label_perturbed=f"{m} {perturb_fraction*100:.0f}%",
            measures_random=measures_rand_full,
            label_random="Random 100%",
        )
        print()




if __name__ == "__main__":
    run_experiment("email-Enron.txt.gz", perturb_fraction=0.05)
