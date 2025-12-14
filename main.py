import pandas as pd  # (kept; not used below but harmless if you use it elsewhere)
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

# Loading graph data
def load_graph(path: str, min_edge_weight: int = 1) -> nx.Graph:
    """
    Load social graph.

    File format:
        Each line: u v
        Lines starting with # are comments.

    Returns:
        Undirected weighted NetworkX graph.
    """
    edge_counts = Counter()

    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.strip().split()

            # Canonicalize for undirected counting: (u,v) == (v,u)
            a, b = (u, v) if u <= v else (v, u)
            edge_counts[(a, b)] += 1

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

# Generate random graph for stats
def random_graph_same_size(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """
    Create a random simple graph with the same number of nodes and edges as G.
    Keeps the original node labels.
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    m_edges = G.number_of_edges()

    # gnm_random_graph uses nodes 0..n_nodes-1
    H = nx.gnm_random_graph(n_nodes, m_edges, seed=seed)

    # Relabel back to original node IDs
    mapping = {i: nodes[i] for i in range(n_nodes)}
    H = nx.relabel_nodes(H, mapping)

    # Optional: add weights to match your measure pipeline expectations
    for u, v in H.edges():
        H[u][v]["weight"] = 1.0

    return H

# Perturbation helper functions
def remove_random_edges(G: nx.Graph, n: int, seed: int | None = None):
    #Randomly remove n edges from G.
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")
    if n > G.number_of_edges():
        raise ValueError("n cannot be larger than number of edges in G")

    rng = random.Random(seed)
    G_removed = G.copy()
    edges = list(G_removed.edges())
    removed_edges = rng.sample(edges, n)
    G_removed.remove_edges_from(removed_edges)
    return G_removed, removed_edges


def add_random_edges(G: nx.Graph, n: int, seed: int | None = None):
    #Randomly insert n edges from the set of non-edges.
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")

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
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")

    G_new = G.copy()
    added_edges = []
    for u, v, score in predictions:
        if len(added_edges) >= n:
            break
        if not G_new.has_edge(u, v) and u != v:
            G_new.add_edge(u, v, weight=1.0, predicted_score=float(score))
            added_edges.append((u, v))
    return G_new, added_edges


# GNN Link prediction method
def score_candidates_gnn(G: nx.Graph, candidates, seed: int | None = None):
    """
    Train a small GNN (GraphSAGE) for link prediction and score `candidates`.
    Returns list[(u, v, score)] where score is in [0,1].

    Requirements:
      pip install torch torch-geometric
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
    except ImportError as e:
        raise ImportError(
            "GNN method requires PyTorch + PyTorch Geometric. "
            "Install e.g. torch and torch-geometric for your platform."
        ) from e

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # Build edge_index (undirected: include both directions)
    edges = [(idx[u], idx[v]) for u, v in G.edges() if u in idx and v in idx and u != v]
    if not edges:
        return [(u, v, 0.0) for (u, v) in candidates if u != v]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # add reverse direction

    # Simple node features: degree
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    x = torch.from_numpy(deg).view(-1, 1)

    data = Data(x=x, edge_index=edge_index)

    class LinkPredSAGE(nn.Module):
        def __init__(self, in_dim=1, hid=64, out=64):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hid)
            self.conv2 = SAGEConv(hid, out)

        def encode(self, x, edge_index):
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = self.conv2(h, edge_index)
            return h

        def score(self, z, src, dst):
            return (z[src] * z[dst]).sum(dim=-1)  # dot product

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkPredSAGE().to(device)
    data = data.to(device)

    # Positives for training
    pos = torch.tensor(edges, dtype=torch.long, device=device)  # [E,2]
    E = pos.size(0)

    # Simple uniform negative sampling
    seen = set((a, b) for a, b in edges) | set((b, a) for a, b in edges)
    rng = random.Random(seed)

    neg = []
    while len(neg) < E:
        a = rng.randrange(num_nodes)
        b = rng.randrange(num_nodes)
        if a == b:
            continue
        if (a, b) in seen:
            continue
        neg.append((a, b))
        seen.add((a, b))
        seen.add((b, a))
    neg = torch.tensor(neg, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Train
    model.train()
    for _ in range(50):
        opt.zero_grad()
        z = model.encode(data.x, data.edge_index)

        pos_logits = model.score(z, pos[:, 0], pos[:, 1])
        neg_logits = model.score(z, neg[:, 0], neg[:, 1])

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

    # Score candidate non-edges
    model.eval()
    scored = []
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        for (u, v) in candidates:
            if u == v:
                continue
            if u not in idx or v not in idx:
                continue
            iu, iv = idx[u], idx[v]
            logit = model.score(
                z,
                torch.tensor([iu], device=device),
                torch.tensor([iv], device=device),
            )
            score = torch.sigmoid(logit).item()
            scored.append((u, v, float(score)))

    return scored


# Link prediction scoring
def _score_candidates(G: nx.Graph, method: str, candidates, *, seed: int | None = None):
    """Score candidate non-edges with a given link prediction method."""
    method = method.lower()
    if method == "jaccard":
        gen = jaccard_coefficient(G, candidates)
        return [(u, v, score) for (u, v, score) in gen]
    elif method == "adamic_adar":
        gen = adamic_adar_index(G, candidates)
        return [(u, v, score) for (u, v, score) in gen]
    elif method == "preferential_attachment":
        gen = preferential_attachment(G, candidates)
        return [(u, v, score) for (u, v, score) in gen]
    elif method == "gnn":
        return score_candidates_gnn(G, candidates, seed=seed)
    else:
        raise ValueError(f"Unknown link prediction method: {method}")


def predict_topn_links(
    G: nx.Graph,
    n: int,
    method: str = "jaccard",
    candidate_sample_factor: int = 10,
    seed: int | None = None,
):
    #Sample candidate non-edges, score them, and return top-n (u, v, score).
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")
    if n <= 0:
        return []

    rng = random.Random(seed)
    non_edges = list(nx.non_edges(G))
    if not non_edges:
        return []

    sample_size = min(len(non_edges), candidate_sample_factor * n)
    candidates = rng.sample(non_edges, sample_size)

    scored = _score_candidates(G, method, candidates, seed=seed)
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
    # Remove n random edges.
    # Insert n edges chosen via "method" link prediction on the post-deletion graph.
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


# Measures from Hay et al.'s paper on anonymizing social networks
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
    Returns both values and timing info.
    """
    timings = {}
    t_total = time.time()

    H = _largest_component_subgraph(G)

    # Degree
    t0 = time.time()
    degrees = np.array([d for _, d in H.degree()], dtype=float)
    median_degree = float(np.median(degrees)) if len(degrees) > 0 else float("nan")
    timings["degree"] = time.time() - t0

    # Diameter (approxiomately for time saving purposes)
    t0 = time.time()
    try:
        diameter = float(nx.approximation.diameter(H)) if H.number_of_edges() > 0 else float("nan")
    except Exception:
        diameter = float("nan")
    timings["diameter"] = time.time() - t0

    # Approximate median path length
    def approximate_median_path_length(H, num_sources=500, seed=42):
        rng = random.Random(seed)
        nodes = list(H.nodes())
        if not nodes:
            return float("nan")
        sources = rng.sample(nodes, min(num_sources, len(nodes)))
        lengths = []
        for s in sources:
            sp = nx.single_source_shortest_path_length(H, s)
            lengths.extend(v for v in sp.values() if v > 0)
        return float(np.median(lengths)) if lengths else float("nan")

    t0 = time.time()
    median_path_len = approximate_median_path_length(H)
    timings["median_path_length"] = time.time() - t0

    # Approximate median closeness
    def approximate_median_closeness(H, num_nodes=2000, seed=42):
        rng = random.Random(seed)
        nodes = list(H.nodes())
        if not nodes:
            return float("nan")
        sample = rng.sample(nodes, min(num_nodes, len(nodes)))
        vals = [nx.closeness_centrality(H, u) for u in sample]
        return float(np.median(vals)) if vals else float("nan")

    t0 = time.time()
    median_closeness = approximate_median_closeness(H)
    timings["median_closeness"] = time.time() - t0

    # Approximate betweenness
    t0 = time.time()
    betweenness = nx.betweenness_centrality(
        H, k=min(200, H.number_of_nodes()), normalized=True, seed=42
    )
    median_betweenness = float(np.median(list(betweenness.values()))) if betweenness else float("nan")
    timings["median_betweenness"] = time.time() - t0

    # Clustering
    t0 = time.time()
    avg_clustering = float(nx.average_clustering(H)) if H.number_of_nodes() > 0 else float("nan")
    timings["avg_clustering"] = time.time() - t0

    timings["total"] = time.time() - t_total

    return {
        "num_nodes": H.number_of_nodes(),
        "num_edges": H.number_of_edges(),
        "median_degree": median_degree,
        "diameter": diameter,
        "median_path_length": median_path_len,
        "median_closeness": median_closeness,
        "median_betweenness": median_betweenness,
        "avg_clustering": avg_clustering,
        "timings": timings,
    }

# Print the measures comparison, between original, 5% and random graph
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


# Caching to save on compute time
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_filename(cache_dir, csv_path, perturb_fraction, seed, tag)

    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    print(f"[CACHE MISS] Computing {tag}...")
    measures = compute_fn()

    # atomic-ish write
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(measures, f, indent=2)
    tmp.replace(path)

    print(f"[CACHE WRITE] {path}")
    return measures



# Run experiment, all different perturbation methods, compared to original and random graph,
# with the Hay measures from the original paper
def run_experiment(
    csv_path: str,
    perturb_fraction: float = 0.05,
    link_methods=("jaccard", "adamic_adar", "preferential_attachment", "gnn"),
    seed: int = 42,
):
    cache_dir = Path("results_cache")

    # Load graph (replace with file name)
    G_original = load_graph(csv_path)
    E = G_original.number_of_edges()
    n = int(round(perturb_fraction * E))
    print(f"Original: {G_original.number_of_nodes()} nodes, {E} edges")
    print(f"Perturbation: delete {n}, add {n} edges ({perturb_fraction*100:.1f}% of edges)\n")

    # Original measures
    measures_orig = _load_or_compute_measures(
        cache_dir,
        csv_path,
        perturb_fraction=0.0,  # important: separate cache key
        seed=seed,
        tag="original",
        compute_fn=lambda: compute_hay_measures(G_original),
    )

    # Random 100% perturbation baseline: random graph with same (n, m)
    def _compute_random_full():
        G_full = random_graph_same_size(G_original, seed=seed)
        return compute_hay_measures(G_full)

    measures_rand_full = _load_or_compute_measures(
        cache_dir,
        csv_path,
        perturb_fraction,  # fine to keep; tag makes it distinct anyway
        seed,
        tag="random_full",
        compute_fn=_compute_random_full,
    )

    # Random partial perturbation baseline
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

        def _compute_lp(method=m):
            G_lp, _, _ = perturb_with_link_prediction(
                G_original,
                n=n,
                method=method,
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
