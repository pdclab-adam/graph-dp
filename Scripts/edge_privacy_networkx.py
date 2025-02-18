# Reference implementations of techniques described in the paper
# Smooth Sensitivity and Sampling in Private Data Analysis
# by Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith
# using networkx and relm

from typing import Any
from common import *
import numpy as np
import networkx as nx
from relm.mechanisms import CauchyMechanism


# =============================================================================
# Triangle Counts
# =============================================================================
def ls_util(s, a, b, n):
    """Utility function used in computation of local sensitivities."""
    return np.minimum(a + np.floor((s + np.minimum(s, b)) / 2.0), n - 2)


def find_survivors(A, B, n):
    """Find surviving pairs A[i,j], B[i,j].

    For each distinct value set(A), keep only the pair
    (A[i,j], B[i,j]) with the largest value of B[i,j].
    """
    W = np.array(A).flatten()
    X = np.array(B).flatten()
    idx = np.argsort(W)
    Y = W[idx]
    Z = X[idx]
    delta = np.concatenate((np.zeros(1), Y[1:] != Y[:-1]))
    new_val_idxs = np.concatenate(
        (np.array([0]), np.where(delta)[0], np.array([len(Y)]))
    )
    M = np.empty((len(new_val_idxs) - 1), dtype=int)
    N = np.empty((len(new_val_idxs) - 1), dtype=int)
    for i in range(len(new_val_idxs) - 1):
        M[i] = Y[new_val_idxs[i]]
        N[i] = np.max(Z[new_val_idxs[i] : new_val_idxs[i + 1]])
    return M[::-1], N[::-1]


def first_hit(x, break_list):
    """Find the key for the first pair (k,v) with v > x."""
    return next((k for k, v in break_list if x <= v))


def max_ls_util(s, n, break_list):
    """Compute the maximal value of f(s, a, b, n) over all possible a, b."""
    a, b = first_hit(s, break_list)
    return ls_util(s, a, b, n)


def local_sensitivity_dist(A, B, n):
    """Compute the local sensitivity at distance s for 0 <= s <= n."""
    M, N = find_survivors(A, B, n)
    prev_survivor = (M[0], N[0])
    break_points = {prev_survivor: n + 1}
    for survivor in zip(M[1:], N[1:]):
        a0, b0 = prev_survivor
        a1, b1 = survivor
        intersection = 2 * (a0 - a1) + b0
        if b0 <= intersection <= b1:
            break_points[prev_survivor] = intersection
            break_points[survivor] = n + 1
            prev_survivor = survivor
    break_list = sorted(break_points.items(), key=lambda _: _[1])
    max_utils = [max_ls_util(s, n, break_list) for s in range(n + 1)]
    return np.array(max_utils)


def count_triangles(graph: nx.Graph, epsilon: float) -> float:
    # =============================================================================
    # Count nodes
    n = len(graph.nodes)

    # Compute the adjacency matrix
    M = nx.linalg.graphmatrix.adjacency_matrix(graph).astype(int)

    # -----------------------------------------------------------------------------
    # Compute the partial count matrices
    A = (M @ M).todense()
    B = M.sum(0) + M.sum(1) - 2 * A

    # Zero out the main diagonal because we are
    # interested only in indices (i,j) with i != j
    np.fill_diagonal(A, 0)
    np.fill_diagonal(B, 0)

    # Compute the local sensitivity at distance s for 0 <= s <= n
    lsd = local_sensitivity_dist(A, B, n)

    # Compute the smooth sensitivity
    smooth_scaling = np.exp(-epsilon * np.arange(n + 1))
    smooth_sensitivity = np.max(lsd * smooth_scaling)

    # -----------------------------------------------------------------------------
    # Compute the exact triangle count
    triangle_count = np.array([sum(nx.triangles(graph).values()) / 3.0])

    # Create a differentially private release mechanism
    beta = epsilon / 6.0
    mechanism = CauchyMechanism(epsilon=epsilon, beta=beta)

    # Compute the differentially private query response
    dp_triangle_count = mechanism.release(triangle_count, smooth_sensitivity)

    print("Exact triangle count = %i" % int(triangle_count[0]))
    print("Differentially private triangle count = %f" % dp_triangle_count[0])
    return dp_triangle_count[0]


# =============================================================================
# Minimum Spanning Tree Cost
# =============================================================================
def lightweight_graph(g, w):
    gw = nx.Graph()
    gw.add_nodes_from(g)
    gw.add_edges_from([e for e in g.edges() if g.edges[e]["weight"] <= w])
    return gw


def min_cut(g, w, **args):
    return nx.edge_connectivity(lightweight_graph(g, w), **args)


def retrieve(costs, key, g, w, **args):
    if key not in costs:
        costs[key] = min_cut(g, w[key], **args)
    return costs[key]


def first_hit_weight(k, w, g, bound, costs, **args):
    cost = retrieve(costs, len(w) - 1, g, w, **args)
    if cost <= k:
        return bound

    cost = retrieve(costs, 0, g, w, **args)
    if cost > k:
        return w[0]

    high = len(w) - 1
    low = 0
    while (high - low) > 1:
        mid = (low + high) // 2
        cost = retrieve(costs, mid, g, w, **args)
        if cost <= k:
            low = mid
        else:
            high = mid
    return w[high]


def minimum_spanning_tree_costs(graph: nx.Graph, epsilon: float):
    # =============================================================================
    # Generate a random graph
    n = len(graph.nodes)

    bound = 10.0  # An upper bound on the edge weights in the graph
    weights = {e: bound * np.random.randint(1, 11) / 10.0 for e in graph.edges()}
    nx.set_edge_attributes(graph, weights, "weight")

    edge_weights = [graph.edges[e]["weight"] for e in graph.edges()]
    edge_weights = sorted(set(edge_weights))

    # -----------------------------------------------------------------------------
    # Compute the local sensitivity at distance s for 0 <= s <= n
    costs = dict()
    lsd1 = np.array(
        [first_hit_weight(k, edge_weights, graph, bound, costs) for k in range(n + 1)]
    )

    mst = nx.minimum_spanning_tree(graph)
    lsd2 = np.zeros(n + 1)
    for e in mst.edges():
        costs = dict()
        first_hit_weights = [
            first_hit_weight(k + 1, edge_weights, graph, bound, costs, s=e[0], t=e[1])
            for k in range(n + 1)
        ]
        lsd2_e = np.array(first_hit_weights) - mst.edges[e]["weight"]
        lsd2 = np.maximum(lsd2, lsd2_e)

    lsd = np.maximum(lsd1, lsd2)

    # Compute the smooth sensitivity
    beta = epsilon / 6.0
    smooth_scaling = np.exp(-beta * np.arange(n + 1))
    smooth_sensitivity = np.max(lsd * smooth_scaling)

    # -----------------------------------------------------------------------------
    # Compute the exact MST cost
    mst = nx.minimum_spanning_tree(graph)
    mst_cost = np.array([mst.size(weight="weight")])

    # Create a differentiall private release mechanism
    mechanism = CauchyMechanism(epsilon=epsilon, beta=beta)

    # Compute the differentially private query response
    dp_mst_cost = mechanism.release(mst_cost, smooth_sensitivity)

    print("Exact MST cost = %f" % mst_cost[0])
    print("Differentially private MST cost = %f" % dp_mst_cost[0])
    return dp_mst_cost[0]


def run(input_file: str, run_id: int, epsilon: float) -> dict:
    graph = nx.read_edgelist(input_file)
    n_triangles = count_triangles(graph, epsilon)
    result = {}
    result["id"] = run_id
    result["n_triangles"] = n_triangles
    return result


if __name__ == "__main__":

    args = parse_args()
    input_file: str = args.input_file
    output_file: str = args.output_file
    epsilon: float = args.epsilon
    delta: float = args.delta
    replications: int = args.replications
    disable: bool = args.disable

    if disable:
        log("Disabled by user input")
        log()
        alert = "disabled"
        write_results(output_file, args, alert)
        status = 0
        log(f"********** Exit Status {status} **********")
        exit({status})

    log(f"Running {replications} replications")
    results: list[dict[str, Any]] = [
        run(input_file, run_id, epsilon) for run_id in range(replications)
    ]
    log(f"Running {replications} replications - Done")
    log()

    alert = None
    write_results(output_file, args, alert, results)

    log("************** Done! **************")
