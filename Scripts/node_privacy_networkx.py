# Reference implementations of techniques described in the paper
# Analyzing Graphs with Node Differential Privacy
# Shiva Prasad Kasiviswanathan, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith
# using networkx, scipy.optimize, and relm

import numpy as np
import networkx as nx

import scipy.optimize
import scipy.interpolate
import scipy.special

from common import *
from relm.mechanisms import LaplaceMechanism, CauchyMechanism


# =============================================================================
# Convex Optimization
# =============================================================================
def exact_count(G, h):
    x = np.arange(len(h))
    h_fun = scipy.interpolate.interp1d(x, h)
    degree_histogram = np.array(nx.degree_histogram(G))
    exact_count = degree_histogram @ h_fun(np.arange(len(degree_histogram)))
    return exact_count


def flow_graph(G, D):
    V_left = list(zip(["left"] * len(G), G.nodes()))
    V_right = list(zip(["right"] * len(G), G.nodes()))
    F = nx.DiGraph()
    F.add_nodes_from(V_left)
    F.add_nodes_from(V_right)
    F.add_nodes_from("st")
    F.add_weighted_edges_from([("s", vl, D) for vl in V_left], weight="capacity")
    F.add_weighted_edges_from([(vr, "t", D) for vr in V_right], weight="capacity")
    F.add_weighted_edges_from(
        [(("left", u), ("right", v), 1) for u, v in G.edges()], weight="capacity"
    )
    F.add_weighted_edges_from(
        [(("left", v), ("right", u), 1) for u, v in G.edges()], weight="capacity"
    )
    return F


def bounded_degree_flow(G, h, D):
    F = flow_graph(G, D)
    nodes = list(F.nodes())
    edges = list(F.edges())
    adjacency = np.zeros((len(nodes), len(edges)))
    for j in range(len(edges)):
        i0 = nodes.index(edges[j][0])
        i1 = nodes.index(edges[j][1])
        adjacency[i0, j] = -1
        adjacency[i1, j] = 1

    capacities = np.array([F.edges[e]["capacity"] for e in F.edges()])
    x0 = np.random.random(capacities.size) * capacities
    mask = np.array([("s" in edge) for edge in edges])
    bounds = [(0, capacity) for capacity in capacities]
    constraint = scipy.optimize.LinearConstraint(adjacency[:-2], 0, 0)

    x = np.arange(D + 1)
    h_fun = scipy.interpolate.interp1d(x, h[: D + 1])
    f = lambda x, *args: -np.sum(h_fun(x[tuple(args[0])]))
    res = scipy.optimize.minimize(
        fun=f, x0=x0, args=[mask], bounds=bounds, constraints=[constraint]
    )
    return -res.fun


def run(input_file, run_id, epsilon):
    # =============================================================================
    # Generate a random graph
    G = nx.read_edgelist(input_file)
    n = G.number_of_nodes() + 1

    # Set the degree bound
    D = 2**3

    # -----------------------------------------------------------------------------
    # edge count
    h = np.arange(n + 1) / 2.0

    # Compute exact edge count
    print("Exact edge count = %i" % exact_count(G, h))

    # Compute max flow for graphs of bounded degree
    bd_res = bounded_degree_flow(G, h, D)
    print("Bounded-degree edge count = %f" % bd_res)

    # Create a differentially private release mechanism
    sensitivity = np.max(h[: (D + 1)]) + np.max(h[1 : (D + 1)] - h[:D])
    mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    # Compute the differentially private query response
    dp_edge_count = mechanism.release(np.array([bd_res]))[0]
    print("Differentially private edge count = %f\n" % dp_edge_count)

    # -----------------------------------------------------------------------------
    # node count
    h = np.ones(n + 1)

    # Compute exact node count
    print("Exact node count = %i" % exact_count(G, h))

    # Compute max flow for graphs of bounded degree
    bd_res = bounded_degree_flow(G, h, D)
    print("Bounded-degree node count = %f" % bd_res)

    # Create a differentially private release mechanism
    y = h[: (D + 1)]
    sensitivity = np.max(h[: (D + 1)]) + np.max(h[1 : (D + 1)] - h[:D])
    mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    # Compute the differentially private query response
    dp_node_count = mechanism.release(np.array([bd_res]))[0]
    print("Differentially private node count = %f\n" % dp_node_count)

    # -----------------------------------------------------------------------------
    # k-star count
    k = 2
    h = scipy.special.comb(np.arange(n + 1), k)

    # Compute exact k-star count
    print("Exact k-star count = %i" % exact_count(G, h))

    # Compute max flow for graphs of bounded degree
    bd_res = bounded_degree_flow(G, h, D)
    print("Bounded-degree k-star count = %f" % bd_res)

    # Create a differentially private release mechanism
    y = h[: (D + 1)]
    sensitivity = np.max(h[: (D + 1)]) + np.max(h[1 : (D + 1)] - h[:D])
    mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    # Compute the differentially private query response
    dp_kstar_count = mechanism.release(np.array([bd_res]))[0]
    print("Differentially private k-star count = %f\n" % dp_kstar_count)

    # =============================================================================
    # Linear Programming
    # =============================================================================

    # Compute the exact triangle count
    k = 3
    triangles = nx.triads_by_type(nx.DiGraph(G))["300"]
    print("Exact triangle count = %i" % len(triangles))

    # Compute bounded-degree triangle count using linear programming
    c = -np.ones(len(triangles))
    A_ub = np.zeros((n, len(triangles)))
    for i, t in enumerate(triangles):
        nodes = triangles[i].nodes()
        for node in nodes:
            A_ub[int(node), i] = 1

    # Set the degree bound
    D = 2**2
    sensitivity = k * D * (D - 1) ** (k - 2)
    b_ub = np.ones(n) * sensitivity
    bounds = (0.0, 1.0)
    triangle_count = -0.0
    if len(c) != 0:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        triangle_count = res.fun
    print("Bounded-degree triangle count = %f" % -triangle_count)

    # Create a differentially private release mechanism
    mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    # Compute the differentially private query response
    dp_triangle_count = mechanism.release(np.array([-triangle_count]))[0]
    print("Differentially private triangle count = %f\n" % dp_triangle_count)
    result = {}
    result["id"] = run_id
    result["n_edges"] = dp_edge_count
    result["n_nodes"] = dp_node_count
    result["n_2_star"] = dp_kstar_count
    result["n_triangles"] = dp_triangle_count
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
    try:
        results: list[dict[str, Any]] = [
            run(input_file, run_id, epsilon) for run_id in range(replications)
        ]
        log(f"Running {replications} replications - Done")
        log()

        alert = None
        write_results(output_file, args, alert, results)

    except:
        alert = "error - graph probably too large"
        results = None
        write_results(output_file, args, alert, results)

    log("************** Done! **************")
