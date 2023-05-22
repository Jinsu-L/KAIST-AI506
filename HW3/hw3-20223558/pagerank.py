import copy
from collections import defaultdict


def run_pagerank_one_step(hypergraph, pr_scores, damping_factor):
    # hypergraph H = (V, E) e : hypergraph edge
    # e : {0,1,2}
    # 여기서 임의 노드 u 의 degree는? u를 포함하는 hyperedges의 수.
    # u를 선택 했을때, 해당 edge가 선택될 확률이라고 보면됨. 2개의 hypergraph에 걸쳐 있으면 반토막 나서
    # propagation 되는거지.

    # u -> v로 가는 확률은 hyperedge내에서 움직이는 거니 |e|가 되어서
    # r_u를 hyperedge가 겹치는 수로 나누고, 이를 다시 확률로 나눠서, u->v에 대한 score를 만드는 것.
    # v에 걸친 hyperedge가 여러개라면, 그것도 다 더해서 r_v를 계산하는 거지.
    # d_u = defaultdict(int) # u : d(u)

    result = [0 for _ in pr_scores]

    e_v = defaultdict(list)  # len(E_v[u]) == d(u)
    for i, hyper_edge in enumerate(hypergraph['edges']):  # (0, 1, 2)
        for e in set(hyper_edge):
            e_v[e].append(i)

    for v in range(hypergraph['num_nodes']):
        for e_i in e_v[v]:  # e in E(v) # 해당 노드가 포함된 hypergraph
            e = hypergraph['edges'][e_i]  # 해당 노드가 포함된 hypergraph에 이웃 노드
            # e = e_v[e_i]
            for u in e:  # u in e
                result[v] += damping_factor * (1. / len(e)) * (
                        pr_scores[u] / len(e_v[u]))  # + (1 - damping_factor) * (1 / len(result))

        result[v] += (1 - damping_factor) * (1. / hypergraph['num_nodes'])

    return result


def run_advanced_pagerank_one_step(hypergraph, pr_scores, damping_factor):
    # construct base
    e_v = defaultdict(list)
    delta = dict()
    gamma_e_v = defaultdict(dict)

    for edge_idx, (hyper_edge, hyper_edge_weight) in enumerate(zip(hypergraph['edges'], hypergraph['edge_weights'])):
        delta_e = 0
        for (node_idx, node_weight) in hyper_edge:
            gamma_e_v[edge_idx][node_idx] = node_weight
            e_v[node_idx].append((edge_idx, hyper_edge_weight))  # 각 노드별로 어떤 hyper edge에 포함되는지.
            delta_e += node_weight
        delta[edge_idx] = delta_e

    result = [0 for _ in pr_scores]
    for v in range(hypergraph['num_nodes']):
        for (e_i, e_w) in e_v[v]:
            e = hypergraph['edges'][e_i]
            for (u, u_w) in e:  # u in e
                d_u = sum([w for _, w in e_v[u]])
                result[v] += damping_factor * (gamma_e_v[e_i][v] / delta[e_i]) * ((e_w * pr_scores[u]) / d_u)

        result[v] += (1 - damping_factor) * (1. / hypergraph['num_nodes'])

    return result


def run_pagerank(hypergraph, initial_scores, damping_factor, use_weight=False):
    prev_pr_scores = copy.copy(initial_scores, )

    if use_weight:
        page_rank_algorithm = run_advanced_pagerank_one_step
    else:
        page_rank_algorithm = run_pagerank_one_step

    while True:
        new_pr_scores = page_rank_algorithm(hypergraph, prev_pr_scores, damping_factor)

        # check
        diff = sum([abs(p_s - n_s) for p_s, n_s in zip(prev_pr_scores, new_pr_scores)])
        if diff < 1e-6:
            break

        prev_pr_scores = new_pr_scores

    return new_pr_scores
