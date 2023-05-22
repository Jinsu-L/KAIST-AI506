def load_hypergraph(filename, use_weight=False):
    with open(filename, 'r') as f:
        edge_weights = []
        for line in f:
            eid, w = line.strip().split()
            if float(w) < 0: break
            edge_weights.append((int(eid), float(w)))
            
        hg_infos = [(lambda eid, nid, w: (int(eid), int(nid), float(w)))(*line.strip().split()) for line in f]
        edge_ids, node_ids, weights = zip(*sorted(hg_infos))
        edge_weights = [v for _, v in sorted(edge_weights)]
        
    prv_eid = None
    edges = [[]]
    for eid, nid, w in zip(edge_ids, node_ids, weights):
        if prv_eid != eid and len(edges[-1]) > 0:
            edges.append([])
        prv_eid = eid
        edges[-1].append((nid, w) if use_weight else nid)
    edges = list(map(tuple, edges))
    
    output = {'num_nodes': max(node_ids) + 1,
             'num_edges': len(edges),
             'edges': edges}
    
    if use_weight:
        output['edge_weights'] = edge_weights
        
    return output