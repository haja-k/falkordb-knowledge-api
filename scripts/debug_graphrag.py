from functions.graphrag_utils import hybrid_candidates, expand_neighbors_by_name, format_graph_facts
from tests.test_graphrag_fallback import FakeGraph

q0 = 'Find KeywordNode details'
g = FakeGraph()
print('Running hybrid_candidates...')
res = hybrid_candidates(g, q0, qvec=None, labels=None, k_vec=max(12, 5), k_kw=max(12, 5), alpha_vec=0.6, beta_kw=0.4)
print('hybrid_candidates ->', res)
seed_names = []
for n, sc in res:
    nm = n.get('name') or n.get('title') or None
    if nm and nm not in seed_names:
        seed_names.append(nm)
print('seed_names->', seed_names)
expanded = expand_neighbors_by_name(g, seed_names[:5], hops=1)
print('expanded ->', expanded)
facts = format_graph_facts(expanded.get('nodes', []), expanded.get('rels', []), include_source=True)
print('facts ->', facts)
