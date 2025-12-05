import math

from functions import (
    anchor_terms,
    extract_keywords,
    cosine,
    format_graph_facts,
    hybrid_candidates,
    expand_neighbors_by_name,
    check_embedding,
)


def test_anchor_terms_quotes_and_titlecase():
    q = 'Find information about "Route Location" and Port Location and small'
    anchors = anchor_terms(q, max_terms=3)
    assert "Route Location" in anchors
    assert any(a.endswith("Location") for a in anchors)


def test_extract_keywords_basic():
    q = "What stakeholders and policies affect Route Location planning in 2025?"
    kws = extract_keywords(q, max_terms=6)
    # Expect the function to return meaningful words and drop stop words
    assert "Route" in kws or "Location" in kws or "planning" in kws
    assert all(len(k) > 2 for k in kws)


def test_cosine_edge_and_identity():
    assert math.isclose(cosine([1, 0], [1, 0]), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0, rel_tol=1e-6)
    assert cosine([], []) == 0.0


def test_format_graph_facts_simple():
    nodes = [
        {"id": "n1", "name": "Alice", "labels": ["Person"]},
        {"id": "n2", "name": "Project X", "labels": ["Project"]},
    ]
    rels = [
        {"type": "LEADS", "props": {"source_text": "Alice leads Project X"}, "start": "n1", "end": "n2"}
    ]
    out = format_graph_facts(nodes, rels, include_source=False)
    assert "Alice" in out and "Project X" in out
    assert "LEADS" in out


class FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeGraph:
    def query(self, q):
        # Return some deterministic results depending on the cypher shape
        if 'exists(n.embedding)' in q:
            # return two nodes with embedding vectors
            return FakeResult([[{"id": "v1", "name": "VecA", "embedding": [1, 0, 0]}], [{"id": "v2", "name": "VecB", "embedding": [1, 0, 0]}]])
        if "toLower(n.name) CONTAINS" in q or "toLower(n.title) CONTAINS" in q:
            # single node match for keywords
            return FakeResult([[{"id": "k1", "name": "KeywordNode", "title": "KeywordNode"}]])
        # neighbor query
        if 'MATCH (a' in q and '-[r]-' in q:
            a = {"id":"nA","name":"NodeA","labels":["Entity"]}
            b = {"id":"nB","name":"NodeB","labels":["Entity"]}
            r = {"type":"RELATED_TO","props":{"source_text":"A->B"}}
            return FakeResult([[a, r, b]])
        return FakeResult([])


def test_hybrid_candidates_and_expand():
    g = FakeGraph()
    # call hybrid_candidates with qvec so vector path is used
    qvec = [1, 0, 0]
    cands = hybrid_candidates(g, "KeywordNode", qvec=qvec, labels=None, k_vec=5, k_kw=5)
    # we expect at least one candidate (vector or keyword)
    assert isinstance(cands, list)

    # test neighbors expansion
    expanded = expand_neighbors_by_name(g, ["NodeA"], hops=1)
    assert expanded["nodes"] and expanded["rels"]


def test_check_embedding_success(monkeypatch):
    class FakeResp:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    class FakeClient:
        def __init__(self, timeout=10.0):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def post(self, endpoint, json=None, headers=None):
            return FakeResp()

    monkeypatch.setattr('functions.graphrag_utils.httpx.AsyncClient', lambda timeout=10.0: FakeClient())

    import asyncio
    ok = asyncio.run(check_embedding('http://fake', 'key'))
    assert ok is True


def test_check_embedding_failure(monkeypatch):
    class BadClient:
        def __init__(self, timeout=10.0):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def post(self, endpoint, json=None, headers=None):
            raise RuntimeError('connection failed')

    monkeypatch.setattr('functions.graphrag_utils.httpx.AsyncClient', lambda timeout=10.0: BadClient())
    import asyncio
    ok = asyncio.run(check_embedding('http://bad', 'key'))
    assert ok is False
