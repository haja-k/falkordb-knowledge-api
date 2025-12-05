import asyncio

import main as app_main


class FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeGraphForCandidates:
    def query(self, q):
        # make hybrid_candidates (vector + fulltext) return nothing for this fake graph
        # but allow retrieve_from_graph-based queries to be simulated in the next class
        return FakeResult([])


class FakeGraphForRetrieve:
    def query(self, q):
        # If retrieve_from_graph checks name/title/text/content, return a node
        if "toLower(n.name) CONTAINS" in q or "toLower(n.title) CONTAINS" in q:
            node = {"id": "s1", "name": "Sarawak", "labels": ["Region"]}
            return FakeResult([[node]])
        if "MATCH (a {name: 'Sarawak'})" in q:
            return FakeResult([[{"type":"POPULATION_OF"}, "Population"]])
        return FakeResult([])


class FakeDB:
    def __init__(self, graph):
        self._graph = graph

    def select_graph(self, name):
        return self._graph


def test_graphrag_uses_graph_retriever():
    # patch FalkorDB to return a graph for hybrid (empty) and later a graph for retriever
    orig_FalkorDB = getattr(app_main, "FalkorDB")
    orig_hybrid = getattr(app_main, "hybrid_candidates")
    orig_retrieve = getattr(app_main, "retrieve_from_graph")
    orig_check = getattr(app_main, "check_embedding")
    orig_getemb = getattr(app_main, "get_embedding")

    try:
        # hybrid path returns no candidates
        app_main.hybrid_candidates = lambda *a, **k: []

        # ensure embeddings look available (to mimic your symptom)
        async def fake_check(endpoint, key):
            return True

        async def fake_get(endpoint, key, q):
            # return a dummy vector
            return [0.0, 1.0, 0.0]

        app_main.check_embedding = fake_check
        app_main.get_embedding = fake_get

        # Use a retrieve_from_graph implementation that returns the Sarawak node
        def fake_retrieve(graph, q, limit_per_word=5):
            return {"entities": [{"name": "Sarawak", "type": "Region"}], "relationships": [{"relationship":"POPULATION_OF", "target":"Population"}], "type":"graph"}

        app_main.retrieve_from_graph = fake_retrieve

        # patch FalkorDB to give any graph object (unused by the fake retriever)
        app_main.FalkorDB = lambda host, port, password=None: FakeDB(FakeGraphForCandidates())

        body = app_main.RagBody(question="sarawak population", top_k=5, hops=1)
        resp = asyncio.run(app_main.graphrag(body))

        assert resp.get("success") is True
        # After fallback, we should have seeds (converted from retrieve_from_graph) or facts
        assert resp.get("seeds") or "Sarawak" in resp.get("facts", "")

    finally:
        app_main.FalkorDB = orig_FalkorDB
        app_main.hybrid_candidates = orig_hybrid
        app_main.retrieve_from_graph = orig_retrieve
        app_main.check_embedding = orig_check
        app_main.get_embedding = orig_getemb
