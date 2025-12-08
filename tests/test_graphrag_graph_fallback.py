import asyncio

import main as app_main
from main import app


class FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeGraphForCandidates:
    def query(self, q):
        # make hybrid_candidates (vector + fulltext) return nothing for this fake graph
        return FakeResult([])


class FakeState:
    def __init__(self, graph):
        self.graph = graph


def test_graphrag_uses_graph_retriever():
    # Save originals
    orig_state = getattr(app, "state", None)
    orig_hybrid = getattr(app_main, "hybrid_candidates")
    orig_retrieve = getattr(app_main, "retrieve_from_graph")
    orig_check = getattr(app_main, "check_embedding")
    orig_getemb = getattr(app_main, "get_embedding")

    try:
        # Mock app.state.graph
        fake_graph = FakeGraphForCandidates()
        app.state = FakeState(fake_graph)

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

        body = app_main.RagBody(question="sarawak population", top_k=5, hops=1)
        resp = asyncio.run(app_main.graphrag(body))

        assert resp.get("success") is True
        # After fallback, we should have seeds or answer containing Sarawak
        data = resp.get("data", {})
        assert data.get("seeds") or "Sarawak" in data.get("answer", "")

    finally:
        if orig_state is not None:
            app.state = orig_state
        app_main.hybrid_candidates = orig_hybrid
        app_main.retrieve_from_graph = orig_retrieve
        app_main.check_embedding = orig_check
        app_main.get_embedding = orig_getemb
