import asyncio

import main as app_main
from main import RagBody, app


class FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeGraph:
    def __init__(self):
        pass

    def query(self, q):
        # If searching for name/title contains 'keyword' return a node dict
        if "toLower(toString(n.name)) CONTAINS" in q or "toLower(toString(n.title)) CONTAINS" in q:
            node = {"id": "k1", "name": "KeywordNode", "title": "KeywordNode", "labels": ["Entity"]}
            return FakeResult([[node]])
        # fallback: no results
        return FakeResult([])


class FakeState:
    def __init__(self, graph):
        self.graph = graph


def test_graphrag_fallback():
    # Create fake graph and state
    fake_graph = FakeGraph()
    fake_state = FakeState(fake_graph)

    # Save original state and check_embedding
    orig_state = getattr(app, "state", None)
    orig_check_embedding = getattr(app_main, "check_embedding")
    
    try:
        # Mock app.state.graph
        app.state = fake_state

        async def fake_check(endpoint, key):
            return False

        app_main.check_embedding = fake_check

        body = RagBody(question="Find KeywordNode details", top_k=5, hops=1)

        # call the async endpoint directly
        resp = asyncio.run(app_main.graphrag(body))

        assert resp.get("success") is True
        # When the endpoint successfully processes the request we expect a
        # well-formed response with an 'answer' payload.
        assert "data" in resp and "answer" in resp.get("data", {})
    finally:
        if orig_state is not None:
            app.state = orig_state
        app_main.check_embedding = orig_check_embedding
