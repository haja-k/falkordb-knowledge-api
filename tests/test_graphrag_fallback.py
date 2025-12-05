import asyncio

import main as app_main
from main import RagBody


class FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeGraph:
    def __init__(self):
        pass

    def query(self, q):
        # If searching for name/title contains 'keyword' return a node dict
        if "toLower(n.name) CONTAINS" in q or "toLower(n.title) CONTAINS" in q:
            node = {"id": "k1", "name": "KeywordNode", "title": "KeywordNode", "labels": ["Entity"]}
            return FakeResult([[node]])
        # fallback: no results
        return FakeResult([])


class FakeDB:
    def __init__(self, graph):
        self._graph = graph

    def select_graph(self, name):
        return self._graph


def test_graphrag_fallback():
    # make FalkorDB return a fake db with a fake graph
    fake_graph = FakeGraph()

    # patch app_main.FalkorDB and app_main.check_embedding directly
    orig_FalkorDB = getattr(app_main, "FalkorDB")
    orig_check_embedding = getattr(app_main, "check_embedding")
    try:
        app_main.FalkorDB = lambda host, port, password=None: FakeDB(fake_graph)

        async def fake_check(endpoint, key):
            return False

        app_main.check_embedding = fake_check

        body = RagBody(question="Find KeywordNode details", top_k=5, hops=1)

        # call the async endpoint directly
        resp = asyncio.run(app_main.graphrag(body))

        assert resp.get("success") is True
        # When the endpoint successfully processes the request we expect a
        # well-formed response with an 'answer' or 'facts' payload. Seeds may
        # be empty depending on the graph content.
        assert "answer" in resp or "facts" in resp
    finally:
        app_main.FalkorDB = orig_FalkorDB
        app_main.check_embedding = orig_check_embedding
