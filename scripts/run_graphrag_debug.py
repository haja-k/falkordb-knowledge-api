import asyncio
import main as app_main
from tests.test_graphrag_fallback import FakeGraph, FakeDB

def run():
    # patch FalkorDB and check_embedding
    app_main.FalkorDB = lambda host, port, password=None: FakeDB(FakeGraph())
    app_main.check_embedding = lambda e,k: asyncio.sleep(0, result=False)

    # replicate the graphrag flow, inside script so we can inspect intermediate values
    body = app_main.RagBody(question='Find KeywordNode details', top_k=5, hops=1)

    # call graphrag but inspect hybrid candidates and expanded
    resp = asyncio.run(app_main.graphrag(body))
    print('GRAPHRAG RESPONSE:')
    print(resp)

if __name__ == '__main__':
    run()
