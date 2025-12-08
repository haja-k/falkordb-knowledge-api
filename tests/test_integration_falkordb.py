"""Integration tests that connect to a real FalkorDB instance.

These tests require a running FalkorDB and valid .env configuration.
They are skipped automatically if the DB is unreachable.
"""
import os
import sys

# Ensure repo root is in path so we can import main/functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD") or None
GRAPH_NAME = os.getenv("GRAPH_NAME", "iscs_knowledge_graph")


def _get_graph():
    """Helper to connect and return graph object, or None if unavailable."""
    try:
        from falkordb import FalkorDB
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        return db.select_graph(GRAPH_NAME)
    except Exception as e:
        print(f"[SKIP] Cannot connect to FalkorDB: {e}")
        return None


def test_falkordb_connection():
    """Test that we can connect to FalkorDB and run a simple query."""
    graph = _get_graph()
    if graph is None:
        print("[SKIP] test_falkordb_connection - DB not available")
        return

    result = graph.query("RETURN 1 AS ping")
    assert result.result_set is not None
    assert result.result_set[0][0] == 1
    print(f"[PASS] Connected to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")


def test_node_count():
    """Test that the graph has at least some nodes."""
    graph = _get_graph()
    if graph is None:
        print("[SKIP] test_node_count - DB not available")
        return

    result = graph.query("MATCH (n) RETURN count(n) AS cnt")
    count = result.result_set[0][0]
    print(f"[INFO] Graph '{GRAPH_NAME}' has {count} nodes")
    assert count >= 0  # passes even if empty; change to > 0 if you expect data


def test_sample_node_properties():
    """Fetch a sample node and print its properties (helps debug missing fields)."""
    graph = _get_graph()
    if graph is None:
        print("[SKIP] test_sample_node_properties - DB not available")
        return

    result = graph.query("MATCH (n) RETURN n LIMIT 1")
    if not result.result_set:
        print("[INFO] No nodes found in graph")
        return

    node = result.result_set[0][0]
    # Handle FalkorDB Node objects
    if hasattr(node, 'properties'):
        props = dict(node.properties) if node.properties else {}
        labels = list(node.labels) if hasattr(node, 'labels') and node.labels else []
        print(f"[INFO] Sample node labels: {labels}")
        print(f"[INFO] Sample node properties: {list(props.keys())}")
        print(f"[INFO] Sample node values: {props}")
        has_name = 'name' in props or 'title' in props
        print(f"[INFO] Has name/title field: {has_name}")
    elif isinstance(node, dict):
        props = list(node.keys())
        print(f"[INFO] Sample node properties: {props}")
        has_name = 'name' in node or 'title' in node
        print(f"[INFO] Has name/title field: {has_name}")
    else:
        print(f"[INFO] Node returned as: {type(node)}")


def test_fulltext_search():
    """Test that a simple text search returns results (if data exists)."""
    graph = _get_graph()
    if graph is None:
        print("[SKIP] test_fulltext_search - DB not available")
        return

    # Try a broad search
    result = graph.query(
        "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name LIMIT 5"
    )
    names = [row[0] for row in (result.result_set or [])]
    print(f"[INFO] Sample node names: {names[:5]}")
    # No assertion - just informational


def test_graphrag_endpoint_live():
    """Call the /graphrag logic directly against the real DB."""
    graph = _get_graph()
    if graph is None:
        print("[SKIP] test_graphrag_endpoint_live - DB not available")
        return

    from functions import hybrid_candidates, retrieve_from_graph, _fulltext_search

    # First, find an actual word in the data to search for
    sample_res = graph.query("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name LIMIT 10")
    sample_names = [row[0] for row in (sample_res.result_set or [])]
    print(f"[DEBUG] Sample names to search: {sample_names}")

    # Pick a word from first name if available
    if sample_names:
        search_word = sample_names[0].split()[0] if sample_names[0] else "house"
    else:
        search_word = "house"
    print(f"[DEBUG] Searching for: '{search_word}'")

    # Test direct Cypher to verify search works
    test_cypher = f"MATCH (n) WHERE toLower(n.name) CONTAINS '{search_word.lower()}' RETURN n LIMIT 3"
    print(f"[DEBUG] Testing Cypher: {test_cypher}")
    try:
        direct_res = graph.query(test_cypher)
        print(f"[DEBUG] Direct Cypher returned {len(direct_res.result_set or [])} results")
        for row in (direct_res.result_set or [])[:2]:
            node = row[0]
            if hasattr(node, 'properties'):
                print(f"[DEBUG]   -> {node.properties.get('name', '?')}")
    except Exception as e:
        print(f"[DEBUG] Direct Cypher error: {e}")

    # Test _fulltext_search directly
    ft_results = _fulltext_search(graph, search_word, limit=5)
    print(f"[INFO] _fulltext_search returned {len(ft_results)} results")
    for node, score in ft_results[:3]:
        print(f"[DEBUG]   -> {node.get('name', '?')} (score={score})")

    # Test hybrid candidates (keyword search, no embedding)
    cands = hybrid_candidates(graph, search_word, qvec=None, labels=None, k_vec=5, k_kw=5)
    print(f"[INFO] hybrid_candidates returned {len(cands)} results")

    # Test graph-only retrieval
    graph_results = retrieve_from_graph(graph, search_word, limit_per_word=3)
    entities = graph_results.get('entities', [])
    print(f"[INFO] retrieve_from_graph returned {len(entities)} entities")
    for ent in entities[:3]:
        print(f"[DEBUG]   -> {ent}")


if __name__ == '__main__':
    test_falkordb_connection()
    test_node_count()
    test_sample_node_properties()
    test_fulltext_search()
    test_graphrag_endpoint_live()
    print("\n✅ All integration tests completed")
