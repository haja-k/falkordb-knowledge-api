import math

from functions import graphrag_utils as gu


def test_anchor_terms_quotes_and_titlecase():
    q = 'Find information about "Route Location" and Port Location and small'
    anchors = gu.anchor_terms(q, max_terms=3)
    assert "Route Location" in anchors
    assert any(a.endswith("Location") for a in anchors)


def test_extract_keywords_basic():
    q = "What stakeholders and policies affect Route Location planning in 2025?"
    kws = gu.extract_keywords(q, max_terms=6)
    assert any(k.lower() in ("route","location","planning") for k in kws)
    assert all(len(k) > 2 for k in kws)


def test_cosine_edge_and_identity():
    assert math.isclose(gu.cosine([1, 0], [1, 0]), 1.0, rel_tol=1e-6)
    assert math.isclose(gu.cosine([1, 0], [0, 1]), 0.0, rel_tol=1e-6)


def test_format_graph_facts_simple():
    nodes = [
        {"id": "n1", "name": "Alice", "labels": ["Person"]},
        {"id": "n2", "name": "Project X", "labels": ["Project"]},
    ]
    rels = [
        {"type": "LEADS", "props": {"source_text": "Alice leads Project X"}, "start": "n1", "end": "n2"}
    ]
    out = gu.format_graph_facts(nodes, rels, include_source=False)
    assert "Alice" in out and "Project X" in out
    assert "LEADS" in out
