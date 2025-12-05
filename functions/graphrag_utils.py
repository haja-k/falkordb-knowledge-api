"""Moved graphrag helper functions into package location.
"""
from typing import List, Optional, Any, Tuple, Dict
import os
import logging
import re
import httpx

_ANCHOR_RE = re.compile(r'"([^\"]+)"|“([^”]+)”|‘([^’]+)’|\'([^\']+)\'')


def anchor_terms(question: str, max_terms: int = 3) -> List[str]:
    anchors: list[str] = []
    for g in _ANCHOR_RE.findall(question):
        val = next((x for x in g if x), "").strip()
        if val and val.lower() not in ("and", "or", "the"):
            anchors.append(val)

    if len(anchors) < max_terms:
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", question)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1[0].isupper() and w2[0].isupper():
                pair = f"{w1} {w2}"
                if pair not in anchors:
                    anchors.append(pair)
                    if len(anchors) >= max_terms:
                        break

    if not anchors:
        kws = re.findall(r"[A-Za-z0-9\-']+", question)
        kws = sorted(kws, key=len, reverse=True)
        if kws:
            anchors.append(kws[0])

    seen = set()
    out = []
    for a in anchors:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out[:max_terms]


def extract_keywords(question: str, max_terms: int = 8) -> List[str]:
    stop = {"the","is","are","a","an","of","in","for","to","with","and","or","on"}
    toks = re.findall(r"[A-Za-z0-9\-']+", question)
    toks = [t for t in toks if t.lower() not in stop and len(t) > 2]
    toks = sorted(toks, key=len, reverse=True)
    return toks[:max_terms]


async def get_embedding(endpoint: Optional[str], api_key: Optional[str], question: str) -> Optional[List[float]]:
    if not endpoint:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            payload = {"input": question}
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("data") and isinstance(data["data"], list):
                emb = data["data"][0].get("embedding")
                return emb
            if isinstance(data, dict) and data.get("embedding"):
                return data["embedding"]
    except Exception:
        return None


async def check_embedding(endpoint: Optional[str], api_key: Optional[str]) -> bool:
    """Quick health/check for an embedding endpoint.

    Returns True when the endpoint responds to a small request and returns
    JSON indicating embeddings (or at least status 200). This is intentionally
    tolerant — it returns False for any exception or unexpected response.
    """
    if not endpoint:
        if os.getenv('DEBUG_EMBEDDING_CHECK'):
            print('[check_embedding] no endpoint configured')
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            # lightweight payload
            payload = {"input": "ping"}
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # many embedding services return {data:[{embedding: [...]}, ...]}
            if isinstance(data, dict) and (data.get("data") or data.get("embedding")):
                return True
            # If it's a plain success status code, consider it OK
            return resp.status_code == 200
    except Exception as e:
        # When debugging is enabled, provide extra info to help tracing failures
        if os.getenv('DEBUG_EMBEDDING_CHECK'):
            try:
                # Print limited information but avoid leaking full API key
                print(f"[check_embedding] exception contacting {endpoint}: {type(e).__name__} - {str(e)[:200]}")
            except Exception:
                pass
        return False


def cosine(a: List[float], b: List[float]) -> float:
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / ((da ** 0.5) * (db ** 0.5))


def format_graph_facts(nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]], include_source: bool = False, max_lines: Optional[int] = None, snippet_chars: Optional[int] = None) -> str:
    if not nodes or not rels:
        return "Graph Facts: (no results)"

    nodes_by_id = {n.get('id') or n.get('name'): n for n in nodes}
    out_lines = ["Graph Facts:"]
    for r in rels[:max_lines] if isinstance(max_lines, int) and max_lines > 0 else rels:
        s = nodes_by_id.get(r.get('start')) or {}
        t = nodes_by_id.get(r.get('end')) or {}
        s_name = s.get('name') or s.get('title') or "?"
        t_name = t.get('name') or t.get('title') or "?"
        s_label = (s.get('labels') or ["Entity"])[0]
        t_label = (t.get('labels') or ["Entity"])[0]
        snip = (r.get('props') or {}).get('source_text', '')
        if snippet_chars and isinstance(snippet_chars, int) and len(snip) > snippet_chars:
            snip = snip[:snippet_chars].rstrip() + "..."
        snip_str = f' [snippet: "{snip}"]' if snip else ''
        out_lines.append(f'- {s_label}("{s_name}") -[{r.get("type") }]-> {t_label}("{t_name}"){snip_str}')
    return "\n".join(out_lines)


def _fulltext_search(graph, question: str, limit: int = 12, labels: Optional[List[str]] = None) -> List[Tuple[dict, float]]:
    labels = labels or []
    anchors = anchor_terms(question, max_terms=3)
    kws = extract_keywords(question, max_terms=8)
    terms = anchors + [k for k in kws if k not in anchors]

    if not terms:
        return []

    out = []
    seen = set()
    for t in terms:
        pattern = t.lower()
        # try matching against common text fields in nodes: name, title, text, content
        q = (
            f"MATCH (n) WHERE toLower(n.name) CONTAINS '{pattern}' OR toLower(n.title) CONTAINS '{pattern}' "
            f"OR toLower(n.text) CONTAINS '{pattern}' OR toLower(n.content) CONTAINS '{pattern}' RETURN n LIMIT {limit}"
        )
        try:
            res = graph.query(q)
            for row in (res.result_set or []):
                node = row[0]
                # handle cases where cypher returns scalar fields rather than full node
                if not isinstance(node, dict):
                    # attempt to synthesize node dict if we got a scalar name
                    node = {"id": str(node), "name": str(node)}
                nid = node.get('id') or node.get('name') or str(node)
                if nid in seen:
                    continue
                seen.add(nid)
                score = len(t)
                out.append((node, float(score)))
        except Exception:
            continue

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:limit]


def _vector_search(graph, qvec: List[float], top_k: int = 12) -> List[Tuple[dict, float]]:
    try:
        res = graph.query("MATCH (n) WHERE exists(n.embedding) RETURN n LIMIT 500")
        rows = res.result_set or []
        candidates = []
        for row in rows:
            node = row[0]
            emb = node.get('embedding')
            if not emb:
                continue
            try:
                sc = cosine(qvec, emb)
            except Exception:
                sc = 0.0
            candidates.append((node, float(sc)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    except Exception:
        return []


def retrieve_from_graph(graph, query: str, limit_per_word: int = 5) -> Dict[str, Any]:
    """Refer-style graph retrieval: search for tokens in name/title/text/content and
    then fetch relationships for the top match.

    Returns a dict with keys: 'entities' (list of {name,type}), 'relationships' (list)
    """
    results = []
    query_words = [w.lower() for w in re.findall(r"[A-Za-z0-9\-']+", query) if len(w) > 2]

    try:
        for word in query_words:
            try:
                cypher_query = (
                    f"MATCH (n) WHERE toLower(n.name) CONTAINS '{word}' OR toLower(n.title) CONTAINS '{word}' "
                    f"OR toLower(n.text) CONTAINS '{word}' OR toLower(n.content) CONTAINS '{word}' RETURN n LIMIT {limit_per_word}"
                )
                res = graph.query(cypher_query)
            except Exception:
                continue

            for row in (res.result_set or []):
                node = row[0]
                if not isinstance(node, dict):
                    node = {"id": str(node), "name": str(node)}
                results.append({"name": node.get('name') or node.get('title') or str(node.get('id')), "type": (node.get('labels') or [None])[0] if isinstance(node.get('labels', []), list) else None})

        # deduplicate
        seen = set()
        unique = []
        for r in results:
            if r['name'] not in seen:
                seen.add(r['name'])
                unique.append(r)

        relationships = []
        if unique:
            # fetch relationships for the top entity if any
            entity_name = unique[0]['name']
            safe_name = entity_name.replace("'", "\\'")
            try:
                rel_q = f"MATCH (a {{name: '{safe_name}'}})-[r]->(b) RETURN type(r) as relationship, b.name as target LIMIT 20"
                rel_res = graph.query(rel_q)
                for row in (rel_res.result_set or []):
                    relationships.append({"relationship": row[0], "target": row[1]})
            except Exception:
                # don't fail hard on relationships
                pass

        return {"entities": unique, "relationships": relationships, "type": 'graph'}

    except Exception as e:
        # return a typed error response so callers can handle it
        return {"entities": [], "relationships": [], "type": 'graph_error', "error": str(e)}


def hybrid_candidates(graph, question: str, qvec: Optional[List[float]] = None, labels: Optional[List[str]] = None, k_vec: int = 12, k_kw: int = 12, alpha_vec: float = 0.6, beta_kw: float = 0.25) -> List[Tuple[dict, float]]:
    labels = labels or []
    vec_hits = _vector_search(graph, qvec, top_k=k_vec) if qvec else []
    kw_hits = _fulltext_search(graph, question, limit=k_kw, labels=labels)

    raw: Dict[str, Dict[str, Any]] = {}
    for n, sc in vec_hits:
        nid = n.get('id') or n.get('name') or str(n)
        raw.setdefault(nid, {"node": n, "vec": 0.0, "kw": 0.0})
        raw[nid]["vec"] = max(raw[nid]["vec"], float(sc))
    for n, sc in kw_hits:
        nid = n.get('id') or n.get('name') or str(n)
        raw.setdefault(nid, {"node": n, "vec": 0.0, "kw": 0.0})
        raw[nid]["kw"] = max(raw[nid]["kw"], float(sc))

    if not raw:
        return []

    vec_vals = [v["vec"] for v in raw.values()]
    kw_vals = [v["kw"] for v in raw.values()]
    def _minmax(vals):
        if not vals:
            return []
        lo = min(vals); hi = max(vals)
        if hi - lo < 1e-12:
            return [0.5 for _ in vals]
        return [(x-lo)/(hi-lo) for x in vals]

    vec_n = _minmax(vec_vals)
    kw_n = _minmax(kw_vals)

    out = []
    for entry, vn, kn in zip(raw.values(), vec_n, kw_n):
        w_sum = max(1e-12, (alpha_vec + beta_kw))
        w_vec = alpha_vec / w_sum
        w_kw = beta_kw / w_sum
        combined = w_vec*vn + w_kw*kn
        out.append((entry["node"], float(combined)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:max(k_vec, k_kw)]


def expand_neighbors_by_name(graph, seed_names: List[str], hops: int = 1):
    nodes: List[Dict[str, Any]] = []
    rels: List[Dict[str, Any]] = []
    seen_nodes = {}

    if not seed_names:
        return {"nodes": [], "rels": []}

    for name in seed_names:
        safe = name.replace("'", "\\'")
        try:
            q = f"MATCH (a {{name: '{safe}'}})-[r]-(b) RETURN a,r,b LIMIT 200"
            res = graph.query(q)
            for row in (res.result_set or []):
                a, r, b = row[0], row[1], row[2]
                aid = a.get('id') or a.get('name')
                bid = b.get('id') or b.get('name')
                if aid not in seen_nodes:
                    seen_nodes[aid] = a
                if bid not in seen_nodes:
                    seen_nodes[bid] = b
                rels.append({
                    'type': r.get('type') if isinstance(r, dict) else (r.type if hasattr(r,'type') else 'REL'),
                    'props': r.get('props') if isinstance(r, dict) else {},
                    'start': aid,
                    'end': bid
                })
        except Exception:
            continue

    return {"nodes": list(seen_nodes.values()), "rels": rels}
