"""Moved graphrag helper functions into package location.
"""
from typing import List, Optional, Any, Tuple, Dict
import os
import logging
import re
import math
import httpx

# Shared constants matching Neo4j configuration
DEFAULT_LABELS = [
    "Stakeholder", "Goal", "Challenge", "Outcome", "Policy", "Strategy", "Pillar", "Sector",
    "Time_Period", "Infrastructure", "Technology", "Initiative", "Objective", "Target",
    "Opportunity", "Vision", "Region", "Enabler", "Entity"
]

_ANCHOR_RE = re.compile(r'"([^"]+)"|"([^"]+)"|\u2018([^\u2019]+)\u2019|\'([^\']+)\'')


def _node_to_dict(node) -> dict:
    """Convert a FalkorDB Node object (or any node-like object) to a plain dict.
    
    FalkorDB returns Node objects with .properties and .labels attributes.
    This helper normalizes them to dicts for consistent access.
    """
    if isinstance(node, dict):
        return node
    
    # FalkorDB Node object
    if hasattr(node, 'properties'):
        d = dict(node.properties) if node.properties else {}
        if hasattr(node, 'labels') and node.labels:
            d['labels'] = list(node.labels)
        if hasattr(node, 'id'):
            d.setdefault('id', node.id)
        return d
    
    # Fallback for scalar or unknown types
    return {"id": str(node), "name": str(node)}


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
    """Cosine similarity between two vectors."""
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


def _minmax_norm(values: List[float]) -> List[float]:
    """Min-max normalization matching Neo4j implementation."""
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _node_embedding(node: dict) -> Optional[List[float]]:
    """Return node embedding as a Python list (or None). Matches Neo4j."""
    e = node.get("embedding")
    return list(e) if e is not None else None


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
    """
    Keyword-based search approximating BM25 behavior.
    Scoring heuristics:
    - Anchor terms (quoted/title-cased) get higher weight
    - Exact matches score higher than partial
    - Multiple term matches boost score
    """
    labels = labels or DEFAULT_LABELS
    anchors = anchor_terms(question, max_terms=3)
    kws = extract_keywords(question, max_terms=8)
    terms = anchors + [k for k in kws if k not in anchors]

    if not terms:
        return []

    # Track scores per node for multi-term boosting
    scores: Dict[str, Dict[str, Any]] = {}
    anchor_set = set(a.lower() for a in anchors)

    for idx, t in enumerate(terms):
        pattern = t.lower().replace("'", "\\'")
        # Use toString() to safely handle non-string properties before toLower
        q = (
            f"MATCH (n) WHERE "
            f"(n.name IS NOT NULL AND toLower(toString(n.name)) CONTAINS '{pattern}') OR "
            f"(n.title IS NOT NULL AND toLower(toString(n.title)) CONTAINS '{pattern}') OR "
            f"(n.text IS NOT NULL AND toLower(toString(n.text)) CONTAINS '{pattern}') OR "
            f"(n.content IS NOT NULL AND toLower(toString(n.content)) CONTAINS '{pattern}') "
            f"RETURN n LIMIT {limit * 2}"
        )
        try:
            res = graph.query(q)
            for row in (res.result_set or []):
                node = _node_to_dict(row[0])
                nid = node.get('id') or node.get('name') or str(node)
                name_val = str(node.get('name') or '').lower()

                if nid not in scores:
                    scores[nid] = {"node": node, "score": 0.0, "match_count": 0}

                # Scoring heuristics (approximating BM25 behavior)
                term_score = 0.0
                
                # Base score: term length (longer = more specific)
                term_score += len(t) / 10.0
                
                # Anchor bonus: anchor terms are more important
                if t.lower() in anchor_set:
                    term_score += 2.0
                
                # Position bonus: earlier terms in query are often more important
                term_score += (len(terms) - idx) / len(terms)
                
                # Exact match bonus
                if name_val == pattern:
                    term_score += 3.0
                elif name_val.startswith(pattern) or name_val.endswith(pattern):
                    term_score += 1.5
                
                scores[nid]["score"] += term_score
                scores[nid]["match_count"] += 1
        except Exception:
            continue

    # Multi-term match bonus (nodes matching multiple terms score higher)
    out = []
    for entry in scores.values():
        final_score = entry["score"]
        if entry["match_count"] > 1:
            final_score *= (1.0 + 0.2 * (entry["match_count"] - 1))  # 20% boost per extra match
        out.append((entry["node"], float(final_score)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:limit]


def _vector_search(graph, qvec: List[float], top_k: int = 12) -> List[Tuple[dict, float]]:
    """Vector similarity search using cosine similarity. Matches Neo4j vector_find_similar_nodes."""
    try:
        res = graph.query("MATCH (n) WHERE n.embedding IS NOT NULL RETURN n LIMIT 500")
        rows = res.result_set or []
        candidates = []
        for row in rows:
            node = _node_to_dict(row[0])
            emb = node.get('embedding')
            if not emb:
                continue
            try:
                sc = cosine(qvec, list(emb))
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
                # Use toString() to safely handle non-string properties before toLower
                cypher_query = (
                    f"MATCH (n) WHERE "
                    f"(n.name IS NOT NULL AND toLower(toString(n.name)) CONTAINS '{word}') OR "
                    f"(n.title IS NOT NULL AND toLower(toString(n.title)) CONTAINS '{word}') OR "
                    f"(n.text IS NOT NULL AND toLower(toString(n.text)) CONTAINS '{word}') OR "
                    f"(n.content IS NOT NULL AND toLower(toString(n.content)) CONTAINS '{word}') "
                    f"RETURN n LIMIT {limit_per_word}"
                )
                res = graph.query(cypher_query)
            except Exception:
                continue

            for row in (res.result_set or []):
                node = _node_to_dict(row[0])
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
    """Hybrid retrieval: combine vector and fulltext scores. Matches Neo4j implementation."""
    labels = labels or DEFAULT_LABELS
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

    # Normalize per channel (matches Neo4j _minmax_norm)
    vec_n = _minmax_norm([v["vec"] for v in raw.values()])
    kw_n = _minmax_norm([v["kw"] for v in raw.values()])

    # Blend vector + keyword (renormalize weights to sum to 1, matches Neo4j)
    w_sum = max(1e-12, (alpha_vec + beta_kw))
    w_vec = alpha_vec / w_sum
    w_kw = beta_kw / w_sum

    out = []
    for entry, vn, kn in zip(raw.values(), vec_n, kw_n):
        combined = w_vec * vn + w_kw * kn
        out.append((entry["node"], float(combined)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:max(k_vec, k_kw)]


def expand_neighbors_by_name(graph, seed_names: List[str], hops: int = 1):
    """Expand neighborhood around seeds. Matches Neo4j traverse_neighbors behavior."""
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
                a, r, b = _node_to_dict(row[0]), row[1], _node_to_dict(row[2])
                aid = a.get('id') or a.get('name')
                bid = b.get('id') or b.get('name')
                if aid not in seen_nodes:
                    seen_nodes[aid] = a
                if bid not in seen_nodes:
                    seen_nodes[bid] = b
                rels.append({
                    'type': r.get('type') if isinstance(r, dict) else (r.type if hasattr(r,'type') else 'REL'),
                    'props': r.get('props') if isinstance(r, dict) else (dict(r.properties) if hasattr(r, 'properties') else {}),
                    'start': aid,
                    'end': bid
                })
        except Exception:
            continue

    return {"nodes": list(seen_nodes.values()), "rels": rels}


# =========================
# MMR diversification (matches Neo4j)
# =========================
def mmr_select(candidates: List[Tuple[dict, float]],
               k: int,
               lambda_mult: float = 0.7) -> List[Tuple[dict, float]]:
    """
    Maximal Marginal Relevance over candidate nodes.
    Matches Neo4j implementation with lambda_mult=0.7.
    Uses node embeddings for diversity; falls back to score-only if embeddings missing.
    """
    if not candidates or k <= 0:
        return []

    # Fetch embeddings
    embs: List[Optional[List[float]]] = []
    for n, _ in candidates:
        embs.append(_node_embedding(n))

    selected: List[int] = []
    rest = list(range(len(candidates)))

    # Seed: best score
    best0 = max(rest, key=lambda i: candidates[i][1])
    selected.append(best0)
    rest.remove(best0)

    def _max_sim_to_selected(j: int) -> float:
        ej = embs[j]
        if ej is None or not selected:
            return 0.0
        sims = []
        for i in selected:
            ei = embs[i]
            if ei is None:
                sims.append(0.0)
            else:
                sims.append(cosine(ej, ei))
        return max(sims) if sims else 0.0

    while len(selected) < min(k, len(candidates)) and rest:
        best_j, best_val = None, -1e9
        for j in rest:
            relevance = candidates[j][1]
            diversity_penalty = _max_sim_to_selected(j)
            val = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if val > best_val:
                best_val, best_j = val, j
        if best_j is not None:
            selected.append(best_j)
            rest.remove(best_j)
        else:
            break

    return [candidates[i] for i in selected]


# =========================
# Cross-document coverage (matches Neo4j)
# =========================
def _doc_title_for_node(graph, node: dict) -> Optional[str]:
    """Get document title for a node via SOURCE or MENTIONS relationship."""
    name = node.get('name')
    if not name:
        return None
    safe = str(name).replace("'", "\\'")
    try:
        res = graph.query(f"""
            MATCH (n {{name: '{safe}'}})
            OPTIONAL MATCH (n)-[:SOURCE]->(d1:Document)
            OPTIONAL MATCH (d2:Document)-[:MENTIONS]->(n)
            RETURN coalesce(d1.title, d2.title, d1.name, d2.name) AS t
            LIMIT 1
        """)
        if res.result_set and res.result_set[0]:
            return res.result_set[0][0]
    except Exception:
        pass
    return None


def diversify_by_document(graph, cands: List[Tuple[dict, float]], k: int) -> List[Tuple[dict, float]]:
    """
    Prefer seeds from different documents (round-robin by doc title).
    Matches Neo4j diversify_by_document implementation.
    """
    if not cands:
        return []

    buckets: Dict[str, List[Tuple[dict, float]]] = {}
    for n, sc in cands:
        t = _doc_title_for_node(graph, n) or "__NO_DOC__"
        buckets.setdefault(t, []).append((n, sc))

    # Sort buckets by best score inside
    for b in buckets.values():
        b.sort(key=lambda t: t[1], reverse=True)

    order = sorted(buckets.keys(), key=lambda key: buckets[key][0][1], reverse=True)

    picked: List[Tuple[dict, float]] = []
    ptrs = {key: 0 for key in buckets.keys()}
    while len(picked) < min(k, len(cands)):
        progressed = False
        for key in order:
            i = ptrs[key]
            if i < len(buckets[key]):
                picked.append(buckets[key][i])
                ptrs[key] += 1
                progressed = True
                if len(picked) >= k:
                    break
        if not progressed:
            break

    return picked
