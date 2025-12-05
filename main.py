from fastapi import FastAPI, HTTPException, Body, Request
from falkordb import FalkorDB
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Tuple, Dict
import time
from time import perf_counter
import re
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FalkorDB Knowledge API", version="1.0.0")

# Load environment variables
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
LLM_API_BASE = os.getenv("LLM_API_BASE")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
GENERATION_API_KEY = os.getenv("GENERATION_API_KEY")

# Environment
ENV = os.getenv("ENV", "development")

# Graph name from refer.py
GRAPH_NAME = os.getenv("GRAPH_NAME", "iscs_knowledge_graph")


# GraphRAG request body (compatible with neo4j implementation)
class RagBody(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = 10
    hops: int = 1
    labels: Optional[List[str]] = None
    alpha_vec: float = 0.6
    beta_kw: float = 0.4
    use_mmr: bool = True
    use_cross_doc: bool = True
from functions import (
    anchor_terms,
    extract_keywords,
    get_embedding,
    check_embedding,
    cosine,
    format_graph_facts,
    hybrid_candidates,
    expand_neighbors_by_name,
    retrieve_from_graph,
)

@app.get("/db-summary")
async def db_summary():
    """
    Get a summary of the current database contents
    """
    try:
        # Connect to FalkorDB
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        
        # Select the unified_knowledge_graph
        graph = db.select_graph(GRAPH_NAME)
        
        # Get node count
        node_result = graph.query("MATCH (n) RETURN count(n) as node_count")
        node_count = node_result.result_set[0][0] if node_result.result_set else 0
        
        # Get relationship count
        rel_result = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        rel_count = rel_result.result_set[0][0] if rel_result.result_set else 0
        
        # Get labels
        label_result = graph.query("CALL db.labels() YIELD label RETURN collect(label) as labels")
        labels = label_result.result_set[0][0] if label_result.result_set else []
        
        return {
            "success": True,
            "message": "Database summary retrieved successfully",
            "data": {
                "graph": GRAPH_NAME,
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels
            }
        }
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to retrieve database summary: {str(e)}")


@app.post("/graphrag")
async def graphrag(body: RagBody = Body(...), request: Request = None):
    """A FalkorDB-compatible version of the GraphRAG endpoint.
    Returns graph-based context and seeds. This implementation follows the same
    request body as the Neo4j service and attempts to produce a similar shaped response.
    """
    try:
        if not body.question or not body.question.strip():
            return {"success": False, "message": "Please provide a question.", "answer": "", "facts": "", "seeds": []}

        t0 = perf_counter()
        q0 = body.question.strip()
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        graph = db.select_graph(GRAPH_NAME)
        print(f"[graphrag] Received question: {q0} + graph: {graph}")

        # 1) embeddings (optional) — check connectivity first
        embedding_available = await check_embedding(EMBEDDING_API_BASE, EMBEDDING_API_KEY)
        print(f"[graphrag] Embedding available: {embedding_available}")
        qvec = None
        if embedding_available:
            qvec = await get_embedding(EMBEDDING_API_BASE, EMBEDDING_API_KEY, q0)

        # 2) hybrid candidates
        cands = hybrid_candidates(graph, q0, qvec=qvec, labels=body.labels, k_vec=max(12, body.top_k), k_kw=max(12, body.top_k), alpha_vec=body.alpha_vec, beta_kw=body.beta_kw)
        
        if not cands:
            # 2a) If hybrid failed, try a refer-style graph-only retrieval
            try:
                graph_results = retrieve_from_graph(graph, q0, limit_per_word=max(3, body.top_k))
                if graph_results and graph_results.get('entities'):
                    # convert entities into the same shape hybrid_candidates would return
                    cands = [( { 'name': e.get('name'), 'labels': [e.get('type')] if e.get('type') else [] }, 1.0) for e in graph_results.get('entities') ]
            except Exception:
                # best-effort fallback; ignore errors and continue to return empty result
                pass

            if not cands:
                msg = "There is no available data related to the user query."
                return {
                "success": True,
                "message": msg,
                "answer": msg,
                "facts": f"Q: {q0}\nGraph Facts: (no results)",
                "seeds": [],
                "params": {
                    "top_k": body.top_k,
                    "hops": body.hops,
                    "labels": body.labels,
                    "alpha_vec": body.alpha_vec,
                    "beta_kw": body.beta_kw,
                    "use_mmr": body.use_mmr,
                    "use_cross_doc": body.use_cross_doc,
                    "used_llm": False,
                    "embedding_available": bool(embedding_available),
                },
                "timings": {"total": perf_counter()-t0}
            }

        # limit to top_k
        cands = cands[:body.top_k]

        # choose seed names
        seed_names = []
        for n, sc in cands:
            nm = n.get('name') or n.get('title') or None
            if nm and nm not in seed_names:
                seed_names.append(nm)

        # expand neighbors
        expanded = expand_neighbors_by_name(graph, seed_names[:body.top_k], hops=max(1, min(body.hops, 3)))

        # format facts — prefer expanded neighbor facts when available
        facts = format_graph_facts(expanded.get('nodes', []), expanded.get('rels', []), include_source=True)

        # If expansion yields nothing but we have seed names (from retrieve_from_graph
        # or hybrid candidates) include the seed names into facts so the user gets
        # a helpful response instead of an all-or-nothing "no results" message.
        if (not expanded.get('nodes') and not expanded.get('rels')) and seed_names:
            # build a simple facts string listing seed names and optionally any
            # relationships returned by a graph-only retriever (if present)
            facts_lines = ["Graph Facts:"]
            for nm in seed_names[:body.top_k]:
                facts_lines.append(f'- Seed: {nm}')
            facts = "\n".join(facts_lines)

        if facts.strip().endswith("(no results)"):
            msg = "There is no available data related to the user query."
            return {
                "success": True,
                "message": msg,
                "answer": msg,
                "facts": f"Q: {q0}\n{facts}",
                "seeds": [],
                "params": {
                    "top_k": body.top_k,
                    "hops": body.hops,
                    "labels": body.labels,
                    "alpha_vec": body.alpha_vec,
                    "beta_kw": body.beta_kw,
                    "use_mmr": body.use_mmr,
                    "use_cross_doc": body.use_cross_doc,
                    "used_llm": False,
                    "embedding_available": bool(embedding_available),
                },
                "timings": {"total": perf_counter()-t0}
            }

        # Build seeds metadata consistently — ensure 'name' is a readable string.
        seeds_meta = []
        for n, sc in cands:
            # Normalize node to dict-like access where possible
            if isinstance(n, dict):
                labels = list(n.get('labels', [])) if isinstance(n.get('labels', []), list) else []
                name = n.get('name') or n.get('title') or n.get('id') or None
            else:
                # Node might be a scalar (string or number) returned by some cypher queries
                labels = []
                name = str(n) if n is not None else None

            # Final fallback for name to avoid empty/null names
            if not name:
                name = "<unknown>"

            try:
                score_val = float(sc)
            except Exception:
                # ensure score is numeric
                score_val = 0.0

            seeds_meta.append({"labels": labels, "name": name, "score": score_val})

        return {
            "success": True,
            "message": "Query processed.",
            "answer": facts,
            "facts": f"Q: {q0}\n{facts}",
            "seeds": seeds_meta,
            "params": {
                "top_k": body.top_k,
                "hops": body.hops,
                "labels": body.labels,
                "alpha_vec": body.alpha_vec,
                "beta_kw": body.beta_kw,
                "use_mmr": body.use_mmr,
                "use_cross_doc": body.use_cross_doc,
                "used_llm": False,
                "embedding_available": bool(embedding_available),
            },
            "timings": {"total": perf_counter()-t0}
        }

    except Exception as e:
        return {"success": False, "message": f"Query failed: {str(e)}", "error_type": type(e).__name__, "error_details": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Enable reload only in development
    reload_enabled = ENV == "development"
    
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=reload_enabled)