from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Request
from falkordb import FalkorDB
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Tuple, Dict
from time import perf_counter
import os
from dotenv import load_dotenv

load_dotenv()

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_USERNAME = os.getenv("FALKORDB_USERNAME")
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
GRAPH_NAME = os.getenv("GRAPH_NAME", "iscs_knowledge_graph")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "4040"))
ENV = os.getenv("ENV", "development")

EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")


class RagBody(BaseModel):
    """Request body for /graphrag endpoint."""
    question: str = Field(..., description="User question")
    top_k: int = 10  # Max results to return
    hops: int = 1  # Neighbor expansion depth (1-3)
    labels: Optional[List[str]] = None  # Filter by node labels
    alpha_vec: float = 0.6  # Vector search weight
    beta_kw: float = 0.25  # Keyword search weight (matches Neo4j)
    use_mmr: bool = True  # MMR diversification (lambda=0.7)
    use_cross_doc: bool = True  # Cross-document diversification

from functions import (
    get_embedding,
    check_embedding,
    format_graph_facts,
    hybrid_candidates,
    expand_neighbors_by_name,
    retrieve_from_graph,
    mmr_select,
    diversify_by_document,
)

# Load stuff during startup (before yield). 
# Unload stuff during stopping (after yield).
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Database driver
    database = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
    graph = database.select_graph(GRAPH_NAME)
    app.state.graph = graph
    yield

app = FastAPI(
    title="FalkorDB Knowledge API", 
    version="1.0.0", 
    lifespan=lifespan
    )


@app.get("/db-summary")
async def db_summary():
    """Get node count, relationship count, and labels from the graph."""
    try:
        # Accessing graph using app state resources
        graph = app.state.graph
        
        node_result = graph.query("MATCH (n) RETURN count(n) as node_count")
        node_count = node_result.result_set[0][0] if node_result.result_set else 0
        
        rel_result = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        rel_count = rel_result.result_set[0][0] if rel_result.result_set else 0
        
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
    """
    Hybrid graph RAG endpoint. Combines vector + keyword search to find relevant nodes,
    then expands neighbors to build context. Returns graph facts and seed nodes.
    """
    try:
        if not body.question or not body.question.strip():
            return {
                "success": False,
                "message": "Please provide a question.",
                "data": None
            }

        t0 = perf_counter()
        q0 = body.question.strip()
        graph = app.state.graph
        print(f"[graphrag] Received question: {q0}")

        # Get question embedding if embedding service is available
        embedding_available = await check_embedding(EMBEDDING_API_BASE, EMBEDDING_API_KEY)
        print(f"[graphrag] Using embedding service at: {EMBEDDING_API_BASE}")
        print(f"[graphrag] Embedding available: {embedding_available}")
        qvec = await get_embedding(EMBEDDING_API_BASE, EMBEDDING_API_KEY, q0) if embedding_available else None

        # Hybrid search: vector + keyword
        cands = hybrid_candidates(
            graph, q0, qvec=qvec, labels=body.labels,
            k_vec=max(12, body.top_k), k_kw=max(12, body.top_k),
            alpha_vec=body.alpha_vec, beta_kw=body.beta_kw
        )
        
        # Fallback: text-only graph retrieval if hybrid found nothing
        if not cands:
            try:
                graph_results = retrieve_from_graph(graph, q0, limit_per_word=max(3, body.top_k))
                if graph_results and graph_results.get('entities'):
                    cands = [
                        ({'name': e.get('name'), 'labels': [e.get('type')] if e.get('type') else []}, 1.0)
                        for e in graph_results.get('entities')
                    ]
            except Exception:
                pass

            if not cands:
                return _empty_response(q0, body, t0, embedding_available)

        # Apply MMR diversification if enabled (lambda_mult=0.7 matches Neo4j)
        if body.use_mmr and len(cands) > body.top_k:
            cands = mmr_select(cands, k=body.top_k * 2, lambda_mult=0.7)

        # Apply cross-document diversification if enabled (matches Neo4j)
        if body.use_cross_doc and len(cands) > body.top_k:
            cands = diversify_by_document(graph, cands, k=body.top_k * 2)

        cands = cands[:body.top_k]

        # Extract seed names for neighbor expansion
        seed_names = []
        for n, _ in cands:
            nm = n.get('name') or n.get('title')
            if nm and nm not in seed_names:
                seed_names.append(nm)

        # Expand graph neighbors from seed nodes
        expanded = expand_neighbors_by_name(graph, seed_names[:body.top_k], hops=max(1, min(body.hops, 3)))
        facts = format_graph_facts(expanded.get('nodes', []), expanded.get('rels', []), include_source=True)

        # If no neighbors found, at least return seed names
        if (not expanded.get('nodes') and not expanded.get('rels')) and seed_names:
            facts_lines = ["Graph Facts:"] + [f'- Seed: {nm}' for nm in seed_names[:body.top_k]]
            facts = "\n".join(facts_lines)

        if facts.strip().endswith("(no results)"):
            return _empty_response(q0, body, t0, embedding_available)

        # Build response with seed metadata
        seeds_meta = []
        for n, sc in cands:
            if isinstance(n, dict):
                labels = list(n.get('labels', [])) if isinstance(n.get('labels', []), list) else []
                name = n.get('name') or n.get('title') or n.get('id') or "<unknown>"
            else:
                labels = []
                name = str(n) if n is not None else "<unknown>"
            
            seeds_meta.append({"labels": labels, "name": name, "score": float(sc) if sc else 0.0})

        return {
            "success": True,
            "message": "Query processed successfully",
            "data": {
                "answer": f"Q: {q0}\n{facts}",
                "seeds": seeds_meta,
                "params": _build_params(body, embedding_available),
                "timings": {"total": perf_counter() - t0}
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Query failed: {str(e)}",
            "data": {"error_type": type(e).__name__, "error_details": str(e)}
        }


def _empty_response(question: str, body: RagBody, t0: float, embedding_available: bool) -> dict:
    """Return standardized empty response when no results found."""
    return {
        "success": True,
        "message": "No available data related to the user query",
        "data": {
            "answer": f"Q: {question}\nGraph Facts: (no results)",
            "seeds": [],
            "params": _build_params(body, embedding_available),
            "timings": {"total": perf_counter() - t0}
        }
    }


def _build_params(body: RagBody, embedding_available: bool) -> dict:
    """Build params dict for response."""
    return {
        "top_k": body.top_k,
        "hops": body.hops,
        "labels": body.labels,
        "alpha_vec": body.alpha_vec,
        "beta_kw": body.beta_kw,
        "use_mmr": body.use_mmr,
        "use_cross_doc": body.use_cross_doc,
        "used_llm": False,
        "embedding_available": bool(embedding_available),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=(ENV == "development"))