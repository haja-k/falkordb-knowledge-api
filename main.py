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

EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL")
GENERATION_MODEL_URL = os.getenv("GENERATION_MODEL_URL")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
GENERATION_API_KEY = os.getenv("GENERATION_API_KEY")

# Environment
ENV = os.getenv("ENV", "development")

# Graph name from refer.py
GRAPH_NAME = "unified_knowledge_graph"


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
from graphrag_utils import (
    anchor_terms,
    extract_keywords,
    get_embedding,
    cosine,
    format_graph_facts,
    hybrid_candidates,
    expand_neighbors_by_name,
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

        # 1) embeddings (optional)
        qvec = await get_embedding(EMBEDDING_MODEL_URL, EMBEDDING_API_KEY, q0)

        # 2) hybrid candidates
        cands = hybrid_candidates(graph, q0, qvec=qvec, labels=body.labels, k_vec=max(12, body.top_k), k_kw=max(12, body.top_k), alpha_vec=body.alpha_vec, beta_kw=body.beta_kw)

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

        # format facts
        facts = format_graph_facts(expanded.get('nodes', []), expanded.get('rels', []), include_source=True)

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
                },
                "timings": {"total": perf_counter()-t0}
            }

        seeds_meta = [{"labels": list(n.get('labels', [])) if isinstance(n.get('labels', []), list) else [], "name": n.get('name') or n.get('title'), "score": sc} for n, sc in cands]

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