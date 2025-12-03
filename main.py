from fastapi import FastAPI, HTTPException
from falkordb import FalkorDB
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FalkorDB Knowledge API", version="1.0.0")

# Load environment variables
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6400))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL")
GENERATION_MODEL_URL = os.getenv("GENERATION_MODEL_URL")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
GENERATION_API_KEY = os.getenv("GENERATION_API_KEY")

# Graph name from refer.py
GRAPH_NAME = "unified_knowledge_graph"

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
            "graph": GRAPH_NAME,
            "node_count": node_count,
            "relationship_count": rel_count,
            "labels": labels,
            "message": "Database summary retrieved successfully"
        }
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to retrieve database summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)