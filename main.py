from fastapi import FastAPI, HTTPException
from falkordb import FalkorDB
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies connection to FalkorDB
    """
    try:
        # Connect to FalkorDB
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        
        # Select a graph (use default or create if needed)
        graph = db.select_graph(GRAPH_NAME)
        
        # Try to execute a simple query to check connection
        result = graph.query("RETURN 'FalkorDB is connected'")
        
        # Check if we got a result
        if result.result_set:
            return {
                "success": True,
                "message": "Connection successful",
                "data": {
                    "status": "healthy",
                    "database": "FalkorDB",
                    "host": FALKORDB_HOST,
                    "port": FALKORDB_PORT
                }
            }
        else:
            raise HTTPException(status_code=503, detail="FalkorDB query failed")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"FalkorDB connection failed: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    
    # Enable reload only in development
    reload_enabled = ENV == "development"
    
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=reload_enabled)