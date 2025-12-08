# FalkorDB API

Lightweight API that uses FalkorDB as a knowledge graph backend. This README gives a short, beginner-friendly guide to getting started, running the service, and running tests.

## Quick Start (Conda)

1. Create and activate a conda environment (Python 3.12 recommended):

   ```powershell
   conda create -n falkorpy python=3.12 -y
   conda activate falkorpy
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy the example env and edit values (use placeholders, do NOT commit secrets):

   ```powershell
   copy .env.example .env
   notepad .env    # set FALKORDB_HOST, FALKORDB_PORT, EMBEDDING_API_BASE, etc.
   ```

   - Use `FALKORDB_HOST=your_falkordb_host` and `FALKORDB_PORT=your_port` (replace with your host/port).
   - Do not include real credentials in public repos.

4. Run the API (development):

   ```powershell
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Try the Graph RAG endpoint (`/graphrag`):

   ```powershell
   curl -X POST http://127.0.0.1:8000/graphrag -H "Content-Type: application/json" -d @- <<'JSON'
   {
     "question": "What is sarawak population?",
     "top_k": 5,
     "hops": 1
   }
   JSON
   ```

## Running Tests

- Unit tests use pytest. Install test deps if needed and run:

  ```powershell
  pip install pytest
  pytest -q
  ```

- Quick endpoint test (requires API running):

  ```powershell
  python test_graphrag.py
  ```

## How Retrieval Works

The `/graphrag` endpoint uses a **hybrid retrieval** strategy combining vector and text search.

### The Full Journey (Step by Step)

```none
You ask: "What is sarawak population?"
         ↓
    ┌────────────────────────────────────────┐
    │  1. CHECK EMBEDDING SERVICE            │
    │     Is my embedding API running?       │
    │     ✓ Yes → Get question's embedding   │
    │     ✗ No  → Skip vector search         │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  2. HYBRID SEARCH (two searches!)      │
    │                                        │
    │  🔢 Vector Search (if embedding works) │
    │     "Find nodes whose embedding is     │
    │      similar to my question's numbers" │
    │                                        │
    │  📝 Text Search (always runs)          │
    │     "Find nodes where name contains    │
    │      'sarawak' or 'population'"        │
    │                                        │
    │  Then COMBINE both results with scores │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  3. IF NOTHING FOUND → FALLBACK        │
    │     Try simpler text-only search       │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  4. PICK TOP_K BEST MATCHES            │
    │     Keep only the best 10 (or whatever │
    │     top_k you set)                     │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  5. EXPAND NEIGHBORS (hops)            │
    │     For each match, find connected     │
    │     nodes (friends of friends)         │
    └────────────────────────────────────────┘
         ↓
    Return answer with all the graph facts!
```

### What Each Parameter Does

| Parameter | What it does | Example |
|-----------|--------------|---------|
| `question` | Your question | "What is sarawak population?" |
| `top_k` | How many best matches to keep | 10 = give me 10 best results |
| `hops` | How many friend-of-friend levels to explore | 1 = direct connections, 2 = friends of friends |
| `labels` | Filter to specific node types | Only look at "Person" or "Place" nodes |
| `alpha_vec` | How much to trust vector/embedding search (0-1) | 0.6 = trust vector 60% |
| `beta_kw` | How much to trust text/keyword search (0-1) | 0.4 = trust text 40% |

### The Scoring Math

```none
Final Score = (alpha × vector_score) + (beta × text_score)
            = (0.6 × embedding similarity) + (0.4 × text match score)
```

- A node with great embedding match but poor text match → still scores well
- A node with great text match but no embedding → scores okay  
- A node with both → scores best! 🏆

### Where FalkorDB Fits In

FalkorDB is the **graph database** that stores all your knowledge. Think of it as a giant web of connected information:

```none
┌─────────────────────────────────────────────────────────────────┐
│                        FalkorDB                                 │
│                   (Your Knowledge Graph)                        │
│                                                                 │
│    ┌─────────┐         ┌──────────┐         ┌─────────┐         │
│    │ Sarawak │───HAS───│Population│───IS────│2.8 mil  │         │
│    │ (Place) │         │ (Fact)   │         │(Number) │         │
│    └─────────┘         └──────────┘         └─────────┘         │
│         │                                                       │
│         │──LOCATED_IN──▶ Malaysia                               |
│         │──HAS_CAPITAL──▶ Kuching                               │
│                                                                 │
│    Each node has:                                               │
│    • name: "Sarawak"                                            │
│    • embedding: [0.12, -0.45, 0.78, ...] (1536 numbers)         │
│    • labels: ["Place", "State"]                                 │
│    • other properties...                                        │
└─────────────────────────────────────────────────────────────────┘
```

**What FalkorDB does in each step:**

| Step | What happens | FalkorDB's role |
|------|--------------|-----------------|
| Vector Search | Find similar embeddings | FalkorDB returns nodes with `embedding` property, we compute similarity |
| Text Search | Find matching names | FalkorDB runs Cypher: `WHERE toLower(n.name) CONTAINS 'sarawak'` |
| Expand Neighbors | Get connected nodes | FalkorDB traverses relationships: `MATCH (a)-[r]-(b)` |

**The connection flow:**

```none
Your API (Python/FastAPI)
         │
         │ Cypher queries
         ▼
    ┌─────────┐
    │FalkorDB │ ◄── Graph database at 172.26.93.55:6400
    │ Server  │     Stores: 30,295 nodes in 'iscs_knowledge_graph'
    └─────────┘
         │
         │ Returns nodes & relationships
         ▼
   Process results, format, return to user
```

**In simple terms:** FalkorDB is like a library where all your knowledge lives. Your API is the librarian that knows how to search that library using both "find books with similar topics" (vector) and "find books with this word in the title" (text)

## Folder overview

- `functions/` : helper modules for GraphRAG (search, embedding helpers, formatting). Exports used by `main.py`.
- `tests/` : unit tests for utility functions and endpoint behaviors.
- `scripts/` : small dev/debug scripts (ad-hoc runners). Not required for production.
- `config/` : optional configuration files and templates (env, samples).
- `Dockerfile` : container image build definition.
- `main.py` : FastAPI application and `/graphrag` endpoint.

## Docker (optional)

Build and run with Docker:

```powershell
docker build -t falkordb-api .
docker run -p 8000:8000 --env-file .env falkordb-api
```

## Notes & tips

- If the API cannot find data, check that `GRAPH_NAME` in `.env` matches the graph in FalkorDB.
- If nodes lack `name`/`title` properties the service will fall back to other fields or return `<unknown>`.
- Keep secrets out of the repository; use `.env` or a secret manager for production.
