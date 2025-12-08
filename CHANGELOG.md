# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-12-08

### Added

- **Embedding Availability Check**: Introduced `check_embedding()` helper and surfaced an `embedding_available` field in the `/graphrag` response so clients can detect when the embedding service is reachable.
- **Connectivity Diagnostics**: Added `scripts/test_embedding_connectivity.py` to probe the embedding API (auto-testing `/v1/embeddings`) for quick troubleshooting.
- **Test Runner Fallback**: Added `tests/run_tests.py` to execute the suite without pytest, enabling lightweight runs in constrained CI environments.

### Changed

- **Helper Refactor**: Consolidated GraphRAG helper logic into `functions/graphrag_utils.py`, making keyword/vector retrieval, formatting, and embedding helpers reusable.
- **Environment Templates**: Updated `.env` expectations to include Sains vLLM and Azure embedding configuration to keep local and CI environments aligned.

## [1.0.1] - 2024-12-05

### Fixed

- **FalkorDB Node Object Handling**: Added `_node_to_dict()` helper function to convert `falkordb.node.Node` objects to plain dictionaries. Previously, code was calling `.get()` on Node objects which don't support dict-style access.

- **Type-Safe Cypher Queries**: Fixed "Type mismatch: expected String or Null but was Float" error in graph queries. Some nodes have non-string values (e.g., Float) in their `name` property, causing `toLower()` to fail. Updated `_fulltext_search()` and `retrieve_from_graph()` to use type-safe Cypher:
  ```cypher
  # Before (would fail on non-string properties)
  toLower(n.name) CONTAINS 'pattern'
  
  # After (safely converts any type to string first)
  n.name IS NOT NULL AND toLower(toString(n.name)) CONTAINS 'pattern'
  ```

- **Graph Retrieval Fallback**: Added `retrieve_from_graph()` function as a fallback when `hybrid_candidates()` returns empty results. This mirrors the direct graph querying approach used in `refer.py`.

### Added

- **Integration Tests**: Added `tests/test_integration_falkordb.py` with live FalkorDB connection tests to validate:
  - Database connectivity
  - Node counting
  - Fulltext search functionality
  - Hybrid candidate retrieval
  - Graph retrieval fallback

- **Package Exports**: Updated `functions/__init__.py` to export `_fulltext_search` for testing purposes.

## [1.0.0] - 2024-12-04

### Added

- Initial release with `/graphrag` endpoint
- FalkorDB integration for knowledge graph queries
- Hybrid search combining vector and keyword approaches
- Docker support with `Dockerfile` and environment configuration
- Unit tests for utility functions
