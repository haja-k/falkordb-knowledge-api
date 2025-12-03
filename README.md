# FalkorDB API

This project provides an API that utilizes FalkorDB as the underlying knowledge graph database. It is designed to handle knowledge graph operations efficiently and scalably.

## Features

- Integration with FalkorDB for knowledge graph storage and querying
- RESTful API endpoints for graph operations
- Performance testing suite to ensure optimal performance
- Docker-based deployment for easy containerization and scalability

## Getting Started

### Prerequisites

- Docker
- FalkorDB instance (can be run via Docker)

### Installation

1. Clone the repository:
   ```
   git clone https://gitlab.sains.com.my/dss/data-science/sd-unit/ai-vie/scspedia/falkordb-api.git
   cd falkordb-api
   ```

2. Build the Docker image:
   ```
   docker build -t falkordb-api .
   ```

3. Run the application with Docker:
   ```
   docker run -p 8000:8000 falkordb-api
   ```

### Usage

Once the API is running, you can interact with it via HTTP requests. Refer to the API documentation for detailed endpoints.

Example:
```
curl http://localhost:8000/api/graph/query
```

## Performance Testing

The project includes a performance testing suite to evaluate the API's efficiency under various loads. To run the tests:

1. Ensure the API is running.
2. Execute the performance tests:
   ```
   # Command to run performance tests (update as per your setup)
   python tests/performance_test.py
   ```

## Deployment

The application is containerized using Docker for easy deployment. Use the provided Dockerfile to build and run the application in any environment that supports Docker.

For production deployment, consider using orchestration tools like Kubernetes or Docker Compose for scaling and management.

## Contributing

Contributions are welcome! Please follow the standard Git workflow: fork, branch, commit, and create a merge request.

## License

This project is licensed under [Your License Here].

## Project Status

Active development.
