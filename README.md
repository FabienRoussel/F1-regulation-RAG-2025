# F1 Regulation RAG System

A Retrieval-Augmented Generation (RAG) system for Formula 1 sporting regulations, powered by PostgreSQL vector database and QWEN embeddings.

## Features

- 📄 **PDF Processing**: Extract and structure F1 regulation documents
- 🔍 **Vector Search**: Fast similarity search using pgvector in PostgreSQL
- 🤖 **QWEN Embeddings**: High-quality embeddings with GPU acceleration (CUDA/MPS)
- 🐳 **Docker Setup**: One-command PostgreSQL server with pgvector
- 📊 **Web Interface**: Adminer for database management
- 🔧 **Management Tools**: Backup, restore, and monitoring scripts

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker and Docker Compose
- uv (Python package manager)

### 2. Installation

```bash
# Clone and navigate to project
cd F1-regulation-RAG-2025

# Install Python dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env if needed (default Docker config works out of the box)
```

### 3. Start PostgreSQL Server

```bash
# Start the database server
./start_database.sh start

# Check status
./start_database.sh status
```

The script will:
- Start PostgreSQL with pgvector extension
- Initialize the database schema
- Start Adminer web interface at http://localhost:8080

### 4. Process F1 Regulations

```bash
# Set up database tables and process PDFs
python setup_database.py

# Run the main processing pipeline
python main.py
```

This will:
- Extract text from F1 regulation PDFs
- Generate embeddings using QWEN/sentence-transformers
- Store vectors in PostgreSQL with similarity indexes
- Perform example similarity search

## Database Management

Use the management script for common operations:

```bash
# Database operations
./start_database.sh start     # Start database
./start_database.sh stop      # Stop database
./start_database.sh status    # Show status
./start_database.sh logs      # View logs

# Backup and restore
./start_database.sh backup              # Create backup
./start_database.sh restore backup.sql  # Restore from backup
./start_database.sh reset               # Clean reset
```

## Database Access

- **Adminer Web UI**: http://localhost:8080
  - Server: `postgres`
  - Username: `f1_user`
  - Password: `f1_password`
  - Database: `f1_regulations`

- **Direct Connection**:
  ```bash
  # Via psql in container
  docker-compose exec postgres psql -U f1_user -d f1_regulations
  
  # Via external psql
  psql postgresql://f1_user:f1_password@localhost:5432/f1_regulations
  ```

## Project Structure

```
F1-regulation-RAG-2025/
├── src/
│   ├── database/          # PostgreSQL + pgvector operations
│   ├── embeddings/        # QWEN embedding generation
│   ├── processing/        # PDF text extraction
│   └── utils/            # Utility functions
├── data/
│   ├── pdfs/             # Source F1 regulation PDFs
│   └── processed/        # Extracted CSV data
├── docker/               # Database initialization
├── docker-compose.yml    # PostgreSQL + Adminer setup
├── main.py              # Main processing pipeline
└── setup_database.py    # Database setup script
```

## Configuration

### Environment Variables (.env)

```bash
# Database (default Docker config)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=f1_regulations
DB_USER=f1_user
DB_PASSWORD=f1_password
```

### Embedding Models

The system uses QWEN embeddings with fallback to sentence-transformers:

1. **Primary**: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
2. **Fallback**: `sentence-transformers/all-MiniLM-L6-v2`

Models automatically select optimal device (CUDA → MPS → CPU).

## Usage Examples

### Search Regulations

```python
from src.embeddings import embedding_generator

# Search for similar regulations
results = embedding_generator.search_similar_regulations(
    query="safety car procedures",
    limit=5,
    similarity_threshold=0.7
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

### Database Statistics

```python
from src.database import vector_db

stats = vector_db.get_embedding_stats()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Sources: {stats['sources']}")
```

## Advanced Features

### Custom SQL Functions

The database includes optimized functions:

- `batch_similarity_search()` - Efficient vector similarity search
- `get_vector_stats()` - Embedding statistics
- `embedding_overview` view - Source-wise analytics

### Vector Indexes

Automatically created indexes for fast similarity search:
- Cosine similarity (`vector_cosine_ops`)
- L2 distance (`vector_l2_ops`)

### Monitoring

- Health checks for database containers
- Embedding dimension validation
- Performance-optimized PostgreSQL settings

## Troubleshooting

### Database Connection Issues

```bash
# Check if containers are running
./start_database.sh status

# View logs for errors
./start_database.sh logs

# Reset if needed
./start_database.sh reset
```

### Model Loading Issues

- QWEN models require internet connection for first download
- GPU acceleration requires CUDA/MPS drivers
- Fallback models work offline after first download

### Memory Issues

- Large PDFs may require more RAM for processing
- Batch processing helps with memory management
- Docker containers use ~512MB RAM by default

## License

This project is for educational and research purposes.