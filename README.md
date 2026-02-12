# Scholar's Vault

**Agentic RAG System for Academic Research**

Scholar's Vault is a local-first knowledge base system using Agentic RAG to retrieve quotes, summaries, and insights from your personal library of PDFs, documents, and ebooks.

## Features

- 🚀 **Local-First**: All data stays on your machine
- 🔒 **Privacy-Focused**: No data sent to external services (except optional cloud inference)
- ⚡ **GPU Accelerated**: CUDA support for fast embedding generation
- 📚 **Multi-Format**: Support for PDF, DOCX, EPUB, TXT, and Markdown
- 🔍 **Semantic Search**: Find relevant information using natural language queries
- 🎯 **Production Ready**: CLI interface with rich terminal output

## Quick Start

### Prerequisites

- Python 3.10 or higher
- (Optional) CUDA-compatible GPU for accelerated processing

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Scholar-s-Vault
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the package**
```bash
pip install -e .
```

### Basic Usage

**1. Ingest documents**
```bash
# Ingest a single file
scholars-vault ingest path/to/document.pdf

# Ingest a directory
scholars-vault ingest path/to/documents/

# Ingest directory without recursion
scholars-vault ingest path/to/documents/ --no-recursive
```

**2. Search for information**
```bash
scholars-vault search "What is quantum mechanics?"
```

**3. View statistics**
```bash
scholars-vault stats
```

**4. Clear the database**
```bash
scholars-vault clear --yes
```

## Configuration

Edit `.config/config.yaml` to customize:

- **Chunking strategy**: Adjust chunk size and overlap
- **Embedding model**: Choose different models
- **GPU/CPU**: Toggle CUDA acceleration
- **Supported formats**: Add/remove file types

Example configuration:
```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model_name: "BAAI/bge-small-en-v1.5"
  device: "cuda"  # or "cpu"
  batch_size: 32

vector_db:
  collection_name: "scholars_vault"
  storage_path: "data/vector_db"
```

## Project Structure

```
scholars-vault/
├── .config/          # Configuration files
├── data/             # User documents and vector database
├── src/
│   ├── loaders/      # Document loaders (PDF, DOCX, etc.)
│   ├── core/         # Chunking and embedding logic
│   ├── storage/      # Vector database interface
│   └── cli.py        # Command-line interface
├── tests/            # Test suite
└── setup.py          # Package configuration
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Supported File Formats

- **PDF** (.pdf) - High-quality conversion with table preservation
- **Microsoft Word** (.docx)
- **EPUB** (.epub) - Ebook format
- **Plain Text** (.txt)
- **Markdown** (.md)

## Roadmap

This is **Phase 01: Foundational Ingestion**. Future phases include:

- **Phase 02**: Hybrid Inference (Local + Cloud API)
- **Phase 03**: Agentic RAG Core (LangGraph workflow)
- **Phase 04**: Web UI (Streamlit/FastAPI)
- **Phase 05**: Optimization (Context compression, re-ranking)

## Troubleshooting

### CUDA Not Available

If you see "CUDA not available", the system will automatically fall back to CPU. To use GPU:

1. Install CUDA Toolkit 12.0+
2. Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Embedding Model Download

On first run, the embedding model will be downloaded automatically. This may take a few minutes depending on your internet connection.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
