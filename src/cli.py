"""
Command-line interface for Scholar's Vault

Provides CLI commands for document ingestion, search, and management.
"""
import sys
from pathlib import Path
from typing import Optional
import yaml
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loaders.document_loader import DocumentLoader, DocumentLoadError
from core.chunking import TextChunker
from core.embeddings import EmbeddingGenerator
from storage.vector_store import VectorStore

app = typer.Typer(
    name="scholars-vault",
    help="Scholar's Vault - Agentic RAG for Academic Research",
    add_completion=False
)

console = Console()


def load_config(config_path: str = ".config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def setup_logging(config: dict):
    """Configure logging based on config."""
    log_config = config.get('logging', {})
    level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'logs/scholars_vault.log')
    
    # Remove default logger
    logger.remove()
    
    # Add console logger if enabled
    if log_config.get('console', True):
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
    
    # Add file logger
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Search subdirectories"),
    config_path: str = typer.Option(".config/config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Ingest documents into Scholar's Vault.
    
    Loads documents, chunks them, generates embeddings, and stores in vector database.
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    rprint(f"[bold cyan]Scholar's Vault - Document Ingestion[/bold cyan]\n")
    
    try:
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Load configuration
            vector_config = config.get('vector_db', {})
            embedding_config = config.get('embeddings', {})
            chunking_config = config.get('chunking', {})
            
            # Initialize components
            task1 = progress.add_task("Initializing components...", total=None)
            
            loader = DocumentLoader(
                supported_formats=config.get('supported_formats', None)
            )
            
            chunker = TextChunker(
                strategy=chunking_config.get('strategy', 'recursive'),
                chunk_size=chunking_config.get('chunk_size', 1000),
                chunk_overlap=chunking_config.get('chunk_overlap', 200),
                separators=chunking_config.get('separators', None)
            )
            
            embedder = EmbeddingGenerator(
                model_name=embedding_config.get('model_name', 'BAAI/bge-small-en-v1.5'),
                device=embedding_config.get('device', 'cuda'),
                batch_size=embedding_config.get('batch_size', 32)
            )
            
            store = VectorStore(
                collection_name=vector_config.get('collection_name', 'scholars_vault'),
                storage_path=vector_config.get('storage_path', 'data/vector_db'),
                embedding_dimension=embedder.dimension
            )
            
            progress.update(task1, completed=True)
            
            # Load documents
            task2 = progress.add_task("Loading documents...", total=None)
            
            path_obj = Path(path)
            if path_obj.is_file():
                documents = [loader.load_document(path)]
            elif path_obj.is_dir():
                documents = loader.load_directory(path, recursive=recursive)
            else:
                raise DocumentLoadError(f"Invalid path: {path}")
            
            progress.update(task2, completed=True)
            rprint(f"✓ Loaded [bold]{len(documents)}[/bold] documents")
            
            # Chunk documents
            task3 = progress.add_task("Chunking documents...", total=None)
            chunks = chunker.chunk_documents(documents)
            progress.update(task3, completed=True)
            rprint(f"✓ Created [bold]{len(chunks)}[/bold] chunks")
            
            # Generate embeddings
            task4 = progress.add_task("Generating embeddings...", total=None)
            chunks_with_embeddings = embedder.embed_chunks(chunks)
            progress.update(task4, completed=True)
            rprint(f"✓ Generated [bold]{len(chunks_with_embeddings)}[/bold] embeddings")
            
            # Store in vector database
            task5 = progress.add_task("Storing in vector database...", total=None)
            count = store.add_chunks(chunks_with_embeddings)
            progress.update(task5, completed=True)
            rprint(f"✓ Stored [bold]{count}[/bold] chunks in vector database")
        
        rprint(f"\n[bold green]✓ Ingestion complete![/bold green]")
        
    except DocumentLoadError as e:
        rprint(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Ingestion failed")
        rprint(f"[bold red]Ingestion failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Minimum similarity score (0-1)"),
    config_path: str = typer.Option(".config/config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Search the knowledge base for relevant information.
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    rprint(f"[bold cyan]Scholar's Vault - Search[/bold cyan]\n")
    rprint(f"Query: [italic]{query}[/italic]\n")
    
    try:
        # Initialize components
        vector_config = config.get('vector_db', {})
        embedding_config = config.get('embeddings', {})
        
        embedder = EmbeddingGenerator(
            model_name=embedding_config.get('model_name', 'BAAI/bge-small-en-v1.5'),
            device=embedding_config.get('device', 'cuda'),
            batch_size=embedding_config.get('batch_size', 32)
        )
        
        store = VectorStore(
            collection_name=vector_config.get('collection_name', 'scholars_vault'),
            storage_path=vector_config.get('storage_path', 'data/vector_db'),
            embedding_dimension=embedder.dimension
        )
        
        # Generate query embedding
        query_embedding = embedder.embed_single(query)
        
        # Search
        results = store.search(query_embedding, limit=limit, score_threshold=threshold)
        
        if not results:
            rprint("[yellow]No results found.[/yellow]")
            return
        
        # Display results
        for i, result in enumerate(results, 1):
            rprint(f"\n[bold]Result {i}[/bold] (Score: {result['score']:.4f})")
            rprint(f"[dim]Source: {result.get('source', 'Unknown')}[/dim]")
            rprint(f"{result['text'][:500]}...")
            rprint("[dim]" + "─" * 80 + "[/dim]")
        
    except Exception as e:
        logger.exception("Search failed")
        rprint(f"[bold red]Search failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    config_path: str = typer.Option(".config/config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Show collection statistics.
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    rprint(f"[bold cyan]Scholar's Vault - Statistics[/bold cyan]\n")
    
    try:
        vector_config = config.get('vector_db', {})
        embedding_config = config.get('embeddings', {})
        
        # Get embedding dimension
        embedder = EmbeddingGenerator(
            model_name=embedding_config.get('model_name', 'BAAI/bge-small-en-v1.5'),
            device=embedding_config.get('device', 'cuda')
        )
        
        store = VectorStore(
            collection_name=vector_config.get('collection_name', 'scholars_vault'),
            storage_path=vector_config.get('storage_path', 'data/vector_db'),
            embedding_dimension=embedder.dimension
        )
        
        stats = store.get_stats()
        
        # Create table
        table = Table(title="Vector Store Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Collection Name", stats.get('collection_name', 'N/A'))
        table.add_row("Total Chunks", str(stats.get('vectors_count', 0)))
        table.add_row("Indexed Chunks", str(stats.get('indexed_vectors_count', 0)))
        table.add_row("Status", stats.get('status', 'unknown'))
        
        console.print(table)
        
    except Exception as e:
        logger.exception("Failed to get stats")
        rprint(f"[bold red]Failed to get stats:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    config_path: str = typer.Option(".config/config.yaml", "--config", "-c", help="Path to config file")
):
    """
    Clear all data from the vector database.
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    if not confirm:
        response = typer.confirm("Are you sure you want to clear all data?")
        if not response:
            rprint("[yellow]Operation cancelled.[/yellow]")
            return
    
    try:
        vector_config = config.get('vector_db', {})
        embedding_config = config.get('embeddings', {})
        
        embedder = EmbeddingGenerator(
            model_name=embedding_config.get('model_name', 'BAAI/bge-small-en-v1.5'),
            device=embedding_config.get('device', 'cuda')
        )
        
        store = VectorStore(
            collection_name=vector_config.get('collection_name', 'scholars_vault'),
            storage_path=vector_config.get('storage_path', 'data/vector_db'),
            embedding_dimension=embedder.dimension
        )
        
        store.clear()
        rprint("[bold green]✓ Vector database cleared![/bold green]")
        
    except Exception as e:
        logger.exception("Failed to clear database")
        rprint(f"[bold red]Failed to clear database:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
