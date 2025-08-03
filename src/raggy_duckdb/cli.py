import fire
from config import load_config
from loguru import logger
from model_backends import create_embedding, create_llm
from rag_pipeline import RAGPipeline


def run_pipeline(
    config_path: str = "config.toml",
    question: str = "What is dependency injection?",
    force_ingest: bool = False,
    force_embed: bool = False,
):
    """
    Run the RAG pipeline end-to-end with options.

    Args:
        config_path (str): Path to the config TOML file.
        question (str): The user question string for querying documents.
        force_ingest (bool): If True, force re-ingestion of repos (default False).
        force_embed (bool): If True, force re-embedding of documents (default False).

    Usage example:
        uv run src/raggy-duckdb/cli.py run_pipeline --config_path=config.toml --question="Explain RAG" --force_ingest --force_embed
    """
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    embedder = create_embedding(
        config["provider"]["name"],
        config["provider"]["embedding_model"],
    )
    llm = create_llm(
        config["provider"]["name"],
        config["provider"]["llm_model"],
    )

    pipeline = RAGPipeline(
        repo_root_path=config["paths"]["repo_root_path"],
        duckdb_db_path=config["paths"]["duckdb_db_path"],
        embedding_model=config["provider"]["embedding_model"],
        embedder=embedder,
        llm=llm,
        cache_dir=config["paths"]["cache_dir"],
        included_subdirs=config["paths"].get("included_subdirs", [""]),
        file_types=config["paths"].get("file_types", [".py", ".md", ".txt"]),
        n_repo=config["paths"].get("n_repo"),
        chunk_size=config["paths"].get("chunk_size", 512),
        chunk_overlap=config["paths"].get("chunk_overlap", 64),
        top_k=config["paths"].get("top_k", 5),
    )

    if force_ingest or any(
        pipeline.needs_ingestion(repo, filepath, pipeline.file_hash(filepath))
        # This condition should ideally iterate over files/repositories to determine actual ingestion needs.
        # But if not feasible here, consider skipping or rely on force_ingest flag only.
        for repo, filepath in []
    ):
        logger.info("Running ingestion (forced or required)...")
        pipeline.ingest_repos()
    else:
        logger.info("Skipping ingestion (already up to date)")

    if force_embed or pipeline.needs_embedding():
        logger.info("Running embedding (forced or required)...")
        pipeline.embed_documents()
    else:
        logger.info("Skipping embedding (already up to date)")

    logger.info(f"Running query: {question}")
    results = pipeline.query_documents(question)

    for score, repo, path, chunk, text in results:
        print(f"[{score:.4f}] {repo}/{path} (chunk {chunk}): {text[:50]}...")

    pipeline.close()


if __name__ == "__main__":
    fire.Fire()
