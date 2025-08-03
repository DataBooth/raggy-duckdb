import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import diskcache as dc
import duckdb
import numpy as np
from db import load_sql
from loguru import logger
from tqdm import tqdm
from repo_ingest import RepoIngestor


class RAGPipeline:
    def __init__(
        self,
        repo_root_path: str,
        duckdb_db_path: str,
        embedder,
        llm,
        embedding_model: str,
        cache_dir: str = "./cache_dir",
        included_subdirs: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        n_repo: Optional[int] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        """
        Initialise RAGPipeline with configuration and establish DB connection.

        Args:
            repo_root_path: Root directory containing cloned repos.
            duckdb_db_path: Path to DuckDB database file.
            embedder: Embedding backend instance with embed(texts: List[str]) method.
            llm: LLM backend instance with chat(messages: List[dict]) method.
            embedding_model: Name of the embedding model used.
            cache_dir: Directory path for diskcache.
            included_subdirs: List of subdirectories under root to scan.
            file_types: List of allowed file extensions to ingest.
            n_repo: Optional limit on number of repos to ingest.
            chunk_size: Number of characters per chunk.
            chunk_overlap: Overlap characters between chunks.
            top_k: Number of top documents to retrieve in queries.
        """
        self.repo_root_path = Path(repo_root_path)
        self.duckdb_path = duckdb_db_path
        self.embedder = embedder
        self.llm = llm
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.included_subdirs = included_subdirs or [""]
        self.file_types = file_types or [".py", ".md", ".txt"]
        self.n_repo = n_repo
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.con = duckdb.connect(database=str(self.duckdb_path))
        self._create_tables_from_file("create_tables.sql")

        self.cache = dc.Cache(self.cache_dir)

        logger.info("RAGPipeline initialised")

    def _create_tables_from_file(self, sql_filename: str = "create_tables.sql") -> None:
        """
        Load and execute SQL statements from a file to create database tables.
        """
        sql_script = load_sql(sql_filename)
        self.con.execute(sql_script)
        self.con.commit()
        logger.info(f"Database tables created or already exist via {sql_filename}")

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping chunks of chunk_size."""
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest_repos(self) -> None:
        """
        Ingest repositories: scan repos, read files, chunk texts, store in DB.

        Note: This example uses a simple ingestion loop. You may replace this
        with RepoIngestor or your own ingestion logic.
        """
        repos = self.scan_repos()
        for repo_path in tqdm(repos, desc="Ingesting repos"):
            repo_name = repo_path.name
            files = self.collect_files(repo_path)
            for filepath, content in files:
                chunks = self._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    self.con.execute(
                        """
                        INSERT INTO documents (repo, filepath, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?, NULL)
                        """,
                        (repo_name, filepath, idx, chunk),
                    )
                # Update ingestion metadata
                path_obj = repo_path / filepath
                file_hash = self.file_hash(path_obj)
                self.update_ingestion_metadata(repo_name, filepath, file_hash)
            logger.info(f"Ingested repo {repo_name} with {len(files)} files")
        self.con.commit()
        logger.info("Completed ingestion of all repositories")

    def embed_documents(self, batch_size: int = 32) -> None:
        """
        Embed documents in batches where embedding is NULL.
        """
        docs_to_embed = self.con.execute(
            "SELECT id, content FROM documents WHERE embedding IS NULL"
        ).fetchall()

        if not docs_to_embed:
            logger.info("No documents require embedding.")
            return

        for i in tqdm(
            range(0, len(docs_to_embed), batch_size), desc="Embedding documents"
        ):
            batch = docs_to_embed[i : i + batch_size]
            ids = [doc[0] for doc in batch]
            texts = [doc[1] for doc in batch]

            embeddings = self.embedder.embed(texts)

            for doc_id, emb in zip(ids, embeddings):
                emb_blob = emb.tobytes()
                self.con.execute(
                    "UPDATE documents SET embedding = ? WHERE id = ?",
                    (emb_blob, doc_id),
                )
            self.con.commit()
        logger.info("Completed embedding of documents")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def query_documents(self, query: str, top_k: Optional[int] = None) -> List[Tuple]:
        """
        Query documents by cosine similarity of embeddings.

        Returns list of tuples: (score, repo, filepath, chunk_index, content).
        """
        top_k = top_k or self.top_k
        query_emb = self.embedder.embed([query])[0]
        rows = self.con.execute(
            "SELECT repo, filepath, chunk_index, content, embedding FROM documents WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for repo, filepath, chunk_idx, content, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            score = self._cosine_similarity(query_emb, emb)
            scored.append((score, repo, filepath, chunk_idx, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def query(self, user_query: str) -> str:
        """
        Query database for relevant docs and generate LLM answer.

        Args:
            user_query: The question string.

        Returns:
            The LLM-generated answer string.
        """
        logger.info(f"Processing query: {user_query}")
        top_docs = self.query_documents(user_query, self.top_k)

        context = "\n---\n".join(
            f"Repo: {doc[1]}\nFile: {doc[2]}\nContent:\n{doc[4]}" for doc in top_docs
        )

        prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"

        # The LLM chat method expects list of messages with roles
        messages = [{"role": "user", "content": prompt}]
        answer = self.llm.chat(messages)

        logger.info("Query complete; returning answer")
        return answer

    def file_hash(self, filepath: Path, block_size: int = 65536) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            filepath: Path to the file.
            block_size: Read buffer size.

        Returns:
            Hexadecimal string of SHA256 hash.
        """
        hasher = hashlib.sha256()
        with filepath.open("rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def get_file_metadata(self, repo: str, filepath: str):
        """
        Retrieve file ingestion metadata from DB.

        Returns tuple (file_hash, chunk_size, chunk_overlap, embedding_model) or None.
        """
        sql = load_sql("get_file_metadata.sql")
        row = self.con.execute(sql, (repo, filepath)).fetchone()
        return row

    def needs_ingestion(self, repo: str, filepath: str, current_hash: str) -> bool:
        """
        Determine if a file needs ingestion based on hash and chunk config.

        Returns True if ingestion needed.
        """
        meta = self.get_file_metadata(repo, filepath)
        if meta is None:
            return True
        stored_hash, stored_chunk_size, stored_chunk_overlap, _ = meta
        if stored_hash != current_hash:
            return True
        if (
            stored_chunk_size != self.chunk_size
            or stored_chunk_overlap != self.chunk_overlap
        ):
            return True
        return False

    def update_ingestion_metadata(self, repo: str, filepath: str, file_hash: str):
        """
        Update ingestion metadata in DB after ingestion.

        Args:
            repo: Repository name.
            filepath: File path relative to repo.
            file_hash: SHA256 hash of file content.
        """
        sql = load_sql("insert_or_update_ingested_file.sql")
        now = datetime.utcnow()
        self.con.execute(
            sql,
            (
                repo,
                filepath,
                file_hash,
                self.chunk_size,
                self.chunk_overlap,
                self.embedding_model,
                now,
            ),
        )
        self.con.commit()

    def needs_embedding(self) -> bool:
        """Return True if there are documents without embeddings."""
        sql = load_sql("check_documents_with_null_embedding.sql")
        count_null = self.con.execute(sql).fetchone()[0]
        return count_null > 0

    def close(self):
        """Close database connection and cache."""
        self.con.close()
        self.cache.close()
        logger.info("Closed RAGPipeline DB connection and cache")

    # Placeholder methods you must implement according to your repo structure and requirements

    def scan_repos(self) -> List[Path]:
        """
        Scan and return list of repo root paths to ingest.

        Replace or implement based on your repo layout.
        """
        ingestor = RepoIngestor(self)
        return ingestor.discover_repos()

    def collect_files(self, repo_path: Path) -> List[Tuple[str, str]]:
        """
        Collect files from the given repository path by delegating to RepoIngestor.

        Args:
            repo_path: Path object to the repository root.

        Returns:
            List of tuples (relative filepath, file content).
        """
        ingestor = RepoIngestor(self)
        return ingestor.collect_files(repo_path)
