from pathlib import Path
from typing import List, Optional, Tuple

import diskcache as dc
import duckdb
import numpy as np
from loguru import logger
from repo_ingest import RepoIngestor
from tqdm import tqdm


class RAGPipeline:
    def __init__(
        self,
        repo_root_path: str,
        duckdb_db_path: str,
        embedder,
        llm,
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
            llm: LLM backend instance with generate(prompt: str) or chat(messages) method.
            cache_dir: Directory path for diskcache.
            included_subdirs: List of subdirectories under root to scan (default all).
            file_types: List of allowed file extensions to ingest (default .py, .md, .txt).
            n_repo: Optional limit on number of repos to ingest.
            chunk_size: Number of characters per chunk.
            chunk_overlap: Overlap characters between chunks.
            top_k: Number of top documents to retrieve in queries.
        """
        self.repo_root_path = Path(repo_root_path)
        self.duckdb_path = duckdb_db_path
        self.embedder = embedder
        self.llm = llm
        self.cache_dir = cache_dir
        self.included_subdirs = included_subdirs or [""]
        self.file_types = file_types or [".py", ".md", ".txt"]
        self.n_repo = n_repo
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.con = duckdb.connect(database=str(self.duckdb_path))
        self._prepare_tables()

        self.cache = dc.Cache(self.cache_dir)

        logger.info("RAGPipeline initialized")

    def _prepare_tables(self) -> None:
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                repo TEXT NOT NULL,
                filepath TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB
            )
            """
        )
        self.con.commit()
        logger.info("Database schema and sequence ensured")

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest_repos(self) -> None:
        ingestor = RepoIngestor(self)
        ingestor.ingest()

    def embed_documents(self, batch_size: int = 32) -> None:
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
        Query database for relevant docs, generate LLM answer.

        Args:
            user_query: The question string.

        Returns:
            The LLM-generated answer.
        """
        logger.info(f"Processing query: {user_query}")
        top_docs = self.query_documents(user_query, self.top_k)

        context = "\n---\n".join(
            f"Repo: {doc[1]}\nFile: {doc[2]}\nContent:\n{doc[4]}" for doc in top_docs
        )
        prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"

        # Assumes your LLM adapter has a method `generate(prompt:str) -> str`
        answer = self.llm.chat(prompt)
        logger.info("Query complete; returning answer")
        return answer

    def close(self):
        self.con.close()
        self.cache.close()
        logger.info("Closed RAGPipeline DB connection and cache")
