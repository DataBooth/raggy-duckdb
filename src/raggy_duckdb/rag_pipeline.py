import os
from typing import List, Optional, Tuple
from pathlib import Path

import diskcache as dc
import duckdb
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from together import Together

from raggy_duckdb.config import load_config
from raggy_duckdb.log import setup_logger

load_dotenv()
cache = dc.Cache("./cache_dir")


class TogetherAIEmbedder:
    def __init__(self, api_key: str, model: str):
        self.client = Together(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        results = []
        texts_to_call = []
        indices_to_call = []

        for i, text in enumerate(texts):
            cached_emb = cache.get(text)
            if cached_emb is not None:
                results.append(cached_emb)
            else:
                results.append(None)  # placeholder
                texts_to_call.append(text)
                indices_to_call.append(i)

        if texts_to_call:
            api_embeddings = self.client.embeddings.create(
                model=self.model, input=texts_to_call
            )
            api_embs = [np.array(e, dtype=np.float32) for e in api_embeddings.data]
            for idx, emb in zip(indices_to_call, api_embs):
                cache[texts_to_call[idx]] = emb
                results[idx] = emb
        return results


class TogetherAILLM:
    def __init__(self, api_key: str, model: str):
        self.client = Together(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        cached_response = cache.get(prompt)
        if cached_response is not None:
            logger.info("Using cached LLM response")
            return cached_response

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        cache[prompt] = text
        return text


class RAGPipeline:
    def __init__(
        self,
        duckdb_path: str,
        embedding_model: str,
        llm_model: str,
        repo_root_path: str,
        included_subdirs: List[str],
        file_types: List[str],
        n_repo: Optional[int] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ):
        """
        Initialize RAGPipeline with config and establish DuckDB connection and AI clients.

        Args:
            duckdb_path (str): Path to DuckDB database file.
            embedder_api_key (str): API key for Together.ai embedder.
            embedding_model (str): Embedding model name for Together.ai.
            llm_model (str): LLM model name for Together.ai.
            repo_root_path (str): Root local directory containing cloned repos.
            included_subdirs (List[str]): List of subdirectories to scan for repos.
            file_types (List[str]): Allowed file extensions to ingest (e.g. ['.py', '.md']).
            n_repo (Optional[int]): Number of repos to ingest (None = all).
            chunk_size (int): Number of characters per chunk for chunking text files.
            chunk_overlap (int): Overlap size between chunks in characters.
            top_k (int): Number of top relevant chunks to retrieve for queries.
        """
        self.con = duckdb.connect(database=duckdb_path)
        self.embedder = TogetherAIEmbedder(
            api_key=embedder_api_key, model=embedding_model
        )
        self.llm = TogetherAILLM(api_key=embedder_api_key, model=llm_model)
        self.repo_root_path = repo_root_path
        self.included_subdirs = included_subdirs
        self.file_types = file_types
        self.n_repo = n_repo
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._setup_db()
        logger.info("RAGPipeline initialized")

    def _setup_db(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT,
                filepath TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
        """)
        logger.info("Database schema ensured")

    def discover_repos(self) -> List[str]:
        candidates = []
        logger.info(
            f"Discovering repos under {self.repo_root_path} in {self.included_subdirs}..."
        )
        for subdir in self.included_subdirs:
            search_path = (self.repo_root_path / subdir).glob("*")
            found = [str(p.resolve()) for p in search_path if p.is_dir()]
            candidates.extend(found)
        if self.n_repo is not None:
            candidates = candidates[: self.n_repo]
        logger.info(f"Discovered {len(candidates)} repos to ingest")
        return candidates

    def _collect_files_from_repo(self, repo_path: str) -> List[Tuple[str, str]]:
        files_collected = []
        repo_path_obj = Path(repo_path)
        for file_path in repo_path_obj.rglob("*"):
            if file_path.is_file() and any(
                file_path.name.lower().endswith(ext.lower()) for ext in self.file_types
            ):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(file_path.relative_to(repo_path_obj))
                    files_collected.append((rel_path, content))
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        logger.info(
            f"Collected {len(files_collected)} files from repo {repo_path_obj.name}"
        )
        return files_collected

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping chunks.

        Returns:
            List of chunk strings.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


def ingest_repos(self) -> None:
    """
    Discover repos and ingest their allowed files into DuckDB as chunked documents.
    Embeddings are not computed here.
    """
    repos = self.discover_repos()
    logger.info("Starting ingestion of repos")
    for repo_path_str in repos:
        repo_path = Path(repo_path_str)
        repo_name = repo_path.name  # equivalent to os.path.basename()
        files = self._collect_files_from_repo(str(repo_path))
        for filepath, content in files:
            chunks = self._chunk_text(content)
            for idx, chunk in enumerate(chunks):
                self.con.execute(
                    "INSERT INTO documents (repo, filepath, chunk_index, content, embedding) VALUES (?, ?, ?, ?, NULL)",
                    [repo_name, filepath, idx, chunk],
                )
        logger.info(f"Ingested repo {repo_name} with {len(files)} files")
    logger.info("Completed ingestion for all repos")

    def embed_documents(self, batch_size: int = 32) -> None:
        """
        Embed documents that do not yet have embeddings, batch updating the database.
        """
        logger.info("Starting embedding of documents missing embeddings")
        while True:
            rows = self.con.execute(
                "SELECT id, content FROM documents WHERE embedding IS NULL LIMIT ?",
                [batch_size],
            ).fetchall()
            if not rows:
                logger.info("Embedding complete for all documents")
                break
            ids, texts = zip(*rows)
            embeddings = self.embedder.embed(list(texts))
            for doc_id, emb in zip(ids, embeddings):
                self.con.execute(
                    "UPDATE documents SET embedding = ? WHERE id = ?",
                    (emb.tobytes(), doc_id),
                )
            logger.info(f"Embedded and updated {len(ids)} documents")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def query(self, user_query: str) -> str:
        """
        Query the database for relevant documents and get an LLM-generated answer.

        Args:
            user_query (str): User input query.

        Returns:
            str: Generated answer from LLM.
        """
        logger.info(f"Processing query: {user_query}")
        query_emb = self.embedder.embed([user_query])[0]

        candidates = self.con.execute(
            "SELECT id, repo, filepath, chunk_index, content, embedding FROM documents WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for doc_id, repo, filepath, chunk_idx, content, emb_blob in candidates:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            score = self._cosine_similarity(query_emb, emb)
            scored.append(
                (
                    score,
                    {
                        "id": doc_id,
                        "repo": repo,
                        "filepath": filepath,
                        "chunk_index": chunk_idx,
                        "content": content,
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[: self.top_k]]

        context = "\n---\n".join(
            f"Repo: {doc['repo']}\nFile: {doc['filepath']}\nContent:\n{doc['content']}"
            for doc in top_docs
        )
        prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"
        answer = self.llm.generate(prompt)
        logger.info("Query complete; returning answer")
        return answer


def main():
    config = load_config(config_path="conf/config.roml")
    log_conf = config.get("logging", {})
    setup_logger(
        log_file=log_conf.get("log_file", "logs/rag_app.log"),
        log_level=log_conf.get("log_level", "INFO"),
        rotation=log_conf.get("rotation", "10 MB"),
        retention=log_conf.get("retention", "7 days"),
    )

    db_conf = config["db"]
    together_conf = config["together"]
    ingestion_conf = config["ingestion"]

    logger.info("Starting RAG pipeline - ingestion phase")

    # Instantiate pipeline
    pipeline = RAGPipeline(
        duckdb_path=db_conf["duckdb_path"],
        embedding_model=together_conf["embedding_model"],  # Used later
        llm_model=together_conf["llm_model"],  # Used later
        repo_root_path=ingestion_conf["repo_root_path"],
        included_subdirs=ingestion_conf["included_subdirs"],
        file_types=ingestion_conf["file_types"],
        n_repo=ingestion_conf.get("n_repo", None),
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
    )

    # Step 1: Discover, parse, chunk, and ingest repo files into DuckDB
    pipeline.ingest_repos()

    logger.info("Ingestion complete. Verify DuckDB contents now before proceeding.")

    # Step 2: (Commented out for now)
    # Embed documents without embeddings
    # pipeline.embed_documents()

    # Step 3: (Commented out for now)
    # Query example
    # user_query = "How does the authentication work in the databooth repo?"
    # answer = pipeline.query(user_query)
    # print(f"Query: {user_query}\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
