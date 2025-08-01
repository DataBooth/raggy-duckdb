from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from loguru import logger


class RepoIngestor:
    def __init__(
        self,
        pipeline,
        exclude_dirs: Optional[set] = None,
    ):
        self.pipeline = pipeline
        self.con = pipeline.con
        self.exclude_dirs = exclude_dirs or {
            ".venv",
            "__pycache__",
            ".git",
            "data",
            "logs",
            "cache_dir",
        }

    def discover_repos(self) -> List[Path]:
        candidates = []
        logger.info(
            f"Discovering repos under {self.pipeline.repo_root_path} in {self.pipeline.included_subdirs}..."
        )
        for subdir in self.pipeline.included_subdirs:
            search_path = self.pipeline.repo_root_path / subdir
            if not search_path.exists():
                logger.warning(f"Subdir {search_path} does not exist. Skipping.")
                continue
            found = [p for p in search_path.iterdir() if p.is_dir()]
            candidates.extend(found)
        candidates.sort()
        if self.pipeline.n_repo is not None:
            candidates = candidates[: self.pipeline.n_repo]
        logger.info(f"Discovered {len(candidates)} repositories to ingest")
        return candidates

    def collect_files(self, repo_path: Path) -> List[Tuple[str, str]]:
        collected = []
        for file_path in repo_path.rglob("*"):
            if any(part in self.exclude_dirs for part in file_path.parts):
                continue
            if file_path.is_file() and any(
                file_path.suffix.lower() == ext.lower()
                for ext in self.pipeline.file_types
            ):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel_path = str(file_path.relative_to(repo_path))
                    collected.append((rel_path, content))
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        logger.info(
            f"Collected {len(collected)} files from repo {repo_path.name} (excluded folders: {self.exclude_dirs})"
        )
        return collected

    def ingest(self):
        repos = self.discover_repos()
        logger.info("Starting ingestion of repositories")

        for repo_path in tqdm(repos, desc="Ingesting repos"):
            repo_name = repo_path.name
            files = self.collect_files(repo_path)
            for filepath, content in files:
                chunks = self.pipeline._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    self.con.execute(
                        """
                        INSERT INTO documents (repo, filepath, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?, NULL)
                        """,
                        (repo_name, filepath, idx, chunk),
                    )
            logger.info(f"Ingested repo {repo_name} with {len(files)} files")
        logger.info("Completed ingestion of all repositories")
