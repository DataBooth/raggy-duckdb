import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from tqdm import tqdm


class RepoIngestor:
    def __init__(
        self,
        pipeline,
        exclude_dirs: Optional[set] = None,
    ):
        """
        Initialise RepoIngestor with a reference to the pipeline and excluded directories.

        Args:
            pipeline: An instance of the RAGPipeline or compatible interface.
            exclude_dirs: Set of directory names to exclude during file collection.
        """
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
        """
        Discover repository directories under root path and included subdirectories.

        Returns:
            List of Path objects pointing to discovered repositories.
        """
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

    def _read_file(self, file_path: Path) -> Tuple[Optional[str], str]:
        """
        Helper function to read file content safely.

        Args:
            file_path: Path object to the file.

        Returns:
            A tuple of (content or None if error, relative file path as string).
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return content, str(file_path)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {type(e).__name__}: {e}")
            return None, str(file_path)

    def collect_files(self, repo_path: Path) -> List[Tuple[str, str]]:
        """
        Collect all matching files from a repo, excluding unwanted directories.

        Args:
            repo_path: Path to the repository root.

        Returns:
            List of tuples: (relative filepath to repo, file content)
        """
        collected = []
        all_files = [fp for fp in repo_path.rglob("*") if fp.is_file()]

        # Parallelise reading here
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for file_path in all_files:
                # Exclude folders in path
                if any(part in self.exclude_dirs for part in file_path.parts):
                    continue
                # Exclude files by suffix
                if not any(
                    file_path.suffix.lower() == ext.lower()
                    for ext in self.pipeline.file_types
                ):
                    continue
                futures.append(executor.submit(self._read_file, file_path))

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Reading files in {repo_path.name}",
            ):
                content, file_path_str = future.result()
                if content is not None:
                    # Get relative path relative to repo root
                    rel_path = Path(file_path_str).relative_to(repo_path)
                    collected.append((str(rel_path), content))

        logger.info(
            f"Collected {len(collected)} files from repo {repo_path.name} (excluded folders: {self.exclude_dirs})"
        )
        return collected

    def ingest(self):
        """
        Discover repos and ingest their files into the pipeline's database.

        This method chunks file contents and inserts into the documents table.
        """
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
        self.con.commit()
        logger.info("Completed ingestion of all repositories")
