INSERT INTO ingested_files (repo, filepath, file_hash, chunk_size, chunk_overlap, embedding_model, ingestion_timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (repo, filepath) DO UPDATE SET
    file_hash=excluded.file_hash,
    chunk_size=excluded.chunk_size,
    chunk_overlap=excluded.chunk_overlap,
    embedding_model=excluded.embedding_model,
    ingestion_timestamp=excluded.ingestion_timestamp
