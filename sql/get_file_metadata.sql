SELECT file_hash, chunk_size, chunk_overlap, embedding_model
FROM ingested_files
WHERE repo = ? AND filepath = ?