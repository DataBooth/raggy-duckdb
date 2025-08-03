CREATE TABLE IF NOT EXISTS ingested_files (
    repo TEXT,
    filepath TEXT,
    file_hash TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    embedding_model TEXT,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (repo, filepath)
);

CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
    repo TEXT,
    filepath TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    chunk_hash TEXT PRIMARY KEY,
    embedding BLOB,
    embedding_model TEXT
);
