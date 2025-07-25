import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""Check the data loaded from repos into DuckDB""")
    return


@app.cell
def _():
    import marimo as mo
    import duckdb
    return duckdb, mo


@app.cell
def _():
    queries = {
            "Total chunks": "SELECT COUNT(*) AS total_chunks FROM documents;",
            "Number of distinct repos": "SELECT COUNT(DISTINCT repo) AS num_repos FROM documents;",
            "Number of distinct files": "SELECT COUNT(DISTINCT filepath) AS num_files FROM documents;",
            "Chunks per repo": """
                SELECT repo, COUNT(*) AS chunk_count
                FROM documents
                GROUP BY repo
                ORDER BY chunk_count DESC;
            """,
            "Chunks per file in 'databooth' repo (top 10)": """
                SELECT filepath, COUNT(*) AS chunk_count
                FROM documents
                WHERE repo = 'databooth'
                GROUP BY filepath
                ORDER BY chunk_count DESC
                LIMIT 10;
            """,
            "Sample chunks from 'databooth'": """
                SELECT repo, filepath, chunk_index, substr(content, 1, 200) AS snippet
                FROM documents
                WHERE repo = 'databooth'
                ORDER BY repo, filepath, chunk_index
                LIMIT 10;
            """,
            "Count chunks with embeddings": """
                SELECT COUNT(*) AS embedded_chunks FROM documents WHERE embedding IS NOT NULL;
            """,
            "Average embedding size in bytes": """
                SELECT AVG(LENGTH(embedding)) AS avg_embedding_size_bytes
                FROM documents
                WHERE embedding IS NOT NULL;
            """,
            # Example query to show chunks for a specific file - update filepath accordingly
            "Chunks for a specific file 'some_path/example.py'": """
                SELECT chunk_index, substr(content, 1, 300) AS chunk_start
                FROM documents
                WHERE repo = 'databooth' AND filepath = 'some_path/example.py'
                ORDER BY chunk_index;
            """
        }
    return (queries,)


@app.cell
def _(duckdb, queries):
    def run_queries(duckdb_path: str):
        con = duckdb.connect(duckdb_path)

        for description, query in queries.items():
            print(f"=== {description} ===")
            try:
                rows = con.execute(query).fetchall()
                for row in rows:
                    print(row)
            except Exception as e:
                print(f"Error running query: {e}")
            print()

        con.close()

    return (run_queries,)


@app.cell
def _(run_queries):
    db_path = "repos.duckdb"  
    run_queries(db_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
