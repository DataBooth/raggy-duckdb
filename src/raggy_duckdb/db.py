from functools import cache
from pathlib import Path


@cache
def load_sql(filename: str) -> str:
    base_path = Path.cwd() / "sql"
    sql_path = base_path / filename
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()
