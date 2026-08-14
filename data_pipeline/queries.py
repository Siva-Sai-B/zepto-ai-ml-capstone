import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "zepto_books.db"
SQL_OUTPUT = BASE_DIR / "outputs" / "sql_query_results.txt"
PANDAS_OUTPUT = BASE_DIR / "outputs" / "pandas_comparison.txt"

QUERIES = [
    ("Query 1 - SELECT + WHERE", """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, title
        LIMIT 10;
    """),
    ("Query 2 - ORDER BY + LIMIT", """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """),
    ("Query 3 - DISTINCT", """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """),
    ("Query 4 - BETWEEN", """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """),
    ("Query 5 - IN", """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """),
    ("Query 6 - JOIN", """
        SELECT b.title, b.price_gbp, b.price_inr, b.rating,
               b.in_stock, c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.title
        LIMIT 10;
    """),
]


def run_sql_queries(conn):
    SQL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = []

    for name, query in QUERIES:
        result = pd.read_sql_query(query, conn)
        report += [
            "=" * 80,
            name,
            "=" * 80,
            "SQL:",
            query.strip(),
            "",
            "OUTPUT:",
            result.to_string(index=False),
            "",
        ]

    SQL_OUTPUT.write_text("\n".join(report), encoding="utf-8")
    return pd.read_sql_query(QUERIES[-1][1], conn)


def validate_with_pandas(conn, cleaned_df):
    sql_join = pd.read_sql_query(QUERIES[-1][1], conn)

    category_df = (
        cleaned_df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )
    category_df["category_id"] = range(1, len(category_df) + 1)

    pandas_join = pd.merge(
        cleaned_df,
        category_df,
        on="category",
        how="inner",
    )[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]

    pandas_join = pandas_join.rename(columns={"category": "category_name"})
    pandas_join = pandas_join.sort_values(
        ["rating", "title"], ascending=[False, True]
    ).head(10).reset_index(drop=True)

    sql_normalized = sql_join[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_name"]
    ].copy()
    sql_normalized["in_stock"] = sql_normalized["in_stock"].astype(bool)

    pandas_normalized = pandas_join.copy()
    pandas_normalized["in_stock"] = pandas_normalized["in_stock"].astype(bool)

    equivalent = sql_normalized.equals(pandas_normalized)

    report = [
        "=" * 80,
        "SQL JOIN RESULT (pd.read_sql_query)",
        "=" * 80,
        sql_normalized.to_string(index=False),
        "",
        "=" * 80,
        "PANDAS MERGE RESULT (pd.merge)",
        "=" * 80,
        pandas_normalized.to_string(index=False),
        "",
        f"Results equivalent: {equivalent}",
    ]
    PANDAS_OUTPUT.write_text("\n".join(report), encoding="utf-8")

    if not equivalent:
        raise AssertionError("SQL JOIN and pandas.merge results are not equivalent.")

    return equivalent
