import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "database" / "zepto_books.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_schema(conn):
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
    DROP TABLE IF EXISTS books;
    DROP TABLE IF EXISTS categories;

    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL CHECK (in_stock IN (0,1)),
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)
    conn.commit()


def load_data(conn, df):
    categories = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
        .rename(columns={"category": "category_name"})
    )

    categories.to_sql(
        "categories",
        conn,
        if_exists="append",
        index=False
    )

    category_map = pd.read_sql_query(
        "SELECT category_id, category_name FROM categories",
        conn
    )

    books = df.merge(
        category_map,
        left_on="category",
        right_on="category_name",
        how="left",
    )[[
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id"
    ]]

    books["in_stock"] = books["in_stock"].astype(int)

    books.to_sql(
        "books",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()


def build_database(df):
    with get_connection() as conn:
        create_schema(conn)
        load_data(conn, df)
    return DB_PATH
