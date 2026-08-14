from pathlib import Path
import sqlite3

from scrape_books import scrape_books
from clean_books import clean_books
from database import build_database
from queries import run_sql_queries, validate_with_pandas

BASE_DIR = Path(__file__).resolve().parent
CLEANED_CSV = BASE_DIR / "data" / "books_cleaned.csv"


def main():
    print("\n[1/5] Scraping books.toscrape.com...")
    raw_df = scrape_books()
    print(f"Scraped {len(raw_df)} books across {raw_df['category'].nunique()} categories.")

    print("\n[2/5] Cleaning and converting...")
    cleaned_df = clean_books(raw_df)

    if len(cleaned_df) < 60 or cleaned_df["category"].nunique() < 3:
        raise RuntimeError("Acceptance criterion failed: need >=60 books and >=3 categories.")

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved: {CLEANED_CSV}")

    print("\n[3/5] Creating normalized SQLite database...")
    db_path = build_database(cleaned_df)
    print(f"Saved: {db_path}")

    print("\n[4/5] Executing SQL queries...")
    with sqlite3.connect(db_path) as conn:
        run_sql_queries(conn)
        print("Saved: outputs/sql_query_results.txt")

        print("\n[5/5] Comparing SQL JOIN with pandas.merge...")
        equivalent = validate_with_pandas(conn, cleaned_df)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    print(f"SQL JOIN == pandas.merge: {equivalent}")
    print("Saved: outputs/pandas_comparison.txt")


if __name__ == "__main__":
    main()
