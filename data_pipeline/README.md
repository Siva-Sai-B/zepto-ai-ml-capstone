# Zepto Data Pipeline

## Objective

End-to-end pipeline:

**Scrape -> Clean -> GBP to INR conversion -> SQLite -> SQL queries -> Pandas validation**

Source: https://books.toscrape.com/

## Installation

From this directory:

```cmd
pip install -r requirements.txt
```

SQLite is provided by Python's standard library.

## Run

```cmd
python run_pipeline.py
```

The pipeline discovers book categories, scrapes enough paginated category pages to produce at least 60 books across at least 3 categories, cleans the data, creates the SQLite database, executes the SQL queries, and compares the SQL JOIN with `pd.merge()`.

## Fixed currency conversion

Required project baseline:

**1 GBP = 105.50 INR**

This is a fixed project-defined constant. No live currency API is used.

## Cleaning decisions

- `price` -> `price_gbp` float after removing the currency symbol.
- `One` through `Five` -> `rating` integers 1 through 5.
- Availability text -> boolean `in_stock`.
- Numeric parsing failures are median-imputed.
- Malformed product cards missing required HTML fields are skipped instead of crashing the pipeline.

## Database

`database/zepto_books.db` contains:

- `categories(category_id PRIMARY KEY, category_name UNIQUE)`
- `books(book_id PRIMARY KEY, ..., category_id FOREIGN KEY)`

## SQL

`queries.py` runs six queries covering:

- SELECT / WHERE
- ORDER BY / LIMIT
- DISTINCT
- BETWEEN
- IN
- JOIN

Results are saved to `outputs/sql_query_results.txt`.

## Pandas validation

The JOIN is read using `pd.read_sql_query()` and reproduced in memory using `pd.merge()`. The two results are compared and the comparison is saved to `outputs/pandas_comparison.txt`.

## Generated artifacts

Running `python run_pipeline.py` creates:

- `data/books_cleaned.csv`
- `database/zepto_books.db`
- `outputs/sql_query_results.txt`
- `outputs/pandas_comparison.txt`
