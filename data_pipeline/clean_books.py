import pandas as pd

GBP_TO_INR = 105.50

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_books(df):
    required = {"title", "price", "star_rating", "availability", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    out["price_gbp"] = pd.to_numeric(
        out["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )

    out["rating"] = (
        out["star_rating"].astype(str).str.strip().str.title().map(RATING_MAP)
    )

    availability = out["availability"].astype(str).str.strip().str.lower()
    out["in_stock"] = availability.str.contains(r"\bin stock\b", regex=True)
    out.loc[availability.str.contains("out of stock", regex=False), "in_stock"] = False

    for column in ("price_gbp", "rating"):
        if out[column].isna().all():
            raise ValueError(f"All values failed to parse for {column}.")
        out[column] = out[column].fillna(out[column].median())

    out["rating"] = out["rating"].round().astype(int)
    out["price_gbp"] = out["price_gbp"].astype(float)
    out["price_inr"] = (out["price_gbp"] * GBP_TO_INR).round(2)

    out = out[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]

    if not out["rating"].between(1, 5).all():
        raise ValueError("Rating contains values outside 1-5.")

    return out


if __name__ == "__main__":
    from scrape_books import scrape_books
    print(clean_books(scrape_books()).head(10).to_string(index=False))
