from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
MIN_BOOKS = 60
MIN_CATEGORIES = 3
TIMEOUT = 20

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0)"})


def get_soup(url):
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def discover_categories():
    soup = get_soup(BASE_URL)
    categories = []
    for link in soup.select("ul.nav-list ul li a"):
        name = link.get_text(" ", strip=True)
        href = link.get("href")
        if href:
            categories.append((name, urljoin(BASE_URL, href)))
    return categories


def scrape_category(category_name, category_url):
    rows = []
    next_url = category_url

    while next_url:
        soup = get_soup(next_url)

        for article in soup.select("article.product_pod"):
            title_link = article.select_one("h3 a")
            price_el = article.select_one(".price_color")
            rating_el = article.select_one("p.star-rating")
            availability_el = article.select_one(".availability")

            if not all([title_link, price_el, rating_el, availability_el]):
                continue

            rating_classes = rating_el.get("class", [])
            rating = next(
                (x for x in rating_classes if x in {"One", "Two", "Three", "Four", "Five"}),
                None,
            )

            rows.append({
                "title": title_link.get("title") or title_link.get_text(" ", strip=True),
                "price": price_el.get_text(" ", strip=True),
                "star_rating": rating or rating_el.get_text(" ", strip=True),
                "availability": availability_el.get_text(" ", strip=True),
                "category": category_name,
            })

        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None

    return rows


def scrape_books():
    categories = discover_categories()
    if len(categories) < MIN_CATEGORIES:
        raise RuntimeError("Fewer than 3 categories were discovered.")

    all_rows = []

    for category_name, category_url in categories:
        all_rows.extend(scrape_category(category_name, category_url))
        if len(all_rows) >= MIN_BOOKS and len({r["category"] for r in all_rows}) >= MIN_CATEGORIES:
            break

    df = pd.DataFrame(all_rows)

    if len(df) < MIN_BOOKS:
        raise RuntimeError(f"Only {len(df)} books scraped; at least 60 are required.")
    if df["category"].nunique() < MIN_CATEGORIES:
        raise RuntimeError("Fewer than 3 categories were scraped.")

    return df[["title", "price", "star_rating", "availability", "category"]].reset_index(drop=True)


if __name__ == "__main__":
    df = scrape_books()
    print(f"Scraped {len(df)} books across {df['category'].nunique()} categories.")
    print(df.head(10).to_string(index=False))
