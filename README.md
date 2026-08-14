# Zepto Data & AI Platform Capstone

This repository contains the capstone project for the **Certificate Program in Artificial Intelligence and Machine Learning**.

The project is a single connected platform with three modules:

1. **Data Engineering Pipeline** — `/data_pipeline`
2. **Analytics Pipeline** — `/analytics`
3. **GenAI Support Assistant** — `/support_assistant`

---

## Repository Structure

```text
zepto-ai-ml-capstone/
│
├── data_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   ├── run_pipeline.py
│   ├── scrape_books.py
│   ├── clean_books.py
│   ├── database.py
│   ├── queries.py
│   ├── data/
│   ├── database/
│   └── outputs/
│
├── analytics/
│   ├── README.md
│   ├── requirements.txt
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── models/
│   ├── plots/
│   └── outputs/
│
├── support_assistant/
│   ├── README.md
│   ├── requirements.txt
│   ├── graph_app.py
│   ├── main.py
│   ├── models.py
│   ├── prompts.py
│   └── ...
│
├── requirements_overall.txt
├── .gitignore
└── README.md
```

---

# 1. Data Engineering Pipeline

Location:

`/data_pipeline`

The data pipeline demonstrates an end-to-end raw-to-relational workflow:

**Scrape → Clean → Convert → Store → Query → Validate with Pandas**

### Key tasks

- Scrapes book data from `books.toscrape.com`
- Collects at least 60 books across multiple categories
- Cleans price, rating, and availability fields
- Converts GBP prices to INR using the required fixed project rate:
  **1 GBP = 105.50 INR**
- Stores the cleaned data in a normalized SQLite database
- Uses a two-table primary-key/foreign-key schema
- Executes SQL queries covering filtering, ordering, limits, distinct values, range filtering, and joins
- Reads SQL results using `pandas.read_sql`
- Reproduces the join using `pandas.merge`
- Compares the SQL and pandas results

### Run

From the repository root:

```bash
cd data_pipeline
pip install -r requirements.txt
python run_pipeline.py
```

The pipeline generates:

- `data/books_cleaned.csv`
- `database/zepto_books.db`
- `outputs/sql_query_results.txt`
- `outputs/pandas_comparison.txt`

More details are available in `/data_pipeline/README.md`.

---

# 2. Analytics Pipeline

Location:

`/analytics`

The analytics module follows a complete analyst-to-data-scientist workflow using the Titanic dataset.

**Load → Profile → Clean → Explore → Visualize → Model → Tune → Evaluate → Save Pipeline**

### Key tasks

- Loads the Titanic dataset using Seaborn's built-in loader
- Saves `titanic.csv` as the committed offline fallback
- Profiles shape, data types, descriptive statistics, and missing values
- Applies documented missing-value handling
- Performs univariate and bivariate analysis
- Detects outliers using the IQR rule
- Analyzes fare distribution and skewness
- Calculates survival rates by sex, passenger class, and sex/class combination
- Builds the required six-column correlation matrix and heatmap
- Performs exploratory standardization of `age` and `fare`
- Uses a stratified train/test split
- Builds train-only preprocessing using `ColumnTransformer`/`Pipeline`
- Trains Logistic Regression, Decision Tree, and Random Forest classifiers
- Evaluates models using accuracy, precision, recall, F1, confusion matrices, and ROC/AUC
- Compares baseline, `class_weight='balanced'`, and SMOTE approaches
- Performs Random Forest `GridSearchCV` and reports the OOB score
- Performs a multivariate linear-regression side task for `fare`
- Reports MAE, RMSE, R², and Adjusted R²
- Saves the complete preprocessing + model pipeline using `joblib`

### Run

From the repository root:

```bash
cd analytics
pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py
```

The module generates/updates:

- `titanic.csv`
- `plots/`
- `outputs/`
- `models/best_model_pipeline.joblib`

More details are available in `/analytics/README.md`.

---

# 3. GenAI Support Assistant

Location:

`/support_assistant`

The support assistant implements an **offline-first Retrieval-Augmented Generation (RAG)** workflow for answering questions from Zepto policy documents.

### Main technologies

- Python
- Sentence Transformers
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Docker

### Key design

The assistant retrieves relevant policy information from the project's Zepto support documents and uses that retrieved context to generate grounded answers.

The FastAPI service exposes an endpoint for asking support questions.

### Run

From the repository root:

```bash
cd support_assistant
pip install -r requirements.txt
python main.py
```

Refer to `/support_assistant/README.md` for the complete setup, API usage, and Docker instructions.

---

# Installation and Dependencies

Each module maintains its own `requirements.txt`.

Install dependencies separately for the module you are working on:

```bash
cd data_pipeline
pip install -r requirements.txt
```

or:

```bash
cd analytics
pip install -r requirements.txt
```

or:

```bash
cd support_assistant
pip install -r requirements.txt
```

A consolidated dependency file is also included at the repository root:

`requirements_overall.txt`

> The module-level requirements files are the authoritative dependency lists for their respective modules.

---

# Python Virtual Environment

A Python virtual environment is recommended.

From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Then install the requirements for the module being developed.

The virtual environment itself is not committed to Git.

---

# Git Workflow

The project was developed using feature branches and merged back into `main`.

The repository history contains feature branches for the major modules, including:

- `feature/support_assistant`
- `feature/data_pipeline`
- `feature/analytics`

The feature branches were committed and subsequently merged into `main`.

To inspect the project history:

```bash
git log --graph --oneline --all
```

---

# Project Design Summary

## Data Pipeline

The data pipeline separates scraping, cleaning, database creation, and SQL analysis into different Python modules. SQLite was selected because it provides a lightweight relational database without requiring a separate database server.

## Analytics Pipeline

The analytics workflow separates exploratory analysis from predictive modeling while keeping both stages connected through the same cleaned `titanic.csv`. The modeling pipeline uses train-only preprocessing to prevent test-set leakage.

## Support Assistant

The support assistant uses retrieval-augmented generation so that responses are grounded in the project's policy documents. An offline-first approach reduces dependency on external services during normal document retrieval.

---

# Reproducibility

The project is designed so that the main outputs can be regenerated from the repository code.

- The data pipeline can recreate its SQLite database from the public scraping source.
- The analytics module includes `titanic.csv` as an offline fallback and reproducible modeling input.
- The support assistant contains its policy/document assets and application code.

No paid external services are required for the required project functionality.

---

# Final Submission

This repository is the **single public GitHub repository** containing all three capstone modules:

- `/data_pipeline`
- `/analytics`
- `/support_assistant`

Each module contains its own implementation, dependency file, outputs/artifacts where required, and module-level documentation.
