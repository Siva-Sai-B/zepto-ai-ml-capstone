


# Zepto Support Assistant

This project implements an offline-first GenAI customer-support assistant for Zepto policies.

The application uses Retrieval-Augmented Generation (RAG), local sentence embeddings, ChromaDB, LangGraph, Pydantic, FastAPI, and Docker.

The default configuration uses deterministic mock logic and does not require an LLM API key.

---

## Technology Stack

- Python 3.11
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn
- Docker

---

## Project Structure

```text
support_assistant/
│
├── docs/
│ ├── doc_01.txt
│ ├── doc_02.txt
│ ├── doc_03.txt
│ ├── doc_04.txt
│ ├── doc_05.txt
│ ├── doc_06.txt
│ ├── doc_07.txt
│ └── doc_08.txt
│
├── ingest.py
├── graph_app.py
├── prompts.py
├── models.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# Architecture

The application follows a Retrieval-Augmented Generation pipeline with four main stages:

```text
Ingestion
 ↓
Embedding
 ↓
ChromaDB
 ↓
Retrieval
 ↓
LangGraph Routing
 ↓
Generation
 ↓
Pydantic Validation
 ↓
FastAPI Response
```

## 1. Ingestion

The document corpus contains eight Zepto policy documents stored in the `docs/` directory.

The documents cover:

- Delivery policy
- Returns and refunds
- Membership tiers
- Order tracking
- Order cancellation
- Damaged or missing items
- Gift cards
- Customer support hours

Because each document is short, each document is treated as one chunk.

The function:

```python
ensure_documents_indexed()
```

inside `ingest.py` checks whether the ChromaDB collection already contains documents.

If the collection is empty, the application loads and indexes all eight documents.

If the documents are already indexed, the existing collection is reused.

---

## 2. Embedding

The application uses the open-source Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

The model runs locally using the `sentence-transformers` Python package.

Each policy document is converted into a numerical embedding vector.

The same embedding model is also used to convert incoming policy questions into vectors.

No paid embedding API or API key is required.

---

## 3. Vector Storage

The generated embeddings are stored in ChromaDB.

The ChromaDB collection is named:

```text
zepto_policies
```

Each stored entry contains:

- document ID
- document text
- embedding
- source metadata

For example:

```text
ID: doc_01
Source: doc_01
Document: Zepto delivery policy...
Embedding: [vector]
```

---

## 4. Retrieval

Retrieval happens inside the LangGraph node:

```text
retrieve_and_answer
```

When a query is classified as a `policy_question`:

1. The user query is embedded using `all-MiniLM-L6-v2`.
2. ChromaDB compares the query embedding against stored document embeddings.
3. The top 3 most similar documents are retrieved.
4. The IDs of those documents are returned as sources.
5. The most similar document is used to construct the mock answer.

ChromaDB returns results according to vector distance, where a smaller distance represents a closer semantic match.

---

# LangGraph Workflow

The application uses a LangGraph `StateGraph` with a shared `TypedDict` state.

The graph contains three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
 User Query
 |
 v
 classify_intent
 / \
 / \
 policy_question general_question
 | |
 v v
 retrieve_and_answer direct_answer
 | |
 v |
 ChromaDB |
 top-3 retrieval |
 | |
 v v
 Mock answer END
 |
 v
 END
```

---

## Shared State

LangGraph passes a shared state object between nodes.

The state contains fields such as:

```python
query
intent
answer
sources
confidence
```

For example:

```python
{
 "query": "What is the delivery fee?",
 "intent": "policy_question",
 "answer": "Based on the retrieved context: ...",
 "sources": ["doc_01", "doc_03", "doc_05"],
 "confidence": 1.0
}
```

---

# Intent Classification

The `classify_intent` node determines whether retrieval is necessary.

In the default mock mode, the query is converted to lowercase and checked for the following keywords:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If any keyword appears, the query is classified as:

```text
policy_question
```

Otherwise it is classified as:

```text
general_question
```

A LangGraph conditional edge then routes the query to either:

```text
retrieve_and_answer
```

or:

```text
direct_answer
```

---

# MOCK_LLM Configuration

The project uses the environment variable:

```text
MOCK_LLM
```

## Default Graded Mode

When `MOCK_LLM` is unset or:

```text
MOCK_LLM=1
```

the application runs using deterministic offline logic.

No LLM API is called.

### Policy Question

A policy question performs real ChromaDB retrieval.

The final answer follows the template:

```text
Based on the retrieved context: <top retrieved chunk snippet>
```

The first approximately 200 characters of the highest-ranked retrieved policy document are used.

### General Question

A general question does not perform retrieval.

It returns the fixed response:

```text
I can only answer questions about Zepto policies right now.
```

The `sources` list is empty.

## Optional Real LLM Mode

Setting:

```text
MOCK_LLM=0
```

is reserved for an optional real-LLM extension.

The required graded implementation does not depend on this optional mode.

The project contains a structured prompt for the optional real-LLM path and retry/validation infrastructure for validating structured LLM responses.

# Structured Prompt

The structured prompt is stored in:

```text
prompts.py
```

The prompt contains the required components:

- Role
- Context
- Task
- Format
- Length
- Negative constraint
- Few-shot example

The negative constraint instructs the model not to use information outside the retrieved Zepto policy context.

The few-shot example demonstrates the expected question-and-answer behavior.

---

# Structured Output

The API output is validated with Pydantic.

The response model contains:

```json
{
 "answer": "string",
 "sources": ["document IDs"],
 "confidence": 1.0
}
```

The fields are:

- `answer` — final response text
- `sources` — IDs of retrieved documents, or an empty list for general questions
- `confidence` — floating-point value between 0 and 1

In mock mode, confidence is deterministically set to:

```text
1.0
```

---

# Real LLM Retry Validation

The optional real-LLM path contains validation/retry infrastructure.

If an LLM response fails validation against the Pydantic response model, the helper permits:

```text
1 initial attempt
+
2 additional retries
=
3 maximum attempts
```

This retry functionality is not used by the default offline mock path.

---

# FastAPI

The LangGraph workflow is exposed through FastAPI.

The main API endpoint is:

```text
POST /ask
```

Request format:

```json
{
 "query": "What is the delivery fee?"
}
```

Response format:

```json
{
 "answer": "Based on the retrieved context: ...",
 "sources": [
 "doc_01"
 ],
 "confidence": 1.0
}
```

A health-check endpoint is also provided:

```text
GET /
```

---

# Running Locally

## 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start the API

```bash
uvicorn main:app --reload
```

The application will automatically initialize the ChromaDB collection if it does not already exist.

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# Example API Calls

The following examples are executed using the default:

```text
MOCK_LLM=1
```

configuration.



## Example 1 — Policy Question

Request:


curl -X 'POST' \
  'http://127.0.0.1:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "refund policy of zepto"
}'

Response:

```json

{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_03"
  ],
  "confidence": 1
}

```

This query contains the keyword `delivery`, so it is classified as:

```text
policy_question
```

It therefore executes:

```text
classify_intent
 ↓
retrieve_and_answer
 ↓
ChromaDB top-3 retrieval
```


Similar example:


curl -X 'POST' \
  'http://127.0.0.1:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "How to track my delivery"
}'


{
  "answer": "Based on the retrieved context: Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
  "sources": [
    "doc_04",
    "doc_01",
    "doc_06"
  ],
  "confidence": 1
}

The exact second and third source IDs may vary depending on vector similarity results.

---

## Example 2 — General Question

Request:

```json
{
 "query": "What is Python?"
}
```

Response:

```json
{
 "answer": "I can only answer questions about Zepto policies right now.",
 "sources": [],
 "confidence": 1.0
}
```

This query does not contain one of the policy keywords, so it is classified as:

```text
general_question
```

The graph routes directly to:

```text
direct_answer
```

No ChromaDB retrieval or LLM call is performed.

---

# Docker

The application can also run inside Docker.

## Build the image

From the `support_assistant` directory:

```bash
docker build -t zepto-support-assistant .
```

## Run the container

```bash
docker run --rm -p 7860:7860 zepto-support-assistant
```

The API will then be available at:

```text
http://127.0.0.1:7860
```

FastAPI documentation:

```text
http://127.0.0.1:7860/docs
```

The Docker container runs in the required offline mock mode and automatically indexes the Zepto documents if its ChromaDB collection is empty.

---

# Docker API Test

Send a POST request to:

```text
http://127.0.0.1:7860/ask
```

with:

```json
{
 "query": "What is the delivery fee?"
}
```

The service performs local embedding, ChromaDB retrieval, LangGraph routing, Pydantic validation, and returns a JSON response.

---

# Complete Data Flow

For a policy question:

```text
User
 ↓
POST /ask
 ↓
FastAPI
 ↓
Pydantic AskRequest
 ↓
LangGraph
 ↓
classify_intent
 ↓
policy_question
 ↓
retrieve_and_answer
 ↓
Query Embedding
 ↓
ChromaDB Top-3 Search
 ↓
Top Policy Chunk
 ↓
Mock Generated Answer
 ↓
Pydantic AskResponse
 ↓
JSON Response
```

For a general question:

```text
User
 ↓
POST /ask
 ↓
FastAPI
 ↓
Pydantic AskRequest
 ↓
LangGraph
 ↓
classify_intent
 ↓
general_question
 ↓
direct_answer
 ↓
Fixed Mock Answer
 ↓
Pydantic AskResponse
 ↓
JSON Response
```

---

# Offline-First Design

The graded application can run without:

- OpenAI API key
- Groq API key
- Paid embedding API
- Cloud deployment
- LLM network calls

Sentence embeddings are generated locally using `all-MiniLM-L6-v2`.

The LLM behavior is replaced by deterministic mock logic when `MOCK_LLM=1`, which is the default configuration.

---

# Summary

This project demonstrates a complete RAG-based support assistant containing:

- local document ingestion
- local embedding generation
- ChromaDB vector storage
- semantic top-3 retrieval
- LangGraph orchestration
- conditional intent routing
- deterministic mock generation
- structured prompt engineering
- Pydantic output validation
- FastAPI REST API
- Docker containerization

The required application runs fully locally and does not require a paid API or LLM provider.