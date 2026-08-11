from pathlib import Path

DOCS_DIR = Path("docs")

documents = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    documents.append(
        {
            "id": file_path.stem,
            "text": text,
        }
    )

for doc in documents:
    print(doc["id"])
    print(doc["text"][:100])
    print("-" * 50)