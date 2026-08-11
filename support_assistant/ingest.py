from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path("docs")

model = SentenceTransformer("all-MiniLM-L6-v2")




def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
            }
        )

    return documents
def get_collection():
 client=chromadb.PersistentClient(path="./chroma_db")
 collection=client.get_or_create_collection(name="zepto_policies")
 return collection

def ensure_documents_indexed():
    collection=get_collection()
    if collection.count()>0:
     print("Chromadb already contains",collection.count(),"documents")
     return collection
    documents = load_documents()

    for doc in documents:
        embedding = model.encode(doc["text"]).tolist()

        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": doc["id"]
                }
            ],
        )

        print(f"Stored {doc['id']}")
     
    print()
    print("Total documents stored:", collection.count())
    return collection


if __name__ == "__main__":
    ensure_documents_indexed()
