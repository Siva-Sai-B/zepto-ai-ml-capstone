import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_collection(
    name="zepto_policies"
)


query = "Can I return damaged groceries?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)


print("Question:")
print(query)

print()
print("Top results:")

for i in range(len(results["documents"][0])):
    document_id = results["ids"][0][i]
    document_text = results["documents"][0][i]
    distance = results["distances"][0][i]

    print()
    print("ID:", document_id)
    print("Distance:", distance)
    print("Text:", document_text[:200])