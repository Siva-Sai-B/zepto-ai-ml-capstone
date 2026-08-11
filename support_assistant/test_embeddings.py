from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_1 = "Can I return damaged groceries?"
sentence_2 = "Damaged items can be reported for a refund."
sentence_3 = "How do I become a Python developer?"

embedding_1 = model.encode(sentence_1)
embedding_2 = model.encode(sentence_2)
embedding_3 = model.encode(sentence_3)

similarity_12 = cos_sim(embedding_1, embedding_2)
similarity_13 = cos_sim(embedding_1, embedding_3)

print(f'''embeddings:{embedding_1},{embedding_2},{embedding_3}''')
print(f'''cos_sim:{similarity_12},{similarity_13}''')

print("Return question vs refund sentence:")
print(similarity_12.item())

print()

print("Return question vs Python sentence:")
print(similarity_13.item())
