from typing import TypedDict

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from prompts import SUPPORT_PROMPT_TEMPLATE
from models import AskResponse
from ingest import ensure_documents_indexed
import chromadb
import os
MOCK_LLM=os.getenv("MOCK_LLM","1")!="0"

class SupportState(TypedDict, total=False):
 query: str
 intent: str
 answer: str
 sources: list[str]
 confidence: float


POLICY_KEYWORDS = [
 "delivery",
 "return",
 "refund",
 "membership",
 "tracking",
 "cancel",
 "gift card",
 "support hours",
]


def classify_intent(state: SupportState) -> SupportState:
 query = state["query"].lower()
 if MOCK_LLM:
  if any(keyword in query for keyword in POLICY_KEYWORDS):
   state["intent"] = "policy_question"
  else:
   state["intent"] = "general_question"

  return state
 raise NotImplementedError("Real LLM Classification is not implemented yet")


def retrieve_and_answer(state: SupportState) -> SupportState:
 embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
 
 collection=ensure_documents_indexed()
 query = state["query"]

 query_embedding = embedding_model.encode(
 query
 ).tolist()

 results = collection.query(
 query_embeddings=[query_embedding],
 n_results=3,
 )
 retrieved_documents = results["documents"][0]
 retrieved_ids = results["ids"][0]
 if MOCK_LLM:
  top_chunk = retrieved_documents[0]

  top_chunk_snippet = top_chunk

  state["answer"] = (
  f"Based on the retrieved context: "
  f"{top_chunk_snippet}"
  )

  state["sources"] = retrieved_ids
  state["confidence"] = 1.0

  return state
 context="\n\n".join(retrieved_documents)
 prompt=SUPPORT_PROMPT_TEMPLATE.format(context=context,question=query,)
 state["answer"]=("REAL LLM MODE PLACEHOLDER")
 state["sources"]=retrieved_ids
 state["confidence"]=1.0
 return state

def direct_answer(state: SupportState) -> SupportState:
 if MOCK_LLM:
  state["answer"] = (
  "I can only answer questions about Zepto policies right now."
  )
 else:
  state["answer"]=("Real LLM Mode Placeholder")
 state["sources"] = []
 state["confidence"] = 1.0

 return state


def route_by_intent(state: SupportState) -> str:
 if state["intent"] == "policy_question":
  return "retrieve_and_answer"

 return "direct_answer"


graph = StateGraph(SupportState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
 "classify_intent",
 route_by_intent,
 {
 "retrieve_and_answer": "retrieve_and_answer",
 "direct_answer": "direct_answer",
 },
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

app = graph.compile()

def run_support_assistant(query:str)-> AskResponse:
 result=app.invoke({"query":query,})
 response=AskResponse(answer=result["answer"],sources=result["sources"],confidence=result["confidence"])
 return response
if __name__ == "__main__":
 print("MOCK_LLM:",MOCK_LLM)
 test_queries = [
 "What is Zepto's delivery fee?",
 "Who is the president of India?",
 ]

 for query in test_queries:
  response=run_support_assistant(query)

  print()
  print("Query:", query)
  print(response.model_dump())

