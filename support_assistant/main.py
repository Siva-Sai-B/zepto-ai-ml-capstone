from fastapi import FastAPI
from graph_app import run_support_assistant
from models import AskRequest,AskResponse

app=FastAPI(title="Zepto Support Assistant", version="1.0")

@app.get("/")
def health_check():
 return({"status":"running","service":"Zepto Support Assistant"})

@app.post("/ask",response_model=AskResponse)
def ask(request:AskRequest)->AskResponse:
 return run_support_assistant(request.query)