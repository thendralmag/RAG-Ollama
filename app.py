# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time, traceback
from embedding_index import EmbeddingIndex
from rerank import ReRanker
from ollama_client import generate_with_ollama

app = FastAPI(title="Local RAG API")
index = EmbeddingIndex()
# try load index if exists
try:
    index.load_index()
    INDEX_LOADED = True
except Exception:
    INDEX_LOADED = False

reranker = ReRanker()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    re_rank_top_n: int = 5
    model: str = "ollama:your-model"  # adjust to your ollama model ref

@app.get("/status")
def status():
    return {"index_loaded": INDEX_LOADED, "index_size": index.index.ntotal if hasattr(index.index, 'ntotal') else 0}

@app.post("/rebuild-index")
def rebuild_index(files: List[str]):
    # files: paths to ingest; in prod this would be an upload or S3 path
    try:
        from ingest import ingest_files
        chunks = ingest_files(files)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced")
        index.__init__()  # reset
        index.add_chunks(chunks)
        index.save_index()
        return {"status": "ok", "chunks_indexed": len(chunks)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query is required")
    start = time.time()
    try:
        emb_results = index.search(req.query, top_k=req.top_k)
        if not emb_results:
            return {"answer": "", "retrieved": [], "time": time.time() - start}
        reranked = reranker.rerank(req.query, emb_results, top_n=req.re_rank_top_n)
        # Build context
        ctx = "\n\n---\n\n".join([f"Source: {r['meta']['source']}\n\n{r['meta']['text']}" for r in reranked])
        prompt = f"""You are a professional fitness coach. Use the retrieved documents to answer concisely and practically.

Context:
{ctx}

Question:
{req.query}

Answer in bullets, include 2 corrective drills and 3 coaching cues. Be concise.
"""
        gen = generate_with_ollama(req.model, prompt)
        elapsed = time.time() - start
        return {"answer_generation": gen, "retrieved": reranked, "time": elapsed}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
