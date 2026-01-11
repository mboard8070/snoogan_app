# backend/main.py
import sys
from pathlib import Path

# Replicate the exact path setup your Streamlit app uses
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
TRADER_DIR = CODE_DIR / "trader"
RAG_DIR = CODE_DIR / "rag"  # include if you have rag modules

# Insert paths in the same order Streamlit does (project root first)
for p in [PROJECT_ROOT, CODE_DIR, TRADER_DIR, RAG_DIR]:
    path_str = str(p.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Now all existing absolute imports (from indicators, executor, etc.) work perfectly
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
import os
from dotenv import load_dotenv

# Load variables.env the same way Streamlit does (optional safety net)
ENV_PATH = PROJECT_ROOT / "variables.env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Snoogans API")

# Allow React dev server (and future production) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # "*" for dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def health():
    return {"status": "Snoogans API running – snoochie boochies"}