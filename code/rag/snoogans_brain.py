# code/rag/snoogans_brain.py — Snoogans' RAG brain, 100% local GGUF savage, no Ollama needed
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Paths — Workbench sees /project as root
BASE_DIR = "/project"  # Hardcoded for Workbench container stability
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "supertrader_final", "gemma-2-2b-it.Q5_K_M.gguf")

class SnoogansBrain:
    def __init__(self):
        # Local GGUF embeddings & LLM — full GPU offload, no daemon
        self.embeddings = LlamaCppEmbeddings(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,      # Offload everything to GPU
            n_batch=512,
            verbose=False
        )
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.8,
            n_gpu_layers=-1,      # All layers on GPU
            n_batch=512,
            n_ctx=4096,           # Bigger context if needed
            verbose=False
        )

        if os.path.exists(DB_PATH):
            print("Snoogans loaded his existing brain from disk — ready to sling tiny spreads.")
            self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        else:
            print("Building Snoogans' brain from scratch — this might take a minute, grab a blunt...")
            self._build_brain()

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chain = self._make_chain()

    def _build_brain(self):
        docs = []
        loaded_files = []

        # PDFs: load whole dir once
        pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith('.pdf')]
        if pdf_files:
            pdf_loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
            docs.extend(pdf_loader.load())
            loaded_files.extend(pdf_files)

        # Text files individually
        for file in os.listdir(KNOWLEDGE_DIR):
            if file.lower().endswith(('.txt', '.md', '.markdown')) and not file.startswith('.'):
                path = os.path.join(KNOWLEDGE_DIR, file)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                loaded_files.append(file)

        if not docs:
            raise ValueError("Knowledge dir empty, bro — drop some manifesto .txt or PDFs in data/knowledge/")

        print(f"Loaded {len(docs)} docs: {', '.join(loaded_files)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )
        self.vectorstore.persist()
        print(f"Snoogans ingested {len(chunks)} chunks of pure theta wisdom. Local brain built.")

    def _make_chain(self):
        template = """You are Snoogans — a chill, sarcastic, Kevin Smith-obsessed 0DTE options trader from Red Bank, NJ.
Answer in pure Jersey degenerate energy. Use profanity sparingly but effectively. Never sound corporate.
Keep responses concise, funny, and full of theta decay love.

Context (your own trade diary & rules):
{context}

Question: {question}

Answer like Jay from Clerks would if he discovered defined-risk credit spreads and lunch money theta:"""

        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)

    def add_knowledge(self, text: str, metadata: dict | None = None):
        """Inject new trade logs later without rebuild"""
        from langchain_core.documents import Document
        doc = Document(page_content=text, metadata=metadata or {})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        print("New wisdom injected — Snoogans just leveled up.")


# Global brain — import this everywhere
brain = SnoogansBrain()