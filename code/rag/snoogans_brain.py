# /home/mboard76/nvidia-workbench/snoogan_app/code/rag/snoogans_brain.py 

import os
import shutil 
import stat
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- CONFIG CONSTANTS (FIXED BASE_DIR) ---
# FIX: Using the actual, writable path from your user environment.
BASE_DIR = "/home/mboard76/nvidia-workbench/snoogan_app" 
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")

# NOTE: Using a fixed host IP is robust for Docker/Ollama setup
OLLAMA_HOST = "http://172.17.0.1:11434" 
DATA_DIR = os.path.join(BASE_DIR, "data") 

# --- RAG CLASS DEFINITION ---

class SnoogansBrain:
    def __init__(self, chunk_size=400, chunk_overlap=150): 
        # This will now succeed because the path is writable by the user
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True) 
        
        # NOTE: Permission setting is generally not needed on a standard system but kept for robustness
        if os.path.exists(DATA_DIR):
            try:
                os.chmod(DATA_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                print(f"WARNING: Could not set permissions on {DATA_DIR}. Error: {e}")

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_HOST
        )
        self.llm = ChatOllama(
            model="snoogans:latest",
            temperature=0.1, 
            base_url=OLLAMA_HOST
        )
        
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap

        # Load or Build Vector Store
        if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
            print("Snoogans loaded his existing brain from disk — ready to sling spreads.")
            self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        else:
            print("Building Snoogans' brain from scratch — grab a blunt...")
            self._build_brain()

        if not hasattr(self, 'vectorstore'):
             raise RuntimeError("Vector store failed to build. Check 'data/knowledge' contents.")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.rag_chain = self._make_rag_chain()

    def _build_brain(self):
        docs = []
        loaded_files = []

        # Load PDFs
        pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith('.pdf')]
        if pdf_files:
            pdf_loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
            docs.extend(pdf_loader.load())
            loaded_files.extend(pdf_files)

        # Load Text Files
        for file in os.listdir(KNOWLEDGE_DIR):
            if file.lower().endswith(('.txt', '.md', '.markdown')) and not file.startswith('.'):
                path = os.path.join(KNOWLEDGE_DIR, file)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                loaded_files.append(file)

        if not docs:
            raise ValueError("Knowledge dir empty — drop manifesto .txt or PDFs in data/knowledge/")

        print(f"Loaded {len(docs)} docs: {', '.join(loaded_files)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )
        
        print(f"Snoogans ingested {len(chunks)} chunks of theta wisdom. Brain built.")

    def _make_rag_chain(self):
        """Creates the STRICT RAG chain for manifesto rules."""
        template = """<SYSTEM_INSTRUCTIONS>
You are a highly specialized data extraction bot. Your ONLY goal is to retrieve the answer to the user's question using the provided CONTEXT.

1.  **Strict Rule:** You MUST quote the answer directly and word-for-word from the context.
2.  **Formatting:** Present the answer as simple, unadorned text.
3.  **Fallback:** If the exact answer or rule is not in the CONTEXT, your entire response MUST be the single phrase: 'Manifesto don't specify that yet, bro.'
</SYSTEM_INSTRUCTIONS>

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_rag(self, question: str) -> str:
        """Invokes the strict RAG chain."""
        return self.rag_chain.invoke(question)

    def ask_general(self, question: str) -> str:
        """Handles conversational/general questions without using RAG context."""
        general_template = """You are Snoogans — Red Bank degenerate 0DTE options trader. 
You are chill, concise, and funny. Use pure Jersey energy.
Always sign off with "Lunch money secured." or "Snoochie boochies."

User Question: {question}

Answer:"""
        
        general_prompt = ChatPromptTemplate.from_template(general_template)
        general_llm = ChatOllama(model="snoogans:latest", temperature=0.6, base_url=OLLAMA_HOST) 
        general_chain = general_prompt | general_llm | StrOutputParser()
        
        return general_chain.invoke({"question": question})