# code/rag/snoogans_brain.py — Snoogans' RAG brain, local & savage
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "../../data/knowledge")
DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/vector_db")

# code/rag/snoogans_brain.py — Snoogans' RAG brain, local & savage
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "../../data/knowledge")
DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/vector_db")

class SnoogansBrain:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0.8)
        
        if os.path.exists(DB_PATH):
            self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
            print("Snoogans loaded his existing brain from disk")
        else:
            print("Building Snoogans' brain from scratch — this might take a minute...")
            self._build_brain()
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chain = self._make_chain()

    def _build_brain(self):
        docs = []
        for file in os.listdir(KNOWLEDGE_DIR):
            path = os.path.join(KNOWLEDGE_DIR, file)
            if file.endswith(".pdf"):
                loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
            else:
                loader = TextLoader(path)
            )
            docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )
        print(f"Snoogans just ingested {len(chunks)} chunks of pure theta wisdom")

    def _make_chain(self):
        template = """You are Snoogans — a chill, sarcastic, Kevin Smith-obsessed 0DTE options trader from New Jersey.
Answer in pure Red Bank energy. Use profanity sparingly but effectively. Never be corporate.

Context: {context}

Question: {question}

Answer like Jay from Clerks would if he discovered theta decay and defined risk:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)

# Global brain — import this everywhere
brain = SnoogansBrain()