"""
Enhanced PDF QA Engine - Conversational & Intelligent
Features: Intent detection, query understanding, conversation memory, robust error handling
"""

# ===================== IMPORTS =====================

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import logging
import time
import os
import re
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# ===================== LLM BACKENDS =====================

try:
    from llm_providers import RobustLLM, LLMConfig, ProviderType
    ROBUST_LLM_AVAILABLE = True
except ImportError:
    ROBUST_LLM_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

# ===================== CONFIG FALLBACK =====================

try:
    from config import EMBEDDING_CONFIG, PDF_CONFIG, RERANKER_CONFIG, RETRIEVAL_CONFIG
except ImportError:
    EMBEDDING_CONFIG = {"model_name": "all-MiniLM-L6-v2"}
    PDF_CONFIG = {"chunk_size": 800, "chunk_overlap": 200}
    RERANKER_CONFIG = {"enabled": False}
    RETRIEVAL_CONFIG = {"initial_k": 3, "summary_k": 8}

# ===================== CONSTANTS =====================

VECTOR_STORE_DIR = "vector_stores"
UPLOADS_DIR = "uploads"
MAX_CONTEXT_LENGTH = 3000
MAX_CHUNK_LENGTH = 500

logger = logging.getLogger(__name__)

# ===================== DATA MODELS =====================

class QueryIntent(Enum):
    GREETING = "greeting"
    SUMMARY = "summary"
    QUESTION = "question"

@dataclass
class ConversationContext:
    last_query: str = ""
    last_response: str = ""
    turn_count: int = 0

    def update(self, query: str, response: str):
        self.last_query = query
        self.last_response = response[:300]
        self.turn_count += 1

@dataclass
class PDFMetadata:
    filename: str
    file_hash: str
    chunk_count: int
    indexed_at: float

# ===================== ENGINE =====================

class PDFQAEngine:
    """
    Main PDF Question Answering Engine
    """

    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        provider: str = "ollama",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.provider_name = provider

        self._init_llm(model_name, base_url, provider, api_key, temperature)
        self._init_embeddings()

        self.vector_cache: Dict[str, Any] = {}
        self.contexts: Dict[str, ConversationContext] = {}

        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(UPLOADS_DIR, exist_ok=True)

    # ===================== INIT HELPERS =====================

    def _init_llm(self, model_name, base_url, provider, api_key, temperature):
        if ROBUST_LLM_AVAILABLE:
            try:
                cfg = LLMConfig(
                    provider=ProviderType(provider),
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    streaming=True,
                )
                self.llm = RobustLLM([cfg])
                self._robust = True
                return
            except Exception:
                pass

        if ChatOllama:
            self.llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                streaming=True,
            )
            self._robust = False
        else:
            raise RuntimeError("No LLM backend available")

    def _init_embeddings(self):
        model = EMBEDDING_CONFIG.get("model_name", "all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=model)

    # ===================== PDF INGEST =====================

    def _file_hash(self, path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def ingest_pdf(self, pdf_path: str) -> PDFMetadata:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=PDF_CONFIG["chunk_size"],
            chunk_overlap=PDF_CONFIG["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)

        store = FAISS.from_documents(chunks, self.embeddings)
        store_path = os.path.join(VECTOR_STORE_DIR, os.path.basename(pdf_path))
        store.save_local(store_path)

        self.vector_cache[pdf_path] = store

        return PDFMetadata(
            filename=os.path.basename(pdf_path),
            file_hash=self._file_hash(pdf_path),
            chunk_count=len(chunks),
            indexed_at=time.time(),
        )

    def is_pdf_indexed(self, pdf_path: str) -> bool:
        return os.path.exists(os.path.join(VECTOR_STORE_DIR, os.path.basename(pdf_path)))

    # ===================== QA =====================

    def _get_context(self, session_id: str) -> ConversationContext:
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext()
        return self.contexts[session_id]

    def answer_question(
        self,
        question: str,
        pdf_file_path: str,
        callbacks: Optional[List] = None,
        chat_history: Optional[List] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        start = time.time()
        context = self._get_context(session_id)

        if not self.is_pdf_indexed(pdf_file_path):
            return {
                "result": "PDF not indexed yet.",
                "response_time": 0,
                "source_documents": [],
            }

        if pdf_file_path not in self.vector_cache:
            self.vector_cache[pdf_file_path] = FAISS.load_local(
                os.path.join(VECTOR_STORE_DIR, os.path.basename(pdf_file_path)),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        docs = self.vector_cache[pdf_file_path].similarity_search(
            question, k=RETRIEVAL_CONFIG["initial_k"]
        )

        context_text = "\n\n".join(
            d.page_content[:MAX_CHUNK_LENGTH] for d in docs
        )[:MAX_CONTEXT_LENGTH]

        prompt = f"""
Answer ONLY from the context below.

Context:
{context_text}

Question:
{question}

Answer:
"""

        if self._robust:
            answer = self.llm.invoke(prompt)
        else:
            resp = self.llm.invoke(prompt, config={"callbacks": callbacks})
            answer = resp.content if hasattr(resp, "content") else str(resp)

        context.update(question, answer)

        return {
            "result": answer,
            "response_time": time.time() - start,
            "source_documents": docs,
        }

    # ===================== UTILS =====================

    def clear_conversation(self, session_id: str):
        self.contexts.pop(session_id, None)

    def delete_pdf_index(self, pdf_path: str):
        store_path = os.path.join(VECTOR_STORE_DIR, os.path.basename(pdf_path))
        if os.path.exists(store_path):
            shutil.rmtree(store_path)

    def get_model_info(self):
        return {
            "model": self.model_name,
            "provider": self.provider_name,
        }
