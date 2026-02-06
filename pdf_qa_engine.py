"""
Enhanced PDF QA Engine - Conversational & Intelligent
Features: Intent detection, query understanding, conversation memory, robust error handling
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import time
import os
import re
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import LLM providers
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

# Import configs
try:
    from config import EMBEDDING_CONFIG, PDF_CONFIG, RERANKER_CONFIG, RETRIEVAL_CONFIG
except ImportError:
    EMBEDDING_CONFIG = {'model_name': 'all-MiniLM-L6-v2'}
    PDF_CONFIG = {'chunk_size': 800, 'chunk_overlap': 200}
    RERANKER_CONFIG = {'enabled': False, 'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'top_k': 5}
    RETRIEVAL_CONFIG = {'initial_k': 3, 'summary_k': 8, 'use_mmr': False, 'mmr_lambda': 0.5}

# Speed optimization constants
MAX_CONTEXT_LENGTH = 3000  # Limit context to reduce LLM processing time
MAX_CHUNK_LENGTH = 500  # Truncate individual chunks

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_stores"
UPLOADS_DIR = "uploads"


class QueryIntent(Enum):
    """Types of user query intents"""
    GREETING = "greeting"
    SUMMARY = "summary"
    SPECIFIC_QUESTION = "specific"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    LIST_REQUEST = "list"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    YES_NO = "yes_no"
    DEFINITION = "definition"
    HOW_TO = "how_to"
    WHY = "why"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    """Tracks conversation state for better understanding"""
    topics: List[str] = field(default_factory=list)
    last_query: str = ""
    last_response: str = ""
    last_intent: QueryIntent = QueryIntent.UNKNOWN
    mentioned_entities: List[str] = field(default_factory=list)
    current_focus: str = ""
    turn_count: int = 0

    def update(self, query: str, response: str, intent: QueryIntent, entities: List[str] = None):
        """SPEED: Simplified update - skip expensive operations"""
        self.last_query = query
        self.last_response = response[:300]  # SPEED: Truncate stored response
        self.last_intent = intent
        self.turn_count += 1
        # SPEED: Simple focus extraction from query
        words = [w for w in query.split() if len(w) > 4]
        if words:
            self.current_focus = words[0]


@dataclass
class PDFMetadata:
    """Metadata for a processed PDF"""
    filename: str
    filepath: str
    page_count: int
    chunk_count: int
    file_hash: str
    indexed_at: float
    file_size: int


class QueryUnderstanding:
    """Intelligent query analysis and understanding"""

    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy|yo)[\s!.,]*$',
        r'^(what\'?s?\s*up|how\s*are\s*you)[\s!.,?]*$',
    ]

    SUMMARY_KEYWORDS = ['summary', 'summarize', 'summarise', 'overview', 'about', 'tldr',
                        'brief', 'main points', 'key points', 'gist', 'outline', 'describe']

    COMPARISON_KEYWORDS = ['compare', 'comparison', 'difference', 'differ', 'versus', 'vs',
                           'contrast', 'similar', 'similarity', 'better', 'worse']

    LIST_KEYWORDS = ['list', 'enumerate', 'all the', 'what are the', 'name all',
                     'give me all', 'show all', 'types of', 'kinds of', 'examples of']

    DEFINITION_KEYWORDS = ['what is', 'what are', 'define', 'definition', 'meaning of',
                           'what does', "what's", 'explain what']

    HOW_TO_KEYWORDS = ['how to', 'how do', 'how can', 'how should', 'steps to',
                       'process of', 'procedure', 'method for', 'way to']

    WHY_KEYWORDS = ['why', 'reason', 'cause', 'because', 'purpose of', 'motivation']

    FOLLOW_UP_PATTERNS = [
        r'^(and|also|what about|how about|tell me more|more about|elaborate|explain more)',
        r'^(can you|could you|please)\s*(explain|elaborate|tell|give)',
        r'\b(this|that|it|they|these|those|the same)\b',
        r'^(yes|no|ok|okay|sure|right|exactly|correct)',
    ]

    @classmethod
    def detect_intent(cls, query: str, context: ConversationContext = None) -> QueryIntent:
        """Detect the intent behind a user query"""
        query_lower = query.lower().strip()

        # Check for greetings
        for pattern in cls.GREETING_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return QueryIntent.GREETING

        # Check for follow-up (context-dependent)
        if context and context.turn_count > 0:
            for pattern in cls.FOLLOW_UP_PATTERNS:
                if re.search(pattern, query_lower):
                    return QueryIntent.FOLLOW_UP

        # Check for yes/no questions
        if query_lower.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                                    'can ', 'could ', 'will ', 'would ', 'should ', 'has ', 'have ')):
            return QueryIntent.YES_NO

        # Check for definition requests
        for keyword in cls.DEFINITION_KEYWORDS:
            if keyword in query_lower:
                return QueryIntent.DEFINITION

        # Check for how-to questions
        for keyword in cls.HOW_TO_KEYWORDS:
            if keyword in query_lower:
                return QueryIntent.HOW_TO

        # Check for why questions
        for keyword in cls.WHY_KEYWORDS:
            if query_lower.startswith(keyword) or f" {keyword} " in query_lower:
                return QueryIntent.WHY

        # Check for summaries
        for keyword in cls.SUMMARY_KEYWORDS:
            if keyword in query_lower:
                return QueryIntent.SUMMARY

        # Check for comparisons
        for keyword in cls.COMPARISON_KEYWORDS:
            if keyword in query_lower:
                return QueryIntent.COMPARISON

        # Check for lists
        for keyword in cls.LIST_KEYWORDS:
            if keyword in query_lower:
                return QueryIntent.LIST_REQUEST

        # Check for explanations
        if any(word in query_lower for word in ['explain', 'elaborate', 'clarify', 'describe']):
            return QueryIntent.EXPLANATION

        return QueryIntent.SPECIFIC_QUESTION

    @classmethod
    def reformulate_query(cls, query: str, context: ConversationContext = None, intent: QueryIntent = None) -> str:
        """Reformulate query for better retrieval"""
        reformulated = query.strip()

        # Handle follow-up queries by adding context
        if intent == QueryIntent.FOLLOW_UP and context:
            # Replace pronouns with actual references
            pronouns = ['it', 'this', 'that', 'they', 'these', 'those', 'the same']
            query_lower = reformulated.lower()

            for pronoun in pronouns:
                if f" {pronoun} " in f" {query_lower} " or query_lower.startswith(f"{pronoun} "):
                    if context.current_focus:
                        reformulated = re.sub(
                            rf'\b{pronoun}\b',
                            context.current_focus,
                            reformulated,
                            flags=re.IGNORECASE
                        )
                    elif context.mentioned_entities:
                        reformulated = re.sub(
                            rf'\b{pronoun}\b',
                            context.mentioned_entities[-1],
                            reformulated,
                            flags=re.IGNORECASE
                        )

            # If query is too short, add context from last query
            if len(reformulated.split()) < 4 and context.last_query:
                # Extract key terms from last query
                last_terms = [w for w in context.last_query.split()
                             if len(w) > 3 and w.lower() not in ['what', 'where', 'when', 'how', 'why', 'the', 'and', 'for']]
                if last_terms:
                    reformulated = f"{reformulated} regarding {' '.join(last_terms[:3])}"

        # For definition queries, ensure the term is clear
        if intent == QueryIntent.DEFINITION:
            # Extract the term being defined
            match = re.search(r'(?:what is|define|meaning of|what\'s|what are)\s+(?:a|an|the)?\s*(.+?)[\?.]?$',
                            reformulated, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                reformulated = f"definition and explanation of {term}"

        return reformulated

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """Extract potential entities/topics from text"""
        # Simple extraction - words that are capitalized or in quotes
        entities = []

        # Quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        entities.extend(quoted)

        # Capitalized words (potential proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and len(word) > 2:
                entities.append(word.strip('.,!?'))

        # Technical terms (words with numbers or special patterns)
        technical = re.findall(r'\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b', text)
        entities.extend(technical)

        return list(set(entities))[:10]


class PromptBuilder:
    """Builds concise prompts optimized for speed"""

    # SPEED: Shorter, more direct prompts
    FAST_PROMPTS = {
        QueryIntent.SUMMARY: "Summarize the key points concisely using bullets.",
        QueryIntent.SPECIFIC_QUESTION: "Answer directly and concisely.",
        QueryIntent.COMPARISON: "Compare briefly with key differences.",
        QueryIntent.LIST_REQUEST: "List the items concisely.",
        QueryIntent.DEFINITION: "Define clearly in 1-2 sentences.",
        QueryIntent.HOW_TO: "Explain the steps briefly.",
        QueryIntent.WHY: "Explain the reason concisely.",
        QueryIntent.EXPLANATION: "Explain clearly and briefly.",
        QueryIntent.YES_NO: "Answer yes/no, then briefly explain.",
        QueryIntent.FOLLOW_UP: "Continue from the previous answer.",
    }

    @staticmethod
    def build_prompt(
        question: str,
        context_text: str,
        chat_history: str,
        intent: QueryIntent,
        conversation_context: ConversationContext = None
    ) -> str:
        """Build concise prompt for faster LLM response"""

        task = PromptBuilder.FAST_PROMPTS.get(intent, "Answer concisely.")

        # SPEED: Minimal prompt structure
        prompt = f"""Answer based ONLY on the context below. Be concise. {task}

Context:
{context_text}

Question: {question}

Answer:"""

        # Add minimal history for follow-ups only
        if intent == QueryIntent.FOLLOW_UP and conversation_context and conversation_context.last_query:
            prompt = f"""Previous: {conversation_context.last_query[:100]}

{prompt}"""

        return prompt


class PDFQAEngine:
    """
    Intelligent PDF Question-Answering Engine
    Features: Intent detection, conversation memory, query understanding, robust error handling
    """

    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        provider: str = "ollama",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_workers: int = 4,
    ):
        self._init_llm(model_name, base_url, provider, api_key, temperature)
        self._init_embeddings()
        self._init_reranker()

        # Caches and state
        self._vector_store_cache: Dict[str, Any] = {}
        self._pdf_metadata_cache: Dict[str, PDFMetadata] = {}
        self._conversation_contexts: Dict[str, ConversationContext] = {}
        self._response_cache: Dict[str, Dict] = {}  # SPEED: Cache recent responses
        self._cache_max_size = 50

        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        for dir_path in [VECTOR_STORE_DIR, UPLOADS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def _init_llm(self, model_name, base_url, provider, api_key, temperature):
        """Initialize LLM with robust fallback"""
        self.model_name = model_name
        self.provider_name = provider
        self._use_robust_llm = False

        if ROBUST_LLM_AVAILABLE:
            try:
                provider_type = ProviderType(provider.lower())
                config = LLMConfig(
                    provider=provider_type,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    streaming=True,
                )
                self.llm = RobustLLM([config])
                self._use_robust_llm = True
                logger.info(f"Initialized RobustLLM: {provider}/{model_name}")
                return
            except Exception as e:
                logger.warning(f"RobustLLM failed: {e}")

        if ChatOllama:
            self.llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                streaming=True,
            )
            logger.info(f"Initialized ChatOllama: {model_name}")
        else:
            raise RuntimeError("No LLM backend available")

    def _init_embeddings(self):
        """Initialize embeddings with fallback"""
        model_name = EMBEDDING_CONFIG.get('model_name', 'all-MiniLM-L6-v2')
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"Loaded embeddings: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _init_reranker(self):
        """Initialize reranker if available"""
        self.reranker = None
        if CROSSENCODER_AVAILABLE and RERANKER_CONFIG.get('enabled', False):
            try:
                model = RERANKER_CONFIG.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.reranker = CrossEncoder(model)
                logger.info(f"Loaded reranker: {model}")
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")

    def _get_conversation_context(self, session_id: str = "default") -> ConversationContext:
        """Get or create conversation context for a session"""
        if session_id not in self._conversation_contexts:
            self._conversation_contexts[session_id] = ConversationContext()
        return self._conversation_contexts[session_id]

    def _get_vector_store_path(self, pdf_file_path: str) -> str:
        pdf_filename = os.path.basename(pdf_file_path)
        return os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}.faiss")

    def _compute_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def ingest_pdf(
        self,
        pdf_file_path: str,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> PDFMetadata:
        """Ingest a PDF with improved chunking"""
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)
        current_hash = self._compute_file_hash(pdf_file_path)

        if not force_reindex and os.path.exists(vector_store_path):
            cached = self._pdf_metadata_cache.get(pdf_filename)
            if cached and cached.file_hash == current_hash:
                return cached

        logger.info(f"Ingesting: {pdf_file_path}")

        if progress_callback:
            progress_callback(0.1, f"Loading {pdf_filename}...")

        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        if progress_callback:
            progress_callback(0.3, "Preprocessing...")

        for doc in documents:
            doc.page_content = self._preprocess_text(doc.page_content)

        chunk_size = PDF_CONFIG.get('chunk_size', 800)
        chunk_overlap = PDF_CONFIG.get('chunk_overlap', 200)
        separators = PDF_CONFIG.get('separators', ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "])

        if progress_callback:
            progress_callback(0.4, "Chunking...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            chunks = [Document(
                page_content="[No extractable text in this document.]",
                metadata={"source": pdf_file_path, "page": 0}
            )]

        if progress_callback:
            progress_callback(0.6, "Creating embeddings...")

        vector_store = FAISS.from_documents(chunks, self.embeddings)

        if progress_callback:
            progress_callback(0.9, "Saving...")

        vector_store.save_local(vector_store_path)

        metadata = PDFMetadata(
            filename=pdf_filename,
            filepath=pdf_file_path,
            page_count=len(documents),
            chunk_count=len(chunks),
            file_hash=current_hash,
            indexed_at=time.time(),
            file_size=os.path.getsize(pdf_file_path)
        )
        self._pdf_metadata_cache[pdf_filename] = metadata

        if progress_callback:
            progress_callback(1.0, "Done!")

        return metadata

    def _load_vector_store(self, pdf_file_path: str):
        """Load vector store with caching"""
        if pdf_file_path == "ALL_PDFS":
            stores = []
            if os.path.exists(VECTOR_STORE_DIR):
                for fn in os.listdir(VECTOR_STORE_DIR):
                    if fn.endswith(".faiss"):
                        try:
                            vs = FAISS.load_local(
                                os.path.join(VECTOR_STORE_DIR, fn),
                                self.embeddings,
                                allow_dangerous_deserialization=True
                            )
                            stores.append(vs)
                        except Exception as e:
                            logger.error(f"Failed to load {fn}: {e}")
            if stores:
                merged = stores[0]
                for vs in stores[1:]:
                    merged.merge_from(vs)
                return merged
            return None

        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            return self._vector_store_cache[pdf_filename]

        if os.path.exists(vector_store_path):
            try:
                vs = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._vector_store_cache[pdf_filename] = vs
                return vs
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")

        return None

    def _retrieve_documents(
        self,
        query: str,
        vector_store,
        intent: QueryIntent
    ) -> List[Document]:
        """Fast retrieval with minimal chunks for speed"""

        # SPEED: Use fewer chunks - 3 for specific, 6 for summaries
        if intent in [QueryIntent.SUMMARY, QueryIntent.LIST_REQUEST]:
            k = min(RETRIEVAL_CONFIG.get('summary_k', 8), 6)
        elif intent in [QueryIntent.COMPARISON]:
            k = 4
        else:
            k = min(RETRIEVAL_CONFIG.get('initial_k', 3), 3)

        try:
            # SPEED: Simple similarity search is faster than MMR
            docs = vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _rerank_documents(self, query: str, documents: List, top_k: int = 5) -> List:
        """Rerank for better relevance"""
        if not self.reranker or not documents:
            return documents[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return documents[:top_k]

    def _format_context(self, documents: List) -> str:
        """Format retrieved documents - optimized for speed"""
        seen = set()
        parts = []
        total_length = 0

        for doc in documents:
            content = doc.page_content

            # SPEED: Skip duplicates
            content_hash = hash(content[:50])
            if content_hash in seen:
                continue
            seen.add(content_hash)

            # SPEED: Truncate long chunks
            if len(content) > MAX_CHUNK_LENGTH:
                content = content[:MAX_CHUNK_LENGTH] + "..."

            # SPEED: Stop if context is long enough
            if total_length + len(content) > MAX_CONTEXT_LENGTH:
                break

            parts.append(content)
            total_length += len(content)

        return "\n\n".join(parts)

    def _format_chat_history(self, history: List[Dict]) -> str:
        """Format chat history for context"""
        if not history:
            return ""

        formatted = []
        for msg in history[-4:]:  # Last 4 messages
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")[:200]  # Truncate long messages
            if role and content:
                formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _generate_greeting_response(self, context: ConversationContext) -> str:
        """Generate contextual greeting"""
        if context.turn_count == 0:
            return """Hello! I'm here to help you explore your documents. Feel free to ask me anything like:

- "What is this document about?"
- "Summarize the main points"
- "Find information about [topic]"
- "Explain [concept] from the document"

What would you like to know?"""
        else:
            return "Hi again! What else would you like to know about your document?"

    def _handle_no_results(self, query: str, intent: QueryIntent) -> str:
        """Generate helpful response when no results found"""
        suggestions = {
            QueryIntent.SPECIFIC_QUESTION: "Try rephrasing your question or asking about a broader topic.",
            QueryIntent.DEFINITION: "The term might not be defined in this document. Try asking about related concepts.",
            QueryIntent.HOW_TO: "This document might not contain instructions for that. Try asking what topics are covered.",
            QueryIntent.SUMMARY: "I couldn't find enough content to summarize. Is the PDF properly indexed?",
        }

        base = "I couldn't find relevant information for your question."
        suggestion = suggestions.get(intent, "Try asking in a different way or about a different topic.")

        return f"{base} {suggestion}"

    def answer_question(
        self,
        question: str,
        pdf_file_path: str,
        callbacks: Optional[List] = None,
        chat_history: Optional[List] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Answer question - optimized for speed"""
        start_time = time.time()

        # SPEED: Check cache for exact same question (skip for streaming)
        cache_key = f"{pdf_file_path}:{question.lower().strip()}"
        if not callbacks and cache_key in self._response_cache:
            cached = self._response_cache[cache_key].copy()
            cached["response_time"] = time.time() - start_time
            cached["cached"] = True
            return cached

        # Get conversation context
        conv_context = self._get_conversation_context(session_id)

        # SPEED: Fast intent detection
        intent = QueryUnderstanding.detect_intent(question, conv_context)

        # Handle greetings
        if intent == QueryIntent.GREETING:
            response = self._generate_greeting_response(conv_context)
            conv_context.update(question, response, intent)
            return {
                "result": response,
                "response_time": time.time() - start_time,
                "source_documents": [],
                "intent": intent.value
            }

        # Load vector store
        vector_store = self._load_vector_store(pdf_file_path)
        if not vector_store:
            return {
                "result": "I don't have access to this document yet. Please make sure it's been uploaded and indexed.",
                "response_time": time.time() - start_time,
                "source_documents": [],
                "intent": intent.value
            }

        # Reformulate query for better retrieval
        search_query = QueryUnderstanding.reformulate_query(question, conv_context, intent)
        logger.info(f"Search query: {search_query}")

        # Retrieve documents
        retrieved_docs = self._retrieve_documents(search_query, vector_store, intent)

        # If follow-up with no results, try original context
        if not retrieved_docs and intent == QueryIntent.FOLLOW_UP and conv_context.last_query:
            logger.info("Retrying with previous query context")
            retrieved_docs = self._retrieve_documents(conv_context.last_query, vector_store, QueryIntent.SPECIFIC_QUESTION)

        if not retrieved_docs:
            response = self._handle_no_results(question, intent)
            return {
                "result": response,
                "response_time": time.time() - start_time,
                "source_documents": [],
                "intent": intent.value
            }

        # Rerank if available
        if self.reranker:
            retrieved_docs = self._rerank_documents(question, retrieved_docs)

        # Format context
        context_text = self._format_context(retrieved_docs)
        history_text = self._format_chat_history(chat_history or [])

        # Build prompt
        prompt = PromptBuilder.build_prompt(
            question, context_text, history_text, intent, conv_context
        )

        # Generate response
        try:
            if self._use_robust_llm and hasattr(self.llm, 'invoke_with_langchain_callbacks'):
                if callbacks:
                    result_text = self.llm.invoke_with_langchain_callbacks(prompt, callbacks)
                else:
                    result_text = self.llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt, config={"callbacks": callbacks} if callbacks else None)
                result_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            result_text = f"I encountered an error generating a response. Please try again. (Error: {str(e)[:100]})"

        # SPEED: Simplified context update
        conv_context.update(question, result_text, intent)

        response = {
            "result": result_text,
            "response_time": time.time() - start_time,
            "source_documents": retrieved_docs,
            "intent": intent.value
        }

        # SPEED: Cache the response
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entries
            oldest = list(self._response_cache.keys())[:10]
            for k in oldest:
                del self._response_cache[k]
        self._response_cache[cache_key] = response.copy()

        return response

    # Utility methods
    def get_pdf_list(self) -> List[Dict[str, Any]]:
        pdfs = []
        if os.path.exists(UPLOADS_DIR):
            for fn in os.listdir(UPLOADS_DIR):
                if fn.endswith(".pdf"):
                    fp = os.path.join(UPLOADS_DIR, fn)
                    pdfs.append({
                        "filename": fn,
                        "filepath": fp,
                        "is_indexed": self.is_pdf_indexed(fp),
                        "file_size": os.path.getsize(fp),
                    })
        return sorted(pdfs, key=lambda x: x["filename"])

    def delete_pdf_index(self, pdf_file_path: str):
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        self._vector_store_cache.pop(pdf_filename, None)
        self._pdf_metadata_cache.pop(pdf_filename, None)

        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)

    def clear_cache(self):
        self._vector_store_cache.clear()
        self._pdf_metadata_cache.clear()

    def clear_conversation(self, session_id: str = "default"):
        """Clear conversation history for a session"""
        if session_id in self._conversation_contexts:
            del self._conversation_contexts[session_id]

    def rebuild_index(self, pdf_file_path: str) -> PDFMetadata:
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        self._vector_store_cache.pop(pdf_filename, None)

        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)

        return self.ingest_pdf(pdf_file_path, force_reindex=True)

    def is_pdf_indexed(self, pdf_file_path: str) -> bool:
        return os.path.exists(self._get_vector_store_path(pdf_file_path))

    def get_model_info(self) -> Dict[str, str]:
        if self._use_robust_llm and hasattr(self.llm, 'current_model'):
            return {"model": self.llm.current_model, "provider": self.llm.current_provider}
        return {"model": self.model_name, "provider": self.provider_name}

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
