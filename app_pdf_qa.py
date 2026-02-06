"""
TMI AI Assistant - Modern PDF Q&A Interface
Features: Multiple PDFs, streaming responses, model selection, robust UI
"""

import streamlit as st
import os
import time
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import (
    MODEL_CONFIG,
    PROVIDER_NAMES,
    AVAILABLE_MODELS,
    PROVIDER_OLLAMA,
)

# Page config
st.set_page_config(
    page_title="TMI AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Chat container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }

    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }

    .assistant-message {
        background: white;
        color: #333;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }

    /* Thinking animation */
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    .thinking-text {
        animation: pulse 1.5s infinite ease-in-out;
        color: #666;
        font-style: italic;
    }

    /* Cursor animation for streaming */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    .cursor {
        animation: blink 1s infinite;
        font-weight: bold;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }

    /* PDF card */
    .pdf-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        transition: all 0.2s ease;
    }

    .pdf-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-indexed {
        background: #d4edda;
        color: #155724;
    }

    .status-pending {
        background: #fff3cd;
        color: #856404;
    }

    /* Model info card */
    .model-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }

    /* Response time badge */
    .response-time {
        background: #e8f4f8;
        color: #0c5460;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    /* Reference expander */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 0.6rem 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }
    </style>
    """, unsafe_allow_html=True)


class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses"""

    def __init__(self, container, initial_text="", message_context=None):
        self.container = container
        self.text = initial_text
        self.token_count = 0
        self.message_context = message_context
        self.update_interval = 2
        self._stopped = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._stopped:
            return
        self.text += token
        self.token_count += 1
        if self.token_count % self.update_interval == 0 or len(token.strip()) == 0:
            self.container.markdown(self.text + " ▌")
        if self.message_context is not None:
            self.message_context["content"] = self.text

    def on_llm_end(self, _response, **_kwargs) -> None:
        if not self._stopped:
            self.container.markdown(self.text)
        if self.message_context is not None:
            self.message_context["content"] = self.text

    def stop(self):
        self._stopped = True


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.1f}s"


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def init_session_state():
    """Initialize session state variables"""
    import uuid
    defaults = {
        "messages": [],
        "engine": None,
        "current_file": None,
        "page": "chat",
        "model_config": MODEL_CONFIG.copy(),
        "is_processing": False,
        "stop_generation": False,
        "session_id": str(uuid.uuid4())[:8],  # Unique session for conversation tracking
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_engine():
    """Initialize or get the PDF QA engine"""
    if st.session_state.engine is None:
        config = st.session_state.model_config
        with st.spinner("Loading AI models..."):
            try:
                st.session_state.engine = PDFQAEngine(
                    model_name=config.get("model_name", "mistral"),
                    base_url=config.get("base_url", "http://localhost:11434"),
                    provider=config.get("provider", "ollama"),
                    api_key=config.get("api_key"),
                    temperature=config.get("temperature", 0.7),
                )
                return True
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.info("Make sure Ollama is running with the required model.")
                return False
    return True


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>📄 TMI AI Assistant</h1>
        <p>Chat with your PDF documents using AI</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        st.markdown("### Navigation")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.page == "chat" else "secondary"):
                st.session_state.page = "chat"
                st.rerun()
        with col2:
            if st.button("📁 Upload", use_container_width=True, type="primary" if st.session_state.page == "upload" else "secondary"):
                st.session_state.page = "upload"
                st.rerun()

        st.markdown("---")

        # PDF Selection
        st.markdown("### Select Document")

        os.makedirs("uploads", exist_ok=True)
        pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]

        if not pdf_files:
            st.info("No PDFs uploaded yet. Go to Upload page to add documents.")
            options = ["No PDF available"]
        else:
            options = ["Select a PDF...", "📚 All PDFs"] + pdf_files

        selected = st.selectbox(
            "Choose document",
            options=options,
            key="pdf_selector",
            label_visibility="collapsed",
        )

        # Handle selection
        if selected and selected not in ["Select a PDF...", "No PDF available"]:
            new_file = "Select All" if selected == "📚 All PDFs" else selected

            if st.session_state.current_file != new_file:
                st.session_state.current_file = new_file
                st.session_state.messages = []

                # Auto-index if needed
                if new_file != "Select All":
                    pdf_path = os.path.join("uploads", new_file)
                    if st.session_state.engine and not st.session_state.engine.is_pdf_indexed(pdf_path):
                        with st.spinner(f"Indexing {new_file}..."):
                            try:
                                st.session_state.engine.ingest_pdf(pdf_path)
                                st.success(f"Indexed {new_file}")
                            except Exception as e:
                                st.error(f"Failed to index: {e}")

        # Show selected PDF info
        if st.session_state.current_file:
            if st.session_state.current_file == "Select All":
                st.success(f"📚 Chatting with all {len(pdf_files)} PDFs")
            else:
                filepath = os.path.join("uploads", st.session_state.current_file)
                if os.path.exists(filepath):
                    size = format_file_size(os.path.getsize(filepath))
                    is_indexed = st.session_state.engine.is_pdf_indexed(filepath) if st.session_state.engine else False
                    status = "✅ Indexed" if is_indexed else "⏳ Not indexed"
                    st.markdown(f"""
                    <div class="model-info">
                        <strong>{st.session_state.current_file}</strong><br>
                        <small>{size} | {status}</small>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Model info
        st.markdown("### Model")
        if st.session_state.engine:
            model_info = st.session_state.engine.get_model_info()
            provider_display = PROVIDER_NAMES.get(model_info.get("provider", ""), model_info.get("provider", ""))
            st.markdown(f"""
            <div class="model-info">
                <strong>{model_info.get('model', 'Unknown')}</strong><br>
                <small>{provider_display}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Engine not initialized")

        # Settings button
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "settings"
            st.rerun()

        st.markdown("---")

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                # Clear conversation context in engine for fresh start
                if st.session_state.engine:
                    st.session_state.engine.clear_conversation(st.session_state.session_id)
                st.rerun()


def render_chat_message(message, idx):
    """Render a single chat message"""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # Metadata
        meta_parts = []
        if "timestamp" in message:
            meta_parts.append(f"🕒 {message['timestamp']}")
        if role == "assistant" and "response_time" in message:
            meta_parts.append(f"⏱️ {format_duration(message['response_time'])}")

        if meta_parts:
            st.caption(" | ".join(meta_parts))

        # Reference pages
        if role == "assistant" and "source_documents" in message and message["source_documents"]:
            with st.expander("📚 Reference Pages", expanded=False):
                seen_pages = set()
                for doc in message["source_documents"]:
                    metadata = doc.metadata if hasattr(doc, "metadata") else {}
                    page = metadata.get("page", "?")
                    if isinstance(page, int):
                        page += 1
                    source = os.path.basename(metadata.get("source", "Unknown"))

                    page_key = f"{source}_{page}"
                    if page_key not in seen_pages:
                        st.markdown(f"- **Page {page}** ({source})")
                        seen_pages.add(page_key)


def render_chat_page():
    """Render the main chat page"""
    render_header()

    # Check if PDF is selected
    if not st.session_state.current_file:
        st.info("👈 Select a PDF from the sidebar to start chatting")
        return

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        render_chat_message(message, idx)

    # Chat input
    if prompt := st.chat_input("Ask a question about your document...", key="chat_input"):
        current_time = datetime.now().strftime("%H:%M:%S")

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": current_time
        })

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {current_time}")

        # Generate response
        with st.chat_message("assistant"):
            col1, col2 = st.columns([0.92, 0.08])

            with col1:
                message_placeholder = st.empty()
            with col2:
                stop_btn = st.button("⏹️", key=f"stop_{current_time}", help="Stop generation")

            if stop_btn:
                st.session_state.stop_generation = True

            # Pre-append assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "timestamp": current_time
            })
            current_msg_index = len(st.session_state.messages) - 1

            try:
                # Show thinking indicator
                message_placeholder.markdown('<span class="thinking-text">Thinking...</span>', unsafe_allow_html=True)

                # Setup stream handler
                stream_handler = StreamHandler(
                    message_placeholder,
                    message_context=st.session_state.messages[current_msg_index]
                )

                # Determine PDF path
                if st.session_state.current_file == "Select All":
                    pdf_path = "ALL_PDFS"
                else:
                    pdf_path = os.path.join("uploads", st.session_state.current_file)

                # Get chat history (excluding current)
                history = st.session_state.messages[:-2]

                # Get response with session tracking for conversation memory
                response = st.session_state.engine.answer_question(
                    prompt,
                    pdf_file_path=pdf_path,
                    callbacks=[stream_handler],
                    chat_history=history,
                    session_id=st.session_state.session_id,
                )

                # Check if stopped
                if st.session_state.stop_generation:
                    st.session_state.stop_generation = False
                    stream_handler.stop()

                full_response = response.get("result", "")

                # Update message state
                st.session_state.messages[current_msg_index].update({
                    "content": full_response,
                    "response_time": response.get("response_time", 0),
                    "source_documents": response.get("source_documents", []),
                    "intent": response.get("intent", "unknown"),
                })

                # Final render
                message_placeholder.markdown(full_response)

                # Show metadata
                st.caption(f"🕒 {current_time} | ⏱️ {format_duration(response['response_time'])}")

                # Show references
                if response.get("source_documents"):
                    with st.expander("📚 Reference Pages", expanded=False):
                        seen_pages = set()
                        for doc in response["source_documents"]:
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            page = metadata.get("page", "?")
                            if isinstance(page, int):
                                page += 1
                            source = os.path.basename(metadata.get("source", "Unknown"))

                            page_key = f"{source}_{page}"
                            if page_key not in seen_pages:
                                st.markdown(f"- **Page {page}** ({source})")
                                seen_pages.add(page_key)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.pop()


def render_upload_page():
    """Render the upload page"""
    st.markdown("""
    <div class="main-header">
        <h1>📁 Document Manager</h1>
        <p>Upload and manage your PDF documents</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("### Upload PDFs")

    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join("uploads", uploaded_file.name)

            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((idx + 0.5) / len(uploaded_files))

            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Index the PDF
            if st.session_state.engine:
                if not st.session_state.engine.is_pdf_indexed(file_path):
                    try:
                        st.session_state.engine.ingest_pdf(file_path)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("All files processed!")
        time.sleep(1)
        st.rerun()

    st.markdown("---")

    # Existing PDFs
    st.markdown("### Uploaded Documents")

    pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")] if os.path.exists("uploads") else []

    if not pdf_files:
        st.info("No documents uploaded yet.")
    else:
        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join("uploads", pdf_file)
            is_indexed = st.session_state.engine.is_pdf_indexed(pdf_path) if st.session_state.engine else False
            file_size = format_file_size(os.path.getsize(pdf_path))

            col1, col2, col3, col4 = st.columns([4, 1.5, 1, 1])

            with col1:
                status = "✅" if is_indexed else "⏳"
                st.markdown(f"**{status} {pdf_file}**")
                st.caption(f"Size: {file_size}")

            with col2:
                if not is_indexed:
                    if st.button("Index", key=f"index_{pdf_file}"):
                        with st.spinner(f"Indexing {pdf_file}..."):
                            try:
                                st.session_state.engine.ingest_pdf(pdf_path)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

            with col3:
                if st.button("🔄", key=f"rebuild_{pdf_file}", help="Rebuild index"):
                    with st.spinner(f"Rebuilding {pdf_file}..."):
                        try:
                            st.session_state.engine.rebuild_index(pdf_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            with col4:
                if st.button("🗑️", key=f"delete_{pdf_file}", help="Delete"):
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    if st.session_state.engine:
                        st.session_state.engine.delete_pdf_index(pdf_path)
                    if st.session_state.current_file == pdf_file:
                        st.session_state.current_file = None
                    st.rerun()

            st.markdown("---")

        # Batch operations
        st.markdown("### Batch Operations")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Rebuild All Indexes", use_container_width=True):
                progress = st.progress(0)
                for idx, pdf_file in enumerate(pdf_files):
                    pdf_path = os.path.join("uploads", pdf_file)
                    try:
                        st.session_state.engine.rebuild_index(pdf_path)
                    except Exception as e:
                        st.warning(f"Failed: {pdf_file}")
                    progress.progress((idx + 1) / len(pdf_files))
                st.success("All indexes rebuilt!")
                time.sleep(1)
                st.rerun()

        with col2:
            if st.button("🗑️ Delete All", use_container_width=True, type="secondary"):
                for pdf_file in pdf_files:
                    pdf_path = os.path.join("uploads", pdf_file)
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    if st.session_state.engine:
                        st.session_state.engine.delete_pdf_index(pdf_path)
                st.session_state.current_file = None
                st.rerun()


def render_settings_page():
    """Render the settings page"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Settings</h1>
        <p>Configure your AI assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Configuration")

    config = st.session_state.model_config

    # Provider selection
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_NAMES.keys()),
        format_func=lambda x: PROVIDER_NAMES.get(x, x),
        index=list(PROVIDER_NAMES.keys()).index(config.get("provider", PROVIDER_OLLAMA)),
    )

    # Model selection based on provider
    available_models = AVAILABLE_MODELS.get(provider, ["mistral"])
    current_model = config.get("model_name", available_models[0])
    if current_model not in available_models:
        current_model = available_models[0]

    model_name = st.selectbox(
        "Model",
        options=available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
    )

    # API Key for non-Ollama providers
    api_key = None
    if provider != PROVIDER_OLLAMA:
        api_key = st.text_input(
            "API Key",
            type="password",
            value=config.get("api_key", ""),
            help=f"Enter your {PROVIDER_NAMES.get(provider, provider)} API key"
        )

    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config.get("temperature", 0.7),
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )

    # Base URL for Ollama
    base_url = config.get("base_url", "http://localhost:11434")
    if provider == PROVIDER_OLLAMA:
        base_url = st.text_input(
            "Ollama URL",
            value=base_url,
            help="URL of your Ollama server"
        )

    st.markdown("---")

    # Save button
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save & Apply", use_container_width=True, type="primary"):
            # Update config
            st.session_state.model_config = {
                "provider": provider,
                "model_name": model_name,
                "api_key": api_key,
                "temperature": temperature,
                "base_url": base_url,
            }

            # Reinitialize engine
            st.session_state.engine = None

            with st.spinner("Applying new configuration..."):
                try:
                    st.session_state.engine = PDFQAEngine(
                        model_name=model_name,
                        base_url=base_url,
                        provider=provider,
                        api_key=api_key,
                        temperature=temperature,
                    )
                    st.success("Configuration saved!")
                    time.sleep(1)
                    st.session_state.page = "chat"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to apply configuration: {e}")

    with col2:
        if st.button("↩️ Back to Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


def main():
    """Main application entry point"""
    load_custom_css()
    init_session_state()

    # Initialize engine
    if not init_engine():
        return

    # Render sidebar
    render_sidebar()

    # Render current page
    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "upload":
        render_upload_page()
    elif st.session_state.page == "settings":
        render_settings_page()


if __name__ == "__main__":
    main()
