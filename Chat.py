import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
from datetime import datetime
import os

st.set_page_config(page_title="Chat - AI Assistant", page_icon="💬", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", message_context=None):
        self.container = container
        self.text = initial_text
        self.token_count = 0
        self.message_context = message_context
        self.update_interval = 2  # Update UI every N tokens for better performance

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.token_count += 1

        if self.token_count % self.update_interval == 0 or len(token.strip()) == 0:
            self.container.markdown(self.text + "▌")

        if self.message_context is not None:
            self.message_context["content"] = self.text

    def on_llm_end(self, _response, **_kwargs) -> None:
        self.container.markdown(self.text)
        if self.message_context is not None:
            self.message_context["content"] = self.text

def format_duration(seconds):
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.2f} sec"

def render_chat_page():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .main {
        background-color: #f8fafc;
    }
    
    /* Style for inline metadata */
    .stCaption {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid #e2e8f0;
    }

    /* Page button styling */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 8px;
        background-color: #fff;
        border: 1px solid #cbd5e1;
        color: #475569;
        transition: all 0.2s ease;
        font-weight: 500;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-color: #94a3b8;
        color: #1e293b;
        background-color: #f8fafc;
    }

    /* Stop button styling - Next to input field */
    div[data-testid="stChatInput"] {
        position: relative;
    }

    /* Chat message animations */
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }
    
    /* Chat container styling */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
    }
    
    .stChatMessage[data-testid="stChatMessage"][data-test-user-name="user"] {
        background-color: #eff6ff;
        border-color: #dbeafe;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
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
    </style>
    """, unsafe_allow_html=True)

    st.title("💬 Chat with your Documents")

    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(
            model_name=MODEL_CONFIG["model_name"],
            base_url=MODEL_CONFIG["base_url"]
        )

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    try:
        uploaded_pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    except FileNotFoundError:
        uploaded_pdf_files = []

    if not uploaded_pdf_files:
        st.warning("No PDF documents found. Please go to the Admin page to upload some.")
        if st.button("Go to Admin Page"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    # Add Select All option
    pdf_options = ["Select All"] + uploaded_pdf_files

    # If there's a previously active PDF, try to keep it, otherwise default to the first one
    if "active_pdf" not in st.session_state or st.session_state.active_pdf not in pdf_options:
        st.session_state.active_pdf = pdf_options[0]
        
    st.session_state.active_pdf = st.selectbox(
        "Select a PDF to chat with:",
        pdf_options,
        index=pdf_options.index(st.session_state.active_pdf)
    )
    
    active_pdf_name = st.session_state.active_pdf
    if active_pdf_name not in st.session_state.messages:
        st.session_state.messages[active_pdf_name] = []

    # Chat Interface
    for message in st.session_state.messages[active_pdf_name]:
        role = message["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            
            # Display Metadata (Timestamp, Duration, Sources)
            meta_text = ""
            if "timestamp" in message:
                meta_text += f"🕒 {message['timestamp']} "
            
            if role == "assistant" and "response_time" in message:
                meta_text += f"| ⏱️ {format_duration(message['response_time'])}"
            
            if meta_text:
                st.caption(meta_text)

            if role == "assistant" and "source_documents" in message:
                with st.expander("📚 Reference Pages"):
                    seen_pages = set()
                    for doc in message["source_documents"]:
                        # Handle metadata safely
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        page = metadata.get("page", "Unknown")
                        # Page numbers are usually 0-indexed
                        if isinstance(page, int):
                            page += 1
                        source = os.path.basename(metadata.get("source", "Unknown"))
                        
                        page_key = f"{source}_{page}"
                        if page_key not in seen_pages:
                            st.markdown(f"- **Page {page}** ({source})")
                            seen_pages.add(page_key)

    if prompt := st.chat_input(f"Ask a question about {active_pdf_name}...", max_chars=2000):
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages[active_pdf_name].append({"role": "user", "content": prompt, "timestamp": current_time})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            st.caption(f"🕒 {current_time}")

        with st.chat_message("assistant", avatar="🤖"):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                message_placeholder = st.empty()
            with col2:
                stop_placeholder = st.empty()
            
            if stop_placeholder.button("⏹️", key=f"stop_{current_time}", help="Stop generation"):
                st.stop()
            
            st.session_state.messages[active_pdf_name].append({"role": "assistant", "content": "", "timestamp": current_time})
            current_msg_index = len(st.session_state.messages[active_pdf_name]) - 1

            try:
                stream_handler = StreamHandler(message_placeholder, message_context=st.session_state.messages[active_pdf_name][current_msg_index])
                
                message_placeholder.markdown('<span class="thinking-text">Thinking...</span>', unsafe_allow_html=True)

                if active_pdf_name == "Select All":
                    pdf_path = "ALL_PDFS"
                else:
                    pdf_path = os.path.join("uploads", active_pdf_name)

                # Get chat history (excluding current question and empty assistant placeholder)
                history = st.session_state.messages[active_pdf_name][:-2]

                response = st.session_state.engine.answer_question(prompt, pdf_file_path=pdf_path, callbacks=[stream_handler], chat_history=history)
                
                stop_placeholder.empty()
                st.session_state.messages[active_pdf_name][current_msg_index]["content"] = response["result"]
                st.session_state.messages[active_pdf_name][current_msg_index]["response_time"] = response["response_time"]
                st.session_state.messages[active_pdf_name][current_msg_index]["source_documents"] = response["source_documents"]
                
                message_placeholder.markdown(response["result"]) # Final update

                # Display metadata for the new message
                st.caption(f"🕒 {current_time} | ⏱️ {format_duration(response['response_time'])}")
                
                if response["source_documents"]:
                    with st.expander("📚 Reference Pages"):
                        seen_pages = set()
                        for doc in response["source_documents"]:
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            page = metadata.get("page", "Unknown")
                            if isinstance(page, int):
                                page += 1
                            source = os.path.basename(metadata.get("source", "Unknown"))
                            
                            page_key = f"{source}_{page}"
                            if page_key not in seen_pages:
                                st.markdown(f"- **Page {page}** ({source})")
                                seen_pages.add(page_key)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# This structure allows calling the function from the main app
def main():
    render_chat_page()

if __name__ == "__main__":
    main()