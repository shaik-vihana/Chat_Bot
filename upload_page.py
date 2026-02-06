"""
Upload Page - Document Manager for TMI AI Assistant
Handles PDF upload, indexing, and management
"""

import streamlit as st
import os
import time
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def render_upload_page():
    """Render the upload and document management page"""
    st.header("Document Manager")
    st.caption("Upload and manage your PDF documents")

    # Initialize engine if needed
    if "engine" not in st.session_state:
        with st.spinner("Loading AI models..."):
            try:
                config = st.session_state.get("model_config", MODEL_CONFIG)
                st.session_state.engine = PDFQAEngine(
                    model_name=config.get("model_name", "mistral"),
                    base_url=config.get("base_url", "http://localhost:11434"),
                    provider=config.get("provider", "ollama"),
                    api_key=config.get("api_key"),
                )
            except Exception as e:
                st.error(f"Failed to initialize engine: {e}")
                st.info("Make sure Ollama is running with the required model.")
                return

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Upload section
    st.subheader("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader_page"
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join("uploads", uploaded_file.name)

            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((idx + 0.5) / len(uploaded_files))

            # Save file if not exists
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Index the PDF
            if not st.session_state.engine.is_pdf_indexed(file_path):
                try:
                    st.session_state.engine.ingest_pdf(file_path)
                    st.success(f"Processed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            else:
                st.info(f"{uploaded_file.name} already indexed.")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("All files processed!")
        time.sleep(1)
        st.rerun()

    st.markdown("---")

    # Existing PDFs section
    st.subheader("Uploaded Documents")

    try:
        pdf_files = sorted([f for f in os.listdir("uploads") if f.endswith(".pdf")])
    except FileNotFoundError:
        pdf_files = []

    if not pdf_files:
        st.info("No documents uploaded yet. Upload PDFs above to get started.")
    else:
        # Display each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join("uploads", pdf_file)
            is_indexed = st.session_state.engine.is_pdf_indexed(pdf_path)
            file_size = format_file_size(os.path.getsize(pdf_path))

            col1, col2, col3, col4 = st.columns([4, 1.5, 1, 1])

            with col1:
                status_icon = "✅" if is_indexed else "⏳"
                st.markdown(f"**{status_icon} {pdf_file}**")
                st.caption(f"Size: {file_size}")

            with col2:
                if not is_indexed:
                    if st.button("Index", key=f"idx_{pdf_file}"):
                        with st.spinner(f"Indexing {pdf_file}..."):
                            try:
                                st.session_state.engine.ingest_pdf(pdf_path)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

            with col3:
                if st.button("🔄", key=f"rb_{pdf_file}", help="Rebuild index"):
                    with st.spinner(f"Rebuilding {pdf_file}..."):
                        try:
                            st.session_state.engine.rebuild_index(pdf_path)
                            st.success("Done!")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            with col4:
                if st.button("🗑️", key=f"del_{pdf_file}", help="Delete"):
                    # Remove file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    # Remove index
                    st.session_state.engine.delete_pdf_index(pdf_path)
                    # Clear selection if this was selected
                    if st.session_state.get("current_file") == pdf_file:
                        st.session_state.current_file = None
                    st.rerun()

            st.markdown("---")

        # Batch operations
        st.subheader("Batch Operations")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Rebuild All Indexes", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                for idx, pdf_file in enumerate(pdf_files):
                    pdf_path = os.path.join("uploads", pdf_file)
                    status.text(f"Rebuilding {pdf_file}...")
                    try:
                        st.session_state.engine.rebuild_index(pdf_path)
                    except Exception as e:
                        st.warning(f"Failed to rebuild {pdf_file}: {e}")
                    progress.progress((idx + 1) / len(pdf_files))
                status.text("All indexes rebuilt!")
                time.sleep(1)
                st.rerun()

        with col2:
            if st.button("🗑️ Delete All Documents", use_container_width=True, type="secondary"):
                for pdf_file in pdf_files:
                    pdf_path = os.path.join("uploads", pdf_file)
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    st.session_state.engine.delete_pdf_index(pdf_path)
                st.session_state.current_file = None
                st.success("All documents deleted!")
                time.sleep(1)
                st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("💬 Go to Chat", use_container_width=True, type="primary"):
        st.session_state.page = "chat"
        st.rerun()
