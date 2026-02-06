# PDF Chat with Mistral 7B

A local PDF chat application using **Mistral 7B**, **Ollama**, **LangChain**, and **Streamlit**.

## Features

- 📄 **Local PDF Processing**: Ingests PDFs securely on your machine.
- 🤖 **Mistral 7B**: Uses the powerful open-source Mistral model via Ollama.
- 💬 **Chat Interface**: Clean, interactive chat UI built with Streamlit.
- 🔍 **RAG Architecture**: Retrieves relevant context from your document to answer questions accurately.

## Prerequisites

1. **Python 3.9+**
2. **Ollama**: [Download here](https://ollama.com)
3. **Mistral Model**: Run `ollama pull mistral`

## Quick Start

1. **Run Setup**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   *This will install dependencies and clean up old files.*

2. **Start App**:
   ```bash
   source venv/bin/activate
   streamlit run app_pdf_qa.py
   ```

3. **Use**:
   - Open browser at `http://localhost:8501`
   - Upload a PDF in the sidebar
   - Start chatting!

## Configuration

Edit `model_config.py` to change the model or base URL:

```python
MODEL_CONFIG = {
    "model_name": "mistral",
    "base_url": "http://localhost:11434",
}
```
