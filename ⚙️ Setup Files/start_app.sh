#!/bin/bash
# PDF Chat with Mistral 7B (Ollama)

set -e

echo "========================================================================="
echo "   PDF CHAT APPLICATION (Mistral 7B + Ollama)"
echo "========================================================================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Check/Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -r requirements.txt --quiet
echo "Installing ColPali..."
pip install colpali-engine --quiet

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama is not installed or not in PATH."
    echo "Please install Ollama from https://ollama.com"
else
    echo "Checking for Mistral model..."
    if ! ollama list | grep -q "mistral"; then
        echo "Mistral model not found. Pulling mistral..."
        ollama pull mistral
    else
        echo "Mistral model found."
    fi
fi

echo
echo "Starting Streamlit App..."
echo "Access at http://localhost:8501"
echo "========================================================================="

streamlit run app_pdf_qa.py
