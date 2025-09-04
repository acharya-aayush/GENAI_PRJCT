# ChitraGupta
A web-based financial advisory system for Nepal using LLaMA AI and RAG (Retrieval Augmented Generation) technology.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Access the web interface:
   ```
   http://localhost:5000
   ```

## Project Structure

```
GENAI_PRJCT/
├── app.py                          # Flask web application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── src/
│   └── ultra_premium_advisor.py    # Core AI advisor engine
├── data/processed/
│   ├── document_chunks.json        # Legal document chunks (1,402 Nepal documents)
│   ├── financialspecificdata.json  # Domain-specific financial data
│   ├── embeddings.npy              # Vector embeddings
│   └── faiss_index.bin             # FAISS search index
├── templates/
│   └── index.html                  # Web interface
└── static/                         # Static assets
```

## Technical Stack

- Backend: Flask + Python 3.10+
- AI Model: LLaMA 3.2 3B Instruct (4-bit quantized)
- Vector Search: FAISS
- Frontend: HTML5/CSS3/JavaScript
- Data: 1,402 processed Nepal legal and financial documents

## Features

- Conversational AI for financial guidance
- Nepal-specific business and tax advice
- RAG system with legal document integration
- Real-time web interface
- Memory-optimized with 4-bit quantization

## Team

Created by **Aayush Acharya**, **Nidhi Pradhan**, and **Suravi Paudel** as a project for the Gen AI workshop.

**Mentor:** Er. Sujan Sharma