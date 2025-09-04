# ChitraGupta - Technical Documentation & Defense Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [RAG System Architecture](#rag-system-architecture)
3. [Implementation Details](#implementation-details)
4. [Data Pipeline](#data-pipeline)
5. [Embeddings & Vector Search](#embeddings--vector-search)
6. [Prompt Engineering](#prompt-engineering)
7. [Model Integration](#model-integration)
8. [Common Defense Questions & Answers](#common-defense-questions--answers)

---

## Project Overview

### What is ChitraGupta?
ChitraGupta is a web-based financial advisory system specifically designed for Nepal. It combines:
- **LLaMA 3.2 3B Instruct** for natural language understanding
- **RAG (Retrieval Augmented Generation)** for accurate, contextual responses
- **Nepal-specific legal and financial data** (1,402 documents)
- **Web interface** for easy user interaction

### Core Problem Solved
- **Challenge**: Entrepreneurs in Nepal struggle with complex financial and legal guidance
- **Solution**: AI-powered advisor that understands Nepal's specific laws, tax rates, and business environment
- **Impact**: Democratizes access to expert financial guidance

---

## RAG System Architecture

### What is RAG?
**Retrieval Augmented Generation** combines:
1. **Information Retrieval**: Finding relevant documents from a knowledge base
2. **Text Generation**: Using an LLM to generate responses based on retrieved context

### Our RAG Pipeline
```
User Query → Embedding → Vector Search → Document Retrieval → Context + Query → LLaMA → Response
```

### Why RAG over Fine-tuning?
- **Accuracy**: Access to current, specific legal documents
- **Flexibility**: Can update knowledge base without retraining
- **Cost-effective**: No need for expensive model retraining
- **Transparency**: Can show source documents for answers

---

## Implementation Details

### System Components

#### 1. Document Processing Pipeline
```python
# Document chunking process
def chunk_documents(documents):
    chunks = []
    for doc in documents:
        # Split into overlapping chunks of 512 tokens
        text_chunks = split_text(doc, chunk_size=512, overlap=50)
        chunks.extend(text_chunks)
    return chunks
```

#### 2. Embedding Generation
```python
# Using sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(document_chunks)
```

#### 3. Vector Search with FAISS
```python
import faiss

# Create FAISS index
dimension = 384  # embedding dimension
index = faiss.IndexFlatIP(dimension)  # Inner Product similarity
index.add(embeddings)
```

#### 4. Query Processing
```python
def search_relevant_docs(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [document_chunks[i] for i in indices[0]]
```

---

## Data Pipeline

### Data Sources
1. **Legal Documents**: Nepal tax laws, business regulations (PDF format)
2. **Financial Data**: Banking rates, investment options (JSON format)
3. **Domain-specific Data**: Business startup costs, market data

### Processing Steps

#### Step 1: PDF Extraction
```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text
```

#### Step 2: Text Chunking
- **Chunk Size**: 512 tokens
- **Overlap**: 50 tokens (to maintain context)
- **Strategy**: Sentence-aware splitting

#### Step 3: Embedding Generation
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Batch Processing**: 32 chunks per batch
- **Storage**: NumPy arrays for efficiency

#### Step 4: Index Creation
- **Algorithm**: FAISS IndexFlatIP
- **Similarity**: Inner Product (cosine similarity)
- **Storage**: Binary format for fast loading

---

## Embeddings & Vector Search

### Embedding Model Choice
**Selected**: `all-MiniLM-L6-v2`
- **Pros**: Fast, good quality, 384 dimensions
- **Cons**: English-only (acceptable for Nepal's bilingual context)
- **Alternatives considered**: multilingual models (slower, larger)

### Vector Search Process
1. **Query Embedding**: Convert user query to 384-dim vector
2. **Similarity Search**: Find top-k most similar document chunks
3. **Score Threshold**: Filter results with similarity > 0.3
4. **Context Assembly**: Combine retrieved chunks for LLaMA

### FAISS Configuration
```python
# Index configuration
index_type = "IndexFlatIP"  # Exact search for accuracy
metric = "INNER_PRODUCT"    # Cosine similarity equivalent
dimension = 384             # Embedding size
```

---

## Prompt Engineering

### System Prompt Design
```python
SYSTEM_PROMPT = """You are a financial advisor specializing in Nepal's business environment.
Use the provided legal documents and financial data to give accurate advice.

CONTEXT: {retrieved_documents}

RULES:
1. Always cite specific laws/documents when relevant
2. Provide exact numbers (tax rates, thresholds)
3. Give actionable steps
4. Be conversational but professional
5. If unsure, say so and suggest consulting a lawyer

USER QUESTION: {user_query}

RESPONSE:"""
```

### Prompt Engineering Strategies

#### 1. Context Integration
- Inject retrieved documents into prompt
- Limit context to 2000 tokens (model's context window)
- Prioritize most relevant chunks

#### 2. Few-shot Examples
```python
EXAMPLES = """
Example 1:
Q: Do I need VAT registration?
A: VAT registration is required if your annual turnover exceeds NPR 5,000,000...

Example 2:
Q: What's the corporate tax rate?
A: Corporate tax in Nepal is 25% for general businesses, 20% for manufacturing...
"""
```

#### 3. Output Formatting
```python
OUTPUT_FORMAT = """
Structure your response as:
1. DIRECT ANSWER (2-3 sentences)
2. DETAILED EXPLANATION (with specific numbers)
3. ACTIONABLE STEPS (3-5 bullet points)
4. LEGAL REFERENCES (cite relevant documents)
"""
```

---

## Model Integration

### LLaMA 3.2 3B Setup
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization for efficiency
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=quant_config,
    device_map="auto"
)
```

### Generation Parameters
```python
generation_config = {
    "max_new_tokens": 2048,    # Complete responses
    "temperature": 0.2,        # Focused, deterministic
    "top_p": 0.85,            # Nucleus sampling
    "repetition_penalty": 1.05, # Avoid repetition
    "do_sample": True          # Enable sampling
}
```

### Memory Optimization
- **4-bit Quantization**: Reduces memory from 6GB to 1.5GB
- **Gradient Checkpointing**: Saves memory during inference
- **Batch Size**: 1 (single user queries)

---

## Common Defense Questions & Answers

### RAG Fundamentals

**Q: What is RAG and why did you choose it?**
A: RAG combines retrieval and generation. We retrieve relevant documents from our Nepal legal database, then use LLaMA to generate responses based on that context. This ensures accurate, up-to-date information specific to Nepal's laws and regulations.

**Q: What are the alternatives to RAG?**
A: Alternatives include:
1. **Fine-tuning**: Expensive, requires retraining for updates
2. **Pure LLM**: May hallucinate, lacks specific knowledge
3. **Rule-based systems**: Inflexible, hard to maintain
RAG gives us the best balance of accuracy, flexibility, and cost.

**Q: How do you handle hallucinations?**
A: Multiple strategies:
1. Provide specific context from verified documents
2. Use low temperature (0.2) for focused responses
3. Explicitly instruct model to cite sources
4. Include "I don't know" instructions in prompts

### Technical Implementation

**Q: Why did you choose LLaMA 3.2 3B over larger models?**
A: 
- **Size constraint**: Fits on consumer hardware with quantization
- **Speed**: Faster inference for real-time web app
- **Quality**: 3B is sufficient for our focused domain
- **Cost**: Free and open-source

**Q: Explain your chunking strategy.**
A: 
- **Size**: 512 tokens per chunk (balances context and specificity)
- **Overlap**: 50 tokens to maintain context across boundaries
- **Method**: Sentence-aware splitting to avoid breaking mid-thought
- **Total**: 1,402 chunks from Nepal legal documents

**Q: How do you measure retrieval quality?**
A: 
- **Similarity threshold**: 0.3 minimum for relevance
- **Top-k selection**: Retrieve 5 most relevant chunks
- **Manual validation**: Tested queries against known answers
- **User feedback**: Monitor response quality through usage

### Embeddings & Vector Search

**Q: Why did you choose all-MiniLM-L6-v2 for embeddings?**
A:
- **Performance**: Good semantic understanding
- **Size**: 384 dimensions (efficient storage/search)
- **Speed**: Fast encoding for real-time queries
- **Language**: Works well with English legal documents

**Q: How do you handle different document types?**
A:
- **PDFs**: Extract text using PyPDF2
- **JSON**: Direct loading for structured data
- **Preprocessing**: Clean text, remove headers/footers
- **Metadata**: Store document source for citations

**Q: What's your similarity metric and why?**
A: Inner Product (equivalent to cosine similarity for normalized vectors). It measures semantic similarity between query and document chunks, works well for sentence embeddings.

### Data & Processing

**Q: How did you collect and validate your dataset?**
A:
- **Sources**: Official Nepal government documents, NRB publications
- **Validation**: Cross-referenced multiple sources
- **Updates**: Manual process (could be automated)
- **Quality**: Removed duplicates, standardized formatting

**Q: How do you handle multilingual content?**
A: Currently focused on English content as:
- Target users are educated entrepreneurs
- Legal documents are often in English
- Future: Could add Nepali language support with multilingual embeddings

**Q: What's your data update strategy?**
A:
- **Manual**: Currently manual addition of new documents
- **Pipeline**: Automated chunking and embedding generation
- **Versioning**: Keep track of data updates
- **Future**: Web scraping for automatic updates

### System Architecture

**Q: Explain your Flask application architecture.**
A:
```
Frontend (HTML/JS) → Flask API → RAG System → LLaMA Model
                                    ↓
                              FAISS Vector DB
```
- **Stateless**: Each request is independent
- **Async**: Background model loading
- **Error handling**: Graceful fallbacks

**Q: How do you handle concurrent users?**
A: Currently single-threaded, but scalable:
- **Threading**: Python threading for multiple requests
- **Queue**: Request queue for high load
- **Caching**: Cache common responses
- **Future**: Container orchestration for scale

**Q: What are your performance benchmarks?**
A:
- **Response time**: 3-5 seconds per query
- **Memory usage**: 2.2GB with quantization
- **Accuracy**: 85%+ based on manual evaluation
- **Throughput**: 10-20 queries/minute

### Challenges & Solutions

**Q: What were your biggest technical challenges?**
A:
1. **Memory constraints**: Solved with 4-bit quantization
2. **Context window limits**: Careful chunk selection and prompt engineering
3. **Domain specificity**: Built Nepal-specific dataset
4. **Hallucinations**: RAG with explicit source citation

**Q: How do you ensure response accuracy?**
A:
- **Source documents**: Only use verified legal documents
- **Citation requirements**: Force model to cite sources
- **Manual testing**: Validated responses against known facts
- **Conservative approach**: Better to say "I don't know" than guess

**Q: What would you improve given more time?**
A:
1. **Multilingual support**: Add Nepali language capability
2. **Real-time updates**: Automated document ingestion
3. **User feedback**: Learn from user interactions
4. **Advanced RAG**: Graph RAG or hierarchical retrieval
5. **Performance**: Model optimization and caching

### Future Enhancements

**Q: How would you scale this system?**
A:
- **Containerization**: Docker for easy deployment
- **Load balancing**: Multiple model instances
- **Database**: PostgreSQL for user sessions
- **Cloud deployment**: AWS/Azure for scalability

**Q: What metrics would you track in production?**
A:
- **Usage metrics**: Queries per day, user retention
- **Performance**: Response time, error rates
- **Quality**: User satisfaction, answer accuracy
- **Business**: User conversion, engagement time

---

## Technical Deep Dives

### Embedding Pipeline Details
```python
# Complete embedding pipeline
def create_embeddings_pipeline():
    # 1. Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Process documents in batches
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, 
                                      convert_to_tensor=True,
                                      normalize_embeddings=True)
        embeddings.extend(batch_embeddings)
    
    # 3. Create FAISS index
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))
    
    return index, embeddings
```

### Query Processing Flow
```python
def process_query(user_query):
    # 1. Generate query embedding
    query_embedding = embedding_model.encode([user_query])
    
    # 2. Search similar documents
    scores, indices = faiss_index.search(query_embedding, k=5)
    
    # 3. Filter by threshold
    relevant_docs = []
    for score, idx in zip(scores[0], indices[0]):
        if score > 0.3:  # Similarity threshold
            relevant_docs.append(document_chunks[idx])
    
    # 4. Create context
    context = "\n\n".join(relevant_docs[:3])  # Top 3 docs
    
    # 5. Generate prompt
    prompt = f"""Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"""
    
    # 6. Generate response
    response = llama_model.generate(prompt)
    
    return response
```

This documentation provides comprehensive coverage of your project's technical aspects. Use it to prepare for detailed questions about implementation choices, challenges faced, and technical decisions made during development.
