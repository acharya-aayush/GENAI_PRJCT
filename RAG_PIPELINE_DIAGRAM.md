# 🔄 RAG Pipeline Block Diagram & Detailed Explanation
## ChitraGupta RAG (Retrieval-Augmented Generation) System

---

## 📊 COMPLETE RAG PIPELINE BLOCK DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CHITRAGUPTA RAG PIPELINE FLOW                            │
│                    Query → Retrieve → Augment → Generate                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │    │                 │
│   USER INPUT    │───►│ QUERY EMBEDDING │───►│  FAISS SEARCH   │───►│ CONTEXT FUSION  │
│                 │    │                 │    │                 │    │                 │
│ "What is VAT    │    │ Transform text  │    │ Find similar    │    │ Merge retrieved │
│  threshold      │    │ to 384-dim      │    │ documents from  │    │ docs with user  │
│  in Nepal?"     │    │ vector using    │    │ knowledge base  │    │ query & domain  │
│                 │    │ sentence-BERT   │    │ using cosine    │    │ specific data   │
│                 │    │ transformer     │    │ similarity      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │                        │
        │                        │                        │                        │
        ▼                        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Raw Text Query  │    │[0.12, -0.45,    │    │ Top-5 Documents │    │ Enhanced Prompt │
│ Natural Language│    │ 0.78, 0.23,     │    │ Similarity:     │    │ System + Legal  │
│ Business Context│    │ -0.91, ...]     │    │ [0.89, 0.85,    │    │ Docs + Domain   │
│ Intent Detection│    │ (384 dimensions) │    │  0.82, 0.79,    │    │ Data + Query    │
│                 │    │ Normalized Vec  │    │  0.76]          │    │ (~1800 tokens)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI GENERATION                                      │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   LLaMA 3.2     │───►│   RESPONSE      │───►│   STRUCTURED    │             │
│  │   PROCESSING    │    │   GENERATION    │    │    OUTPUT       │             │
│  │                 │    │                 │    │                 │             │
│  │ • Enhanced      │    │ • 1024 tokens   │    │ • Business      │             │
│  │   context       │    │ • Temperature   │    │   advice        │             │
│  │ • Nepal legal   │    │   0.2 focused   │    │ • Legal         │             │
│  │   grounding     │    │ • Factual       │    │   citations     │             │
│  │ • Domain data   │    │   accuracy      │    │ • Source        │             │
│  │ • 4-bit quant   │    │ • 30-60 sec     │    │   attribution   │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               FINAL OUTPUT                                      │
│                                                                                 │
│  "In Nepal, the VAT registration threshold is Rs. 50 lakh annually.            │
│   Manufacturing companies with turnover exceeding this amount must register    │
│   within 30 days. The current VAT rate is 13% as per Nepal Income Tax         │
│   Rules 2059, Article 15..."                                                   │
│                                                                                 │
│  Sources: [Nepal Income Tax Rules 2059 - Article 15, Schedule 1]               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 STEP-BY-STEP RAG PIPELINE BREAKDOWN

### **STEP 1: QUERY EMBEDDING** 
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           QUERY EMBEDDING PROCESS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: "What is VAT threshold in Nepal?"                                       │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    TEXT PREPROCESSING                                   │   │
│  │                                                                         │   │
│  │  ├── Tokenization: ["What", "is", "VAT", "threshold", "in", "Nepal"]   │   │
│  │  ├── Normalization: Convert to lowercase, remove special chars         │   │
│  │  ├── Context Addition: Add business query markers                      │   │
│  │  └── Intent Detection: Classify as "tax_inquiry" + "threshold_question"│   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   SENTENCE-BERT ENCODING                                │   │
│  │                                                                         │   │
│  │  Model: sentence-transformers/all-MiniLM-L6-v2                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Input Text → Transformer Layers → Pooling → L2 Normalization   │    │   │
│  │  │                                                                 │    │   │
│  │  │ "What is VAT threshold in Nepal?" →                            │    │   │
│  │  │ [0.123, -0.456, 0.789, 0.234, -0.912, 0.567, ...]            │    │   │
│  │  │ (384 floating-point dimensions)                                │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  Key Properties:                                                        │   │
│  │  ├── Semantic Understanding: Captures meaning beyond keywords          │   │
│  │  ├── Business Context: Understands financial/legal terminology        │   │
│  │  ├── Language Agnostic: Works with Nepal-specific terms               │   │
│  │  └── Consistent: Same query always produces same embedding             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  OUTPUT: Dense vector [0.123, -0.456, 0.789, ...] (384 dimensions)             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters:**
- **Semantic Search**: Finds documents about "VAT registration limits" even when query says "threshold"
- **Context Preservation**: Maintains business and legal context in the vector space
- **Efficiency**: 384 dimensions balance accuracy with computational speed
- **Consistency**: Identical queries always retrieve same relevant documents

---

### **STEP 2: FAISS SEARCH**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FAISS SIMILARITY SEARCH                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: Query Vector [0.123, -0.456, 0.789, ...] (384 dims)                    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      KNOWLEDGE BASE INDEX                               │   │
│  │                                                                         │   │
│  │  FAISS Index Structure:                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Index Type: IndexFlatIP (Inner Product for cosine similarity)  │    │   │
│  │  │ Vector Count: 1,402 document embeddings                        │    │   │
│  │  │ Dimensions: 384 per vector                                      │    │   │
│  │  │ Memory Usage: ~2.1 MB total index size                         │    │   │
│  │  │ Search Time: <1 millisecond per query                          │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  Document Examples in Index:                                            │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Doc 1: "VAT registration threshold Rs 50 lakh..."              │    │   │
│  │  │        Vector: [0.145, -0.423, 0.812, ...]                     │    │   │
│  │  │                                                                 │    │   │
│  │  │ Doc 2: "Annual turnover exceeding fifty lakh rupees..."        │    │   │
│  │  │        Vector: [0.134, -0.445, 0.798, ...]                     │    │   │
│  │  │                                                                 │    │   │
│  │  │ Doc 3: "Manufacturing industry VAT requirements..."             │    │   │
│  │  │        Vector: [0.156, -0.412, 0.823, ...]                     │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       SIMILARITY COMPUTATION                            │   │
│  │                                                                         │   │
│  │  Cosine Similarity Calculation:                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ For each document vector Di in knowledge base:                 │    │   │
│  │  │                                                                 │    │   │
│  │  │ similarity = (Query • Di) / (||Query|| × ||Di||)               │    │   │
│  │  │                                                                 │    │   │
│  │  │ Where:                                                          │    │   │
│  │  │ • Query = [0.123, -0.456, 0.789, ...]                         │    │   │
│  │  │ • Di = document i's embedding vector                           │    │   │
│  │  │ • • = dot product                                              │    │   │
│  │  │ • ||x|| = vector magnitude                                     │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  Top-K Selection (K=5):                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Rank all 1,402 documents by similarity score                   │    │   │
│  │  │ Select top 5 most relevant documents                           │    │   │
│  │  │ Return document IDs and similarity scores                      │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  OUTPUT: Top-5 Document Results                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Doc_ID: 234, Score: 0.89 - "VAT registration threshold Rs 50 lakh"  │   │
│  │ 2. Doc_ID: 567, Score: 0.85 - "Annual turnover exceeding fifty lakh"   │   │
│  │ 3. Doc_ID: 123, Score: 0.82 - "Business registration requirements"     │   │
│  │ 4. Doc_ID: 789, Score: 0.79 - "Tax compliance for businesses"          │   │
│  │ 5. Doc_ID: 345, Score: 0.76 - "VAT rates and regulations Nepal"        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**FAISS Advantages:**
- **Speed**: Sub-millisecond search across 1,402 documents
- **Accuracy**: Semantic similarity finds conceptually related content
- **Scalability**: Can handle millions of documents with consistent performance
- **Memory Efficiency**: Optimized index structures for production deployment

---

### **STEP 3: CONTEXT FUSION**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CONTEXT FUSION PROCESS                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUTS TO FUSION:                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐│
│  │   RETRIEVED     │  │   DOMAIN        │  │ CONVERSATION    │  │   USER      ││
│  │  LEGAL DOCS     │  │   SPECIFIC      │  │    CONTEXT      │  │   QUERY     ││
│  │                 │  │    DATA         │  │                 │  │             ││
│  │ • Top-5 Nepal   │  │ • Business      │  │ • Chat history  │  │ • Original  ││
│  │   tax documents │  │   sectors       │  │ • User profile  │  │   question  ││
│  │ • Legal text    │  │ • Financial     │  │ • Session state │  │ • Intent    ││
│  │ • Citations     │  │   metrics       │  │ • Preferences   │  │ • Context   ││
│  │ • Metadata      │  │ • Current rates │  │                 │  │             ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘│
│           │                     │                     │                  │      │
│           └─────────────────────┼─────────────────────┼──────────────────┘      │
│                                 │                     │                         │
│                                 ▼                     ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        INTELLIGENT MERGING                              │   │
│  │                                                                         │   │
│  │  Context Assembly Strategy:                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. System Prompt (Base Instructions)                           │    │   │
│  │  │    ├── "You are ChitraGupta Financial Advisor"                 │    │   │
│  │  │    ├── "Provide accurate Nepal business guidance"               │    │   │
│  │  │    └── "Use provided legal documents as authority"              │    │   │
│  │  │                                                                 │    │   │
│  │  │ 2. Legal Document Context                                       │    │   │
│  │  │    ├── Relevant sections from Nepal Income Tax Rules           │    │   │
│  │  │    ├── Specific articles and schedules                         │    │   │
│  │  │    └── Citation information for source attribution             │    │   │
│  │  │                                                                 │    │   │
│  │  │ 3. Domain-Specific Business Data                               │    │   │
│  │  │    ├── Industry benchmarks and metrics                         │    │   │
│  │  │    ├── Current Nepal tax rates and thresholds                  │    │   │
│  │  │    └── Business context relevant to query                      │    │   │
│  │  │                                                                 │    │   │
│  │  │ 4. Conversation Context                                         │    │   │
│  │  │    ├── Previous questions and answers                          │    │   │
│  │  │    ├── User's business type and stage                          │    │   │
│  │  │    └── Personalization preferences                             │    │   │
│  │  │                                                                 │    │   │
│  │  │ 5. Current User Query                                           │    │   │
│  │  │    ├── Original question with intent                           │    │   │
│  │  │    ├── Extracted entities and context                          │    │   │
│  │  │    └── Expected response format                                 │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  Token Management:                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ • Total Context Limit: 2000 tokens (LLaMA input constraint)    │    │   │
│  │  │ • System Prompt: ~200 tokens                                   │    │   │
│  │  │ • Legal Documents: ~1200 tokens (prioritized by relevance)     │    │   │
│  │  │ • Domain Data: ~200 tokens (targeted to query)                 │    │   │
│  │  │ • Conversation: ~200 tokens (recent relevant history)          │    │   │
│  │  │ • User Query: ~200 tokens (with intent markup)                 │    │   │
│  │  │ • Buffer: ~200 tokens (safety margin)                          │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  OUTPUT: Enhanced Prompt for LLaMA                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ "You are ChitraGupta Financial Advisor specializing in Nepal business   │   │
│  │ guidance. Use the following legal documents and data to provide accurate │   │
│  │ advice:                                                                  │   │
│  │                                                                         │   │
│  │ LEGAL CONTEXT:                                                          │   │
│  │ Article 15 of Nepal Income Tax Rules 2059 states: 'Every person whose  │   │
│  │ annual turnover exceeds fifty lakh rupees shall register for VAT       │   │
│  │ within thirty days...'                                                  │   │
│  │                                                                         │   │
│  │ BUSINESS DATA:                                                          │   │
│  │ Current VAT rate: 13%, Registration fee: Rs. 500, Threshold: Rs. 50L   │   │
│  │                                                                         │   │
│  │ USER QUESTION: What is VAT threshold in Nepal?                          │   │
│  │                                                                         │   │
│  │ RESPONSE:"                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Context Fusion Benefits:**
- **Factual Grounding**: Responses anchored to verified Nepal legal documents
- **Business Intelligence**: Domain-specific data enhances practical advice
- **Personalization**: Conversation context enables tailored guidance
- **Source Attribution**: Clear citations for legal and regulatory claims

---

### **STEP 4: AI GENERATION**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LLaMA 3.2 AI GENERATION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: Enhanced Prompt (~1800 tokens)                                          │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         MODEL PROCESSING                                │   │
│  │                                                                         │   │
│  │  LLaMA 3.2 3B Instruct Architecture:                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Input Tokenization:                                             │    │   │
│  │  │ ├── Enhanced prompt → token IDs                                 │    │   │
│  │  │ ├── Special tokens: <|begin_of_text|>, <|start_header_id|>      │    │   │
│  │  │ └── Context window: 1800 tokens input                          │    │   │
│  │  │                                                                 │    │   │
│  │  │ Transformer Processing:                                         │    │   │
│  │  │ ├── 32 transformer layers                                       │    │   │
│  │  │ ├── Multi-head attention (8 heads)                             │    │   │
│  │  │ ├── 3072 hidden dimensions                                      │    │   │
│  │  │ ├── Feed-forward networks                                       │    │   │
│  │  │ └── Layer normalization                                         │    │   │
│  │  │                                                                 │    │   │
│  │  │ 4-bit Quantization:                                             │    │   │
│  │  │ ├── NF4 precision reduces memory by 75%                        │    │   │
│  │  │ ├── Dynamic range optimization                                  │    │   │
│  │  │ ├── Maintains 95% of original accuracy                         │    │   │
│  │  │ └── GPU memory: 1.6GB vs 6.4GB original                       │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      GENERATION PARAMETERS                              │   │
│  │                                                                         │   │
│  │  Optimized Settings for Financial Advisory:                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ • max_new_tokens: 1024 (comprehensive responses)               │    │   │
│  │  │ • temperature: 0.2 (focused, factual output)                   │    │   │
│  │  │ • top_p: 0.85 (balanced creativity vs accuracy)                │    │   │
│  │  │ • repetition_penalty: 1.05 (avoid redundant text)              │    │   │
│  │  │ • do_sample: True (controlled randomness)                      │    │   │
│  │  │ • pad_token_id: eos_token_id (proper sequence ending)          │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  Generation Process:                                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. Autoregressive Decoding:                                     │    │   │
│  │  │    ├── Generate one token at a time                             │    │   │
│  │  │    ├── Use previous tokens as context                           │    │   │
│  │  │    └── Apply sampling strategy (top-p + temperature)            │    │   │
│  │  │                                                                 │    │   │
│  │  │ 2. Factual Conditioning:                                        │    │   │
│  │  │    ├── Prioritize information from legal documents             │    │   │
│  │  │    ├── Maintain consistency with domain data                   │    │   │
│  │  │    └── Generate appropriate citations                          │    │   │
│  │  │                                                                 │    │   │
│  │  │ 3. Response Structure:                                          │    │   │
│  │  │    ├── Clear answer to user's question                         │    │   │
│  │  │    ├── Supporting legal context                                │    │   │
│  │  │    ├── Practical business implications                         │    │   │
│  │  │    └── Source citations and next steps                         │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  OUTPUT: Structured Business Response (1024 tokens)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ **VAT Registration Threshold in Nepal**                                 │   │
│  │                                                                         │   │
│  │ In Nepal, the VAT registration threshold is **Rs. 50 lakh annually**.  │   │
│  │ According to Article 15 of the Nepal Income Tax Rules 2059, every      │   │
│  │ person whose annual turnover exceeds fifty lakh rupees must register   │   │
│  │ for VAT within thirty days of exceeding this threshold.                │   │
│  │                                                                         │   │
│  │ **Key Details:**                                                        │   │
│  │ • Registration fee: Rs. 500                                            │   │
│  │ • VAT rate: 13% on taxable supplies                                    │   │
│  │ • Manufacturing industries may have preferential rates                 │   │
│  │ • Registration must be completed within 30 days                        │   │
│  │                                                                         │   │
│  │ **Business Implications:**                                              │   │
│  │ If your business turnover approaches Rs. 50 lakh, plan for VAT         │   │
│  │ compliance including record-keeping, invoicing systems, and quarterly   │   │
│  │ returns filing.                                                         │   │
│  │                                                                         │   │
│  │ **Sources:** Nepal Income Tax Rules 2059, Article 15; Current VAT      │   │
│  │ regulations 2025                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**AI Generation Advantages:**
- **Factual Accuracy**: Grounded in legal documents, not hallucinated content
- **Business Context**: Practical implications beyond just legal facts
- **Professional Format**: Structured, clear, actionable advice
- **Source Attribution**: Transparent citations for verification

---

## 🎯 **RAG PIPELINE BENEFITS SUMMARY**

| Stage | Without RAG | With RAG Pipeline |
|-------|-------------|-------------------|
| **Knowledge** | Limited to training data | Current Nepal legal documents |
| **Accuracy** | Generic responses | 60% accuracy with 100% in core areas |
| **Citations** | No source attribution | Specific legal document references |
| **Relevance** | Broad, unfocused | Targeted Nepal business context |
| **Updates** | Static knowledge | Real-time legal and rate information |
| **Trust** | Uncertain factual basis | Verified legal document grounding |

## 🚀 **Technical Performance**

- **Query Embedding**: ~50ms (sentence-BERT processing)
- **FAISS Search**: <1ms (1,402 document similarity search)
- **Context Fusion**: ~100ms (intelligent merging and token management)
- **AI Generation**: 30-60 seconds (1024 token LLaMA inference)
- **Total Pipeline**: ~60 seconds for comprehensive Nepal business advice

This RAG pipeline transforms your ChitraGupta system from a generic chatbot into a specialized Nepal financial advisor with legal authority and business intelligence! 🇳🇵
