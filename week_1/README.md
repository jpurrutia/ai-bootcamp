# Week 1: Introduction to LLMs and RAG Systems

This directory contains exercises and experiments from Week 1 of the AI Bootcamp, focusing on Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems.

## Learning Objectives

- Understanding LLM APIs (OpenAI Responses API and Chat Completions API)
- Building RAG (Retrieval-Augmented Generation) systems
- Working with embeddings and vector search
- Implementing structured outputs with Pydantic
- Processing and indexing various data sources (FAQs, YouTube transcripts, GitHub repositories)
- Document chunking strategies for better retrieval

## Files Overview

### Python Scripts

#### `faq_assistant_rag.py` (Production-Ready)
A clean, well-documented RAG implementation for a course FAQ assistant.

**Key features:**
- Vector search using cosine similarity with sentence-transformers
- OpenAI Responses API integration
- Comprehensive docstrings and type hints
- Filters results by course (data-engineering-zoomcamp)

**Technologies:** sentence-transformers, OpenAI, numpy, tqdm

**Usage:**
```bash
python faq_assistant_rag.py
```

#### `test.py` (Experimental/Draft)
Work-in-progress version of the RAG implementation. Contains experimental code and drafts. For production use, refer to `faq_assistant_rag.py`.

### Jupyter Notebooks

#### `env-test.ipynb`
Basic environment setup and OpenAI API testing.

**Topics covered:**
- OpenAI Responses API basics
- Creating bedtime story assistants with system prompts
- Interactive chat with conversation history
- Streaming responses
- Using toyaikit for chat interfaces
- Comparing Responses API with Chat Completions API

#### `llm-test.ipynb`
Exploring different ways to interact with OpenAI models.

**Topics covered:**
- Basic text generation with Responses API
- Working with messages and conversation history
- Response streaming
- Using system prompts for persona definition
- Multi-turn conversations
- Integration with toyaikit runners

#### `rag-test.ipynb`
Core RAG implementation using FAQ data from a course.

**Topics covered:**
- Loading and preprocessing FAQ documents
- Text search with minsearch (keyword-based)
- Vector embeddings with sentence-transformers
- Vector search implementation
- Combining retrieval with LLM generation
- Cosine similarity for document ranking

**Dataset:** Course FAQ data from alexeygrigorev/llm-rag-workshop

#### `rag-test-yt.ipynb`
Processing YouTube transcripts with LLMs to generate summaries and chapters.

**Topics covered:**
- Fetching YouTube transcripts with youtube-transcript-api
- Formatting timestamps (H:MM:SS format)
- Creating video summaries with LLMs
- Structured outputs with Pydantic models
- Extracting chapters with timestamps
- Using `responses.parse()` for typed responses

**Key classes:**
- `Chapter`: Represents a video chapter with timestamp and title
- `YTSummaryResponse`: Contains summary and list of chapters

#### `documentation_assistant.ipynb`
Advanced RAG system for GitHub documentation.

**Topics covered:**
- Downloading and parsing GitHub repositories
- Extracting markdown files from zip archives
- Parsing frontmatter from markdown files
- Document chunking with sliding windows
- Building searchable indexes with minsearch
- Context-aware answer generation with source citations
- Handling large documents (chunking strategy: size=2000, step=1000)

**Target repository:** evidentlyai/docs

**Key utilities:**
- `GithubRepositoryDataReader`: Downloads and filters GitHub repo files
- `sliding_window()`: Creates overlapping document chunks
- `chunk_documents()`: Splits large documents for better retrieval
- `index_documents()`: Creates searchable indexes with optional chunking

## Technologies Used

### Core Libraries
- **OpenAI**: LLM API access (Responses API and Chat Completions)
- **sentence-transformers**: Text embeddings for vector search (multi-qa-distilbert-cos-v1)
- **numpy**: Numerical operations and cosine similarity calculations
- **minsearch**: Lightweight search engine for keyword and vector search

### Utilities
- **requests**: HTTP client for downloading data
- **tqdm**: Progress bars for long-running operations
- **pydantic**: Data validation and structured outputs
- **frontmatter**: Parsing markdown frontmatter
- **youtube-transcript-api**: Fetching YouTube video transcripts
- **toyaikit**: Interactive chat interfaces for Jupyter notebooks

## Key Concepts

### RAG (Retrieval-Augmented Generation)
A technique that combines information retrieval with LLM generation:
1. **Retrieve**: Search for relevant documents using keywords or vectors
2. **Augment**: Build a prompt that includes the retrieved context
3. **Generate**: Ask the LLM to answer based only on the provided context

### Vector Search
Using embeddings to find semantically similar documents:
- Convert text to high-dimensional vectors (embeddings)
- Compare query vector with document vectors using cosine similarity
- Return top-k most similar documents

**Formula:** `cosine_similarity = (A·B) / (||A|| × ||B||)`

### Document Chunking
Breaking large documents into smaller, overlapping pieces:
- **Purpose**: Better retrieval granularity, token limit management
- **Method**: Sliding window with configurable size and step
- **Example**: size=2000 chars, step=1000 chars (50% overlap)

### Structured Outputs
Using Pydantic models to get typed responses from LLMs:
```python
class MyResponse(BaseModel):
    field1: str
    field2: list[Item]

response = client.responses.parse(
    model="gpt-4o-mini",
    input=messages,
    text_format=MyResponse
)
result = response.output[0].content[0].parsed
```

## Prerequisites

### Installation
```bash
# Install required packages
pip install openai sentence-transformers numpy tqdm
pip install minsearch frontmatter youtube-transcript-api
pip install toyaikit  # For interactive notebooks
```

### Environment Setup
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Python Version
Python 3.9+ recommended (for type hints like `list[dict]`)

## Running the Code

### Python Scripts
```bash
# Run the production RAG system
python faq_assistant_rag.py

# Run the experimental version
python test.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Then open any `.ipynb` file in the browser.

## Common Patterns

### Basic LLM Call
```python
from openai import OpenAI

client = OpenAI()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

response = client.responses.create(
    model="gpt-4o-mini",
    input=messages
)
print(response.output_text)
```

### Vector Search
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize model
model = SentenceTransformer("multi-qa-distilbert-cos-v1")

# Create embeddings
embeddings = np.array([model.encode(doc) for doc in documents])
query_vec = model.encode(query)

# Compute cosine similarity
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
q_norm = query_vec / np.linalg.norm(query_vec)
similarities = emb_norm @ q_norm

# Get top-k results
top_k_indices = np.argsort(-similarities)[:k]
```

### RAG Pipeline
```python
# 1. Retrieve relevant documents
search_results = vector_search(query, embeddings, documents, k=5)

# 2. Build prompt with context
prompt = f"""
<QUESTION>{query}</QUESTION>
<CONTEXT>{json.dumps(search_results)}</CONTEXT>
"""

# 3. Generate answer
answer = llm(prompt, instructions="Answer based only on the context.")
```

## Next Steps

Week 1 establishes the foundation for:
- More advanced RAG techniques (hybrid search, re-ranking)
- Agents and tool use
- Evaluation and monitoring
- Production deployment considerations

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [LLM RAG Workshop Repository](https://github.com/alexeygrigorev/llm-rag-workshop)

## Notes

- The `ph1PxZIkz1o.bin` file contains a cached YouTube transcript (pickle format)
- All notebooks use `gpt-4o-mini` model for cost efficiency
- Vector search uses `multi-qa-distilbert-cos-v1` for embeddings (optimized for Q&A tasks)
