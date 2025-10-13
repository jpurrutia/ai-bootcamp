"""
Minimal RAG (Retrieval-Augmented Generation) demo using:
- SentenceTransformer embeddings for vector search over course FAQ documents
- A simple cosine-similarity top-k retriever
- OpenAI Responses API to generate an answer using retrieved context

Workflow:
1) Download documents JSON (course FAQs).
2) Build one embedding per document (question + text).
3) At query time, embed the query and retrieve top-k similar docs.
4) Build a prompt that includes QUESTION + CONTEXT (retrieved docs).
5) Ask the model to answer using only the provided CONTEXT.

Note: This is intentionally minimal—no persistence, truncation/formatting logic is basic.
"""

from typing import Any, Dict, List, Optional
import json
import requests
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def vector_search(
    query: str,
    *,
    embeddings: np.ndarray,
    documents: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    k: int = 5,
    filter_course: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most similar documents to `query` via cosine similarity.

    Args:
        query: Natural-language search string.
        embeddings: Precomputed document embeddings with shape (N, D), where
            N = number of documents, D = embedding dimension.
        documents: List of dicts; each dict contains at least 'question', 'text',
            and optionally a 'course' key used for filtering.
        embedder: SentenceTransformer model used to encode the query.
        k: Number of hits to return.
        filter_course: If provided, only documents whose doc['course'] equals this
            value will be considered.

    Returns:
        A list of up to `k` document dicts augmented with a "_score" float for similarity.
        The list is sorted by descending similarity.
    """
    # Embed the query
    q = embedder.encode(query)

    # Normalize embeddings and compute cosine similarity
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    sims = emb_norm @ q_norm  # shape: (N,)

    # Rank by similarity
    idxs = np.argsort(-sims)  # descending
    hits: List[Dict[str, Any]] = []
    for i in idxs:
        doc = documents[i]
        if filter_course and doc.get("course") != filter_course:
            continue
        hits.append({**doc, "_score": float(sims[i])})
        if len(hits) >= k:
            break
    return hits


def build_prompt(
    question: str,
    *,
    template: str,
    results: List[Dict[str, Any]],
) -> str:
    """
    Fill a prompt template with the end-user QUESTION and retrieved CONTEXT.

    The CONTEXT is a JSON dump of the retrieved documents (kept minimal for demo
    purposes). In production, consider rendering only selected fields (e.g., section
    + snippet) and enforcing strict token limits.

    Args:
        question: The user's question.
        template: A string template that includes `{question}` and `{context}` tokens.
        results: Retrieved documents to include as context.

    Returns:
        The formatted prompt string ready for the LLM.
    """
    ctx = json.dumps(results, ensure_ascii=False)[:6000]  # soft cap for demo
    return template.format(question=question, context=ctx)


def llm(
    user_prompt: str,
    *,
    client: OpenAI,
    instructions: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Call the OpenAI Responses API with optional system instructions and return text.

    Args:
        user_prompt: The fully-assembled prompt to send as the user message.
        client: An initialized `OpenAI()` client.
        instructions: Optional system message to steer behavior (e.g., "answer only
            from context; say 'don't know' otherwise").
        model: OpenAI model name.

    Returns:
        The response text (assistant output) as a string.
    """
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_prompt})

    resp = client.responses.create(model=model, input=messages)
    return resp.output_text


def rag(
    query: str,
    *,
    embeddings: np.ndarray,
    documents: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    template: str,
    instructions: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    End-to-end RAG pipeline: retrieve -> build prompt -> generate answer.

    Args:
        query: The user's question.
        embeddings: Precomputed document embeddings (N, D).
        documents: The original documents corresponding to `embeddings`.
        embedder: SentenceTransformer model used for encoding the query.
        template: Prompt template with `{question}` and `{context}` tokens.
        instructions: System message to constrain model behavior.
        client: OpenAI client to call the model.
        model: OpenAI model name.

    Returns:
        The model's answer as a string.
    """
    search_hits = vector_search(
        query,
        embeddings=embeddings,
        documents=documents,
        embedder=embedder,
        k=5,
        filter_course="data-engineering-zoomcamp",
    )
    prompt = build_prompt(query, template=template, results=search_hits)
    return llm(prompt, client=client, instructions=instructions, model=model)


if __name__ == "__main__":
    # System / behavior instructions for the assistant
    instructions = (
        "You're a course teaching assistant. Answer the QUESTION based only on the CONTEXT. "
        "If the answer isn't in CONTEXT, say you don't know."
    )

    # Simple prompt template with QUESTION and CONTEXT placeholders
    prompt_template = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

    # 1) Load documents
    url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
    documents_raw = requests.get(url, timeout=20).json()

    documents: List[Dict[str, Any]] = []
    for course in documents_raw:
        for doc in course["documents"]:
            doc["course"] = course["course"]
            documents.append(doc)

    # 2) Build embeddings (one per document)
    embedder = SentenceTransformer("multi-qa-distilbert-cos-v1")
    texts = [f"{d.get('question', '')} {d.get('text', '')}" for d in documents]
    embeddings = np.array([embedder.encode(t) for t in tqdm(texts, desc="Embedding docs")])

    # 3) Ask the model with retrieved context
    client = OpenAI()
    answer = rag(
        "I just discovered this course, can I join now?",
        embeddings=embeddings,
        documents=documents,
        embedder=embedder,
        template=prompt_template,
        instructions=instructions,
        client=client,
        model="gpt-4o-mini",
    )
    print(answer)
