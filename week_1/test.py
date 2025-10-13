"""
Draft/experimental RAG implementation for FAQ assistant.

Note: This is a work-in-progress version. For a clean, production-ready implementation,
see faq_assistant_rag.py
"""
import numpy as np
from typing import Any, Dict, List, Optional
import requests
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

openai_client = OpenAI()


def vector_search(
    query: str,
    *,
    embeddings: np.ndarray,  # shape: (N, D)
    documents: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    k: int = 5,
    filter_course: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = embedder.encode(query)
    # cosine sim = (A·B)/(||A|| ||B||)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    sims = emb_norm @ q_norm

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


def build_prompt(question: str, *, template: str, results: List[Dict[str, Any]]) -> str:
    # Keep it simple; pretty-print instead of dumping full JSON.
    ctx = json.dumps(results, ensure_ascii=False)[:6000]  # soft cap
    return template.format(question=question, context=ctx)


def llm(
    user_prompt: str,
    *,
    client: OpenAI,
    instructions: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
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
    search_hits = vector_search(
        query,
        embeddings=embeddings,
        documents=documents,
        embedder=embedder,
        k=5,
        filter_course="data-engineering-zoomcamp",  # fixed typo
    )
    prompt = build_prompt(query, template=template, results=search_hits)
    return llm(prompt, client=client, instructions=instructions, model=model)


if __name__ == "__main__":

    instructions = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.
        """.strip()

    prompt_template = """
        <QUESTION>
        {question}
        </QUESTION>

        <CONTEXT>
        {context}
        </CONTEXT>
        """.strip()

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")

    url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
    documents_raw = requests.get(url, timeout=20).json()
    # documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course["course"]

        for doc in course["documents"]:
            doc["course"] = course_name
            documents.append(doc)

    # Build embeddings for all documents
    embeddings = []
    for doc in tqdm(documents, desc="Embedding documents"):
        text = doc["question"] + " " + doc["text"]
        v = embedding_model.encode(text)
        embeddings.append(v)

    embeddings = np.array(embeddings)

    # Test the RAG system
    answer = rag(
        "I just discovered this course, can I join now?",
        embeddings=embeddings,
        documents=documents,
        embedder=embedding_model,
        template=prompt_template,
        instructions=instructions,
        client=openai_client,
        model="gpt-4o-mini",
    )

    print(answer)
