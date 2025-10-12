import numpy as np
from typing import Any, Dict, List, Optional
import requests, json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def vector_search(
    query: str,
    *,
    embeddings: np.ndarray,                 # shape: (N, D)
    documents: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    k: int = 5,
    filter_course: Optional[str] = None
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
    # Keep it simple; you could pretty-print instead of dumping full JSON.
    ctx = json.dumps(results, ensure_ascii=False)[:6000]  # soft cap
    return template.format(question=question, context=ctx)

def llm(
    user_prompt: str,
    *,
    client: OpenAI,
    instructions: Optional[str] = None,
    model: str = "gpt-4o-mini"
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
    model: str = "gpt-4o-mini"
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
    instructions = (
        "You're a course teaching assistant. Answer the QUESTION based only on the CONTEXT. "
        "If the answer isn't in CONTEXT, say you don't know."
    )

    prompt_template = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

    url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
    documents_raw = requests.get(url, timeout=20).json()

    documents: List[Dict[str, Any]] = []
    for course in documents_raw:
        for doc in course["documents"]:
            doc["course"] = course["course"]
            documents.append(doc)

    embedder = SentenceTransformer("multi-qa-distilbert-cos-v1")

    # Build one embedding per doc on a consistent text field
    texts = [f"{d.get('question','')} {d.get('text','')}" for d in documents]
    embeddings = np.array([embedder.encode(t) for t in tqdm(texts, desc="Embedding docs")])

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
