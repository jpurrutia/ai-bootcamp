# podcast_index.py
# End-to-end: download -> parse -> normalize -> chunk -> index -> search

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import io, re, zipfile, requests, yaml, frontmatter
from minsearch import Index


# Download from GitHub
@dataclass
class RawRepositoryFile:
    filename: str
    content: str


def read_github_repo(
    owner: str,
    repo: str,
    *,
    folder: str = "",
    exts: Tuple[str, ...] = ("md",),
    branch: str = "main",
) -> List[RawRepositoryFile]:
    """
    Download a GitHub repo zip and return selected files (by folder + extensions).
    """
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    out: List[RawRepositoryFile] = []

    for info in zf.infolist():
        parts = info.filename.split("/", 1)
        if len(parts) < 2:
            continue
        path = parts[1]  # strip top-level folder

        if folder and not path.startswith(folder):
            continue
        if not any(path.endswith(f".{ext}") for ext in exts):
            continue

        with zf.open(info) as f:
            content = f.read().decode("utf-8", errors="ignore").strip()
            out.append(RawRepositoryFile(path, content))
    return out


# Parse (lenient)
_FM_SPLIT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_data(files: List[RawRepositoryFile]) -> List[Dict[str, Any]]:
    """
    Extract front matter (metadata) + body content, tolerating YAML quirks.
    Always returns dicts with at least: filename, content (body string), and any FM keys.
    """
    parsed: List[Dict[str, Any]] = []
    for f in files:
        meta: Dict[str, Any] = {}
        body = ""
        try:
            post = frontmatter.loads(f.content)  # uses safe loader
            meta = dict(post.metadata or {})
            body = post.content or ""
        except Exception:
            # fallback: manual split + permissive loader (treats everything as strings)
            m = _FM_SPLIT.match(f.content)
            if m:
                try:
                    meta = yaml.load(m.group(1), Loader=yaml.BaseLoader) or {}
                except Exception:
                    meta = {}
                body = f.content[m.end() :]
            else:
                meta, body = {}, f.content

        meta["filename"] = f.filename
        meta["content"] = body
        parsed.append(meta)
    return parsed


# Normalize transcript -> text
def transcript_to_text(d: Dict[str, Any]) -> str:
    """
    Jekyll podcast pages store `transcript` in front matter as a list of items:
      - {"header": "..."} or {"line": "...", "who": "..."}
    Convert this to readable plain text. If no transcript, fall back to body content.
    """
    tr = d.get("transcript") or []
    if isinstance(tr, list) and tr:
        lines: List[str] = []
        for item in tr:
            if not isinstance(item, dict):
                continue
            if "header" in item:
                lines.append(f"# {str(item['header']).strip()}")
            elif "line" in item:
                line = str(item["line"]).strip()
                who = item.get("who")
                lines.append(f"{who}: {line}" if who else line)
        if lines:
            return "\n".join(lines)
    return str(d.get("content", ""))


# Chunking utilities
def sliding_window(seq: str, size: int, step: int) -> List[Dict[str, Any]]:
    """
    Character-based sliding window with overlap: emits dicts {start, content}.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    out: List[Dict[str, Any]] = []
    for i in range(0, n, step):
        out.append({"start": i, "content": seq[i : i + size]})
        if i + size >= n:
            break
    return out


def chunk_documents(
    docs: Iterable[Dict[str, Any]],
    *,
    size: int = 2000,
    step: int = 1000,
    content_field: str = "content",
) -> List[Dict[str, Any]]:
    """
    Split each doc into overlapping chunks; preserve original metadata per chunk.
    """
    chunks: List[Dict[str, Any]] = []
    for d in docs:
        base = dict(d)
        text = str(base.pop(content_field, ""))  # avoid KeyError; ensure string
        for cid, ch in enumerate(sliding_window(text, size=size, step=step)):
            chunks.append(
                {
                    **base,
                    content_field: ch["content"],
                    "start": ch["start"],
                    "chunk_id": cid,
                }
            )
    return chunks


# Indexing (minsearch)
def index_documents(
    docs: List[Dict[str, Any]],
    *,
    chunk: bool = False,
    chunk_size: int = 2000,
    chunk_step: int = 1000,
) -> Index:
    """
    Build a BM25 index. If chunk=True, chunk docs first.
    """
    if chunk:
        docs = chunk_documents(docs, size=chunk_size, step=chunk_step)
    idx = Index(text_fields=["content", "title", "filename"])
    idx.fit(docs)
    return idx


# Convenience: build from repo
def build_podcast_index(
    owner: str = "DataTalksClub",
    repo: str = "datatalksclub.github.io",
    *,
    folder: str = "_podcast/",
    chunk_size: int = 2000,
    chunk_step: int = 1000,
) -> Tuple[Index, List[Dict[str, Any]]]:
    """
    Download podcast markdown, parse, turn transcripts into text, and index (chunked).
    Returns (index, docs_used_for_index).
    """
    files = read_github_repo(owner, repo, folder=folder, exts=("md",))
    parsed = parse_data(files)

    docs: List[Dict[str, Any]] = []
    for d in parsed:
        docs.append(
            {
                "filename": d["filename"],
                "title": d.get("title", ""),
                "content": transcript_to_text(d),
            }
        )

    idx = index_documents(
        docs,
        chunk=True,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
    )
    return idx, docs


if __name__ == "__main__":
    index, docs = build_podcast_index(
        folder="_podcast/",
        chunk_size=2000,
        chunk_step=1000,
    )

    queries = [
        "How can I make money with AI?",
    ]
    for q in queries:
        print("\nQ:", q)
        for r in index.search(q, num_results=3):
            print("-", r["filename"], "|", r.get("chunk_id"), "@", r.get("start"))
            print(r["content"][:220].replace("\n", " "), "...\n")
