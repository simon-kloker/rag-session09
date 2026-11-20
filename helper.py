import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import FAISS_INDEX

def format_docs(docs: list[Document]) -> str:
    """Format a list of documents into a numbered context block."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}, Page: {page}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


def to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else str(item))
        return "\n".join(parts).strip()
    return str(content)

def get_cohere_api_key() -> str:
    """Retrieve the Cohere API key from environment variables."""
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        raise EnvironmentError(
            "COHERE_API_KEY is not set. Add it to your environment or .env file."
        )
    return cohere_api_key

def check_faiss_store_existence() -> None:
    """Check if the FAISS index file exists."""
    if not all((FAISS_INDEX / name).is_file() for name in ("index.faiss", "documents.json")):
        raise FileNotFoundError(
            f"FAISS index or documents not found at '{FAISS_INDEX}'. "
            "Run '01_embed_documents.ipynb' to build or rebuild the index."
        )


def load_faiss_index(index_path: Path) -> tuple[faiss.Index, list[Document]]:
    index = faiss.read_index(str(index_path / "index.faiss"))
    with (index_path / "documents.json").open(encoding="utf-8") as file:
        documents = [Document(**item) for item in json.load(file)]
    if index.ntotal != len(documents):
        raise ValueError("FAISS index and document count do not match. Rebuild the index.")
    return index, documents


def search_faiss_index(
    index: faiss.Index, documents: list[Document], embeddings: Embeddings, query: str, k: int
) -> list[Document]:
    if k <= 0:
        raise ValueError("k must be positive")
    vector = np.asarray([embeddings.embed_query(query)], dtype=np.float32)
    if vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match the FAISS index.")
    _, indices = index.search(vector, k)
    return [documents[int(i)] for i in indices[0] if i >= 0]