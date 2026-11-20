"""
search_engine.py — the search logic shared by both front ends
(app.py = Gradio, streamlit_app.py = Streamlit).

Step 4 of the pipeline: embed the query, nearest-neighbour search in Chroma,
dedupe chunks by paper.
"""

import urllib.parse

import chromadb
from sentence_transformers import SentenceTransformer

from indexer import CHROMA_PATH, COLLECTION_NAME, MODEL_NAME, build_index


def load_model():
    print(f"[search] Loading embedding model {MODEL_NAME} ...", flush=True)
    return SentenceTransformer(MODEL_NAME)


def get_or_build_collection(model):
    """Open the persistent Chroma collection, building the index first if it's missing or empty."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:  # collection doesn't exist yet
        collection = None

    if collection is None or collection.count() == 0:
        print("[search] No index found — building it now (first launch takes a few minutes) ...", flush=True)
        collection = build_index(client, model=model)
    return collection


def index_stats(collection):
    """Return (number of papers, number of chunks) in the collection."""
    n_chunks = collection.count()
    n_papers = len({m["paper_id"] for m in collection.get(include=["metadatas"])["metadatas"]})
    return n_papers, n_chunks


def search_papers(model, collection, query, k=5):
    """Return the top-k most similar papers for `query` as a list of dicts."""
    # Embed the query with the same model + normalization used for the documents.
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    # Over-fetch chunks, since several chunks may come from the same paper.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k * 4, collection.count()),
        include=["metadatas", "distances", "documents"],
    )

    # Results are sorted by distance, so the first chunk seen for a paper is its best match.
    hits, seen = [], set()
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if meta["paper_id"] in seen:
            continue
        seen.add(meta["paper_id"])
        hits.append({
            "title": meta["title"],
            "score": 1 - dist,  # cosine similarity = 1 - cosine distance
            "link": "https://arxiv.org/search/?" + urllib.parse.urlencode({"query": meta["title"], "searchtype": "title"}),
            "chunk": doc,
            "abstract": meta["abstract"],
        })
        if len(hits) == k:
            break
    return hits
