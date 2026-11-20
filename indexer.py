"""
indexer.py — build the vector index for the ArXiv ML semantic search app.

Pipeline (steps 1-3 of the assignment):
  1. Pick a dataset   -> CShorten/ML-ArXiv-Papers (HF Hub), 5,000-paper subset
  2. Chunk + embed    -> sentence-based chunks, all-MiniLM-L6-v2 embeddings
  3. Store in Chroma  -> persistent collection "arxiv_ml" (cosine distance)

Run standalone to (re)build the index:
    python indexer.py
"""

import re
import time

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_DOCS = 5000                      # number of papers to index (subset for free CPU)
DATASET_NAME = "CShorten/ML-ArXiv-Papers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_ml"
EMBED_BATCH_SIZE = 64              # sentences per forward pass of the model
CHROMA_BATCH_SIZE = 1000           # Chroma rejects very large add() batches


# ---------------------------------------------------------------------------
# Step 1: load the dataset
# ---------------------------------------------------------------------------
def load_papers(n=N_DOCS):
    """Load a reproducible random subset of n papers as {paper_id, title, abstract} dicts."""
    ds = load_dataset(DATASET_NAME, split="train")
    # Drop the leftover pandas index columns; we only need title + abstract.
    ds = ds.remove_columns([c for c in ds.column_names if c.startswith("Unnamed")])
    ds = ds.shuffle(seed=42).select(range(n))

    papers = []
    for i, row in enumerate(ds):
        # arXiv titles/abstracts contain hard line breaks — collapse all whitespace.
        title = " ".join((row["title"] or "").split())
        abstract = " ".join((row["abstract"] or "").split())
        if not abstract:
            continue
        # The dataset has no arXiv ID column, so use the row position in our subset.
        papers.append({"paper_id": str(i), "title": title, "abstract": abstract})
    return papers


# ---------------------------------------------------------------------------
# Step 2a: chunk the text
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, max_words=120, overlap_sentences=1):
    """Split text into sentence-aligned chunks of up to max_words words.

    The last `overlap_sentences` sentences of each chunk are repeated at the start
    of the next one, so ideas that span a chunk boundary aren't lost.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    chunks, current = [], []

    for sentence in sentences:
        current_words = sum(len(s.split()) for s in current)
        # Close the current chunk if adding this sentence would exceed the limit
        # (unless the chunk only holds the overlap carried from the previous one).
        if current and current_words + len(sentence.split()) > max_words and len(current) > overlap_sentences:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Steps 2b + 3: embed the chunks and store them in Chroma
# ---------------------------------------------------------------------------
def build_index(client, model=None, n=N_DOCS):
    """Load, chunk, embed and store the papers in a fresh Chroma collection."""
    start = time.time()
    model = model or SentenceTransformer(MODEL_NAME)

    # Recreate the collection so every build starts clean.
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print(f"[indexer] Loading {n} papers from {DATASET_NAME} ...", flush=True)
    papers = load_papers(n)

    # Chunk every abstract; prefix each chunk with the title so it keeps context.
    ids, documents, metadatas = [], [], []
    for paper in papers:
        for chunk_idx, chunk in enumerate(chunk_text(paper["abstract"])):
            ids.append(f"{paper['paper_id']}-{chunk_idx}")
            documents.append(f"{paper['title']}. {chunk}")
            metadatas.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "chunk_idx": chunk_idx,
                "abstract": paper["abstract"],
            })
    print(f"[indexer] {len(papers)} papers -> {len(documents)} chunks. Embedding with {MODEL_NAME} ...", flush=True)

    # Embed ourselves (instead of Chroma's default embedder) so the exact same
    # model + normalization is used for documents and queries.
    embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Store in Chroma in batches.
    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        j = i + CHROMA_BATCH_SIZE
        collection.add(ids=ids[i:j], documents=documents[i:j], metadatas=metadatas[i:j], embeddings=embeddings[i:j])
        print(f"[indexer] Stored {min(j, len(ids))}/{len(ids)} chunks", flush=True)

    elapsed = time.time() - start
    print(f"[indexer] Done: {len(papers)} papers, {len(ids)} chunks, "
          f"{len(ids) / max(len(papers), 1):.2f} avg chunks/paper, built in {elapsed:.1f}s", flush=True)
    print("[indexer] Sample titles (handy for an exact-title sanity check):", flush=True)
    for paper in papers[:5]:
        print(f"  - {paper['title']}", flush=True)
    return collection


if __name__ == "__main__":
    build_index(chromadb.PersistentClient(path=CHROMA_PATH))
