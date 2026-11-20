"""
app.py — Gradio semantic search over ML arXiv papers.

Pipeline (step 4 of the assignment; steps 1-3 live in indexer.py):
  4. Build a search app -> embed the query, nearest-neighbour search in Chroma,
                           dedupe chunks by paper, render results as Markdown.
  5. Publish            -> this file is the entry point of the Hugging Face Space.

On first launch (or after the Space's ephemeral disk resets) the index is built
automatically, which takes a few minutes on free CPU hardware.
"""

import urllib.parse

import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer

from indexer import CHROMA_PATH, COLLECTION_NAME, DATASET_NAME, MODEL_NAME, build_index

# ---------------------------------------------------------------------------
# Startup: load the model once and make sure the index exists
# ---------------------------------------------------------------------------
print(f"[app] Loading embedding model {MODEL_NAME} ...", flush=True)
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:  # collection doesn't exist yet
    collection = None

if collection is None or collection.count() == 0:
    print("[app] No index found — building it now (first launch takes a few minutes) ...", flush=True)
    collection = build_index(client, model=model)

N_CHUNKS = collection.count()
N_PAPERS = len({m["paper_id"] for m in collection.get(include=["metadatas"])["metadatas"]})
print(f"[app] Index ready: {N_PAPERS} papers, {N_CHUNKS} chunks", flush=True)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query, k=5):
    """Return the top-k most similar papers for `query` as Markdown."""
    query = (query or "").strip()
    if not query:
        return "Type a question or topic above to search the papers. 🙂"
    k = int(k)

    # Embed the query with the same model + normalization used for the documents.
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    # Over-fetch chunks, since several chunks may come from the same paper.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k * 4, N_CHUNKS),
        include=["metadatas", "distances", "documents"],
    )

    # Results are sorted by distance, so the first chunk seen for a paper is its best match.
    hits, seen = [], set()
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if meta["paper_id"] in seen:
            continue
        seen.add(meta["paper_id"])
        hits.append((doc, meta, 1 - dist))  # cosine similarity = 1 - cosine distance
        if len(hits) == k:
            break

    if not hits:
        return "No results found."

    lines = []
    for rank, (doc, meta, score) in enumerate(hits, start=1):
        title = meta["title"]
        link = "https://arxiv.org/search/?" + urllib.parse.urlencode({"query": title, "searchtype": "title"})
        lines.append(
            f"### {rank}. {title}\n"
            f"**Similarity:** `{score:.3f}` · [Find on arXiv]({link})\n\n"
            f"> **Matched passage:** {doc}\n\n"
            f"<details><summary>Full abstract</summary>\n\n{meta['abstract']}\n\n</details>\n\n---"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    "making large language models reason step by step",
    "detecting fraud in credit card transactions",
    "graph neural networks for molecules",
    "reducing bias in machine learning models",
    "reinforcement learning for robotics",
]

HOW_IT_WORKS = f"""
### How it works
1. **Dataset** — a random {N_PAPERS:,}-paper subset of [`{DATASET_NAME}`](https://huggingface.co/datasets/{DATASET_NAME}) (titles + abstracts of ML arXiv papers).
2. **Chunk** — each abstract is split into sentence-aligned chunks of ≤120 words with a one-sentence overlap; every chunk is prefixed with its paper title.
3. **Embed** — chunks are embedded with [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME}) (384-dim, normalized).
4. **Store & search** — the {N_CHUNKS:,} vectors live in a persistent **Chroma** collection (cosine distance). Your query is embedded with the same model, the nearest chunks are retrieved, and results are deduplicated so each paper appears once.
5. **Publish** — this Gradio app is hosted on Hugging Face Spaces.
"""

with gr.Blocks(title="ArXiv ML Semantic Search") as demo:
    gr.Markdown(
        "# 🔎 ArXiv ML Semantic Search\n"
        f"Search {N_PAPERS:,} machine-learning arXiv papers ({N_CHUNKS:,} chunks) by meaning, not keywords. "
        f"Dataset: `{DATASET_NAME}` · Model: `{MODEL_NAME}` · Vector DB: Chroma."
    )
    with gr.Row():
        query_box = gr.Textbox(label="Query", placeholder="e.g. graph neural networks for molecules", scale=4)
        k_slider = gr.Slider(1, 20, value=5, step=1, label="Number of results (k)", scale=1)
    search_btn = gr.Button("Search", variant="primary")
    gr.Examples(examples=EXAMPLES, inputs=query_box)
    results_md = gr.Markdown()

    search_btn.click(search, inputs=[query_box, k_slider], outputs=results_md)
    query_box.submit(search, inputs=[query_box, k_slider], outputs=results_md)

    gr.Markdown(HOW_IT_WORKS)

if __name__ == "__main__":
    demo.launch()
