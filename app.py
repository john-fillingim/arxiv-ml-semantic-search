"""
app.py — Gradio semantic search over ML arXiv papers.

Pipeline (step 4 of the assignment; steps 1-3 live in indexer.py):
  4. Build a search app -> embed the query, nearest-neighbour search in Chroma,
                           dedupe chunks by paper, render results as Markdown.
  5. Publish            -> Gradio entry point (for Hugging Face Spaces). The deployed
                           version uses the Streamlit front end in streamlit_app.py,
                           which shares the same search logic (search_engine.py).

On first launch (or after the host's ephemeral disk resets) the index is built
automatically, which takes a few minutes on free CPU hardware.
"""

import gradio as gr

from indexer import DATASET_NAME, MODEL_NAME
from search_engine import get_or_build_collection, index_stats, load_model, search_papers

# ---------------------------------------------------------------------------
# Startup: load the model once and make sure the index exists
# ---------------------------------------------------------------------------
model = load_model()
collection = get_or_build_collection(model)
N_PAPERS, N_CHUNKS = index_stats(collection)
print(f"[app] Index ready: {N_PAPERS} papers, {N_CHUNKS} chunks", flush=True)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query, k=5):
    """Return the top-k most similar papers for `query` as Markdown."""
    query = (query or "").strip()
    if not query:
        return "Type a question or topic above to search the papers. 🙂"

    hits = search_papers(model, collection, query, int(k))
    if not hits:
        return "No results found."

    lines = []
    for rank, hit in enumerate(hits, start=1):
        lines.append(
            f"### {rank}. {hit['title']}\n"
            f"**Similarity:** `{hit['score']:.3f}` · [Find on arXiv]({hit['link']})\n\n"
            f"> **Matched passage:** {hit['chunk']}\n\n"
            f"<details><summary>Full abstract</summary>\n\n{hit['abstract']}\n\n</details>\n\n---"
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
