"""
streamlit_app.py — Streamlit semantic search over ML arXiv papers.

Pipeline (steps 1-3 live in indexer.py, search logic in search_engine.py):
  4. Build a search app -> Streamlit UI over the shared search_papers() function.
  5. Publish            -> deployed on Streamlit Community Cloud (main file: streamlit_app.py).

On first launch (or after the host's ephemeral disk resets) the index is built
automatically, which takes a few minutes on free CPU hardware.

Run locally:
    streamlit run streamlit_app.py
"""

import streamlit as st

from indexer import DATASET_NAME, MODEL_NAME
from search_engine import get_or_build_collection, index_stats, load_model, search_papers

st.set_page_config(page_title="ArXiv ML Semantic Search", page_icon="🔎")

EXAMPLES = [
    "making large language models reason step by step",
    "detecting fraud in credit card transactions",
    "graph neural networks for molecules",
    "reducing bias in machine learning models",
    "reinforcement learning for robotics",
]


# ---------------------------------------------------------------------------
# Startup: load the model and index once per server process
# (st.cache_resource stops Streamlit from rebuilding them on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and building the search index (first launch takes a few minutes) ...")
def load_engine():
    model = load_model()
    collection = get_or_build_collection(model)
    n_papers, n_chunks = index_stats(collection)
    print(f"[streamlit] Index ready: {n_papers} papers, {n_chunks} chunks", flush=True)
    return model, collection, n_papers, n_chunks


model, collection, N_PAPERS, N_CHUNKS = load_engine()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔎 ArXiv ML Semantic Search")
st.write(
    f"Search {N_PAPERS:,} machine-learning arXiv papers ({N_CHUNKS:,} chunks) by meaning, not keywords. "
    f"Dataset: `{DATASET_NAME}` · Model: `{MODEL_NAME}` · Vector DB: Chroma."
)

# Example buttons fill the query box (it is keyed, so we write to session_state).
st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, width="stretch"):
        st.session_state["query"] = example

# A form makes Enter and the Search button both submit the query.
with st.form("search_form"):
    query = st.text_input("Query", key="query", placeholder="e.g. graph neural networks for molecules")
    k = st.slider("Number of results (k)", min_value=1, max_value=20, value=5)
    st.form_submit_button("Search", type="primary")

query = (query or "").strip()
if not query:
    st.info("Type a question or topic above to search the papers. 🙂")
else:
    hits = search_papers(model, collection, query, k)
    if not hits:
        st.warning("No results found.")
    for rank, hit in enumerate(hits, start=1):
        st.subheader(f"{rank}. {hit['title']}")
        st.markdown(f"**Similarity:** `{hit['score']:.3f}` · [Find on arXiv]({hit['link']})")
        st.markdown(f"> **Matched passage:** {hit['chunk']}")
        with st.expander("Full abstract"):
            st.write(hit["abstract"])
        st.divider()

st.markdown(f"""
### How it works
1. **Dataset** — a random {N_PAPERS:,}-paper subset of [`{DATASET_NAME}`](https://huggingface.co/datasets/{DATASET_NAME}) (titles + abstracts of ML arXiv papers).
2. **Chunk** — each abstract is split into sentence-aligned chunks of ≤120 words with a one-sentence overlap; every chunk is prefixed with its paper title.
3. **Embed** — chunks are embedded with [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME}) (384-dim, normalized).
4. **Store & search** — the {N_CHUNKS:,} vectors live in a persistent **Chroma** collection (cosine distance). Your query is embedded with the same model, the nearest chunks are retrieved, and results are deduplicated so each paper appears once.
5. **Publish** — this Streamlit app is hosted on Streamlit Community Cloud.
""")
