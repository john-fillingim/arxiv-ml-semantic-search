---
title: ArXiv ML Semantic Search
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🔎 ArXiv ML Semantic Search

A semantic search engine over machine-learning arXiv papers. Type a question or topic in plain
English and get the most relevant papers by *meaning*, not keyword overlap — along with the
passage that matched and the full abstract.

## The 5-step pipeline

| Step | What | Where |
|------|------|-------|
| 1. Pick a dataset | [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) — ~117k ML paper titles + abstracts. A reproducible 5,000-paper subset (`shuffle(seed=42)`) is used so the index builds in a few minutes on free CPU. | `indexer.py` → `load_papers` |
| 2. Chunk + embed | Abstracts are split into sentence-aligned chunks of ≤120 words with a 1-sentence overlap; each chunk is prefixed with the paper title. Chunks are embedded with [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim, normalized). | `indexer.py` → `chunk_text`, `build_index` |
| 3. Store in a vector DB | Embeddings, chunk text and metadata (paper id, title, chunk index, full abstract) go into a persistent **Chroma** collection `arxiv_ml` using cosine distance. | `indexer.py` → `build_index` |
| 4. Build a search app | The query is embedded with the same model, the nearest chunks are retrieved, and they're deduped so each paper appears once, with a similarity score (`1 − cosine distance`). Two front ends share this logic: **Streamlit** (deployed) and **Gradio**. | `search_engine.py` → `search_papers`; UIs in `streamlit_app.py` and `app.py` |
| 5. Publish | Hosted on **Streamlit Community Cloud** (free). The index is not committed — the app builds it on first launch. | see *Deploy* below |

### Design notes
- **Why chunk short abstracts?** Most abstracts fit in 1–2 chunks, but chunking keeps each vector focused and lets the UI show *which passage* matched.
- **Why embed outside Chroma?** Passing `embeddings=` explicitly guarantees documents and queries use the identical model and normalization.
- **Why over-fetch?** The app retrieves `4 × k` chunks, then keeps only the best chunk per paper, so you still get `k` distinct papers.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python indexer.py                # optional: builds chroma_db/ (app.py also does this if missing)
streamlit run streamlit_app.py   # Streamlit UI at http://localhost:8501
python app.py                    # or the Gradio UI at http://127.0.0.1:7860
```

To index more or fewer papers, change `N_DOCS` at the top of `indexer.py` and rerun it.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → deploy from the GitHub repo.
3. Repository `john-fillingim/arxiv-ml-semantic-search`, branch `main`, main file `streamlit_app.py`.
4. The first boot installs dependencies and builds the index (several minutes). Free apps sleep when idle and may rebuild the index on wake.

> **Why Streamlit?** Hugging Face now requires a paid PRO plan to host Gradio Spaces, so the app is deployed on Streamlit Community Cloud instead. The Gradio version (`app.py`) is kept and uses the same search code. The YAML header at the top of this README is the Hugging Face Spaces config, if it's ever deployed there.
