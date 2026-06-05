# IPL 2026 RAG System

A Retrieval-Augmented Generation (RAG) system built to answer questions about the 2026 Indian Premier League season. The project compares a dense-only baseline retrieval pipeline against an enhanced pipeline using hybrid retrieval and cross-encoder reranking, with full quantitative evaluation.

## Pipeline Overview

**Baseline** - Dense retrieval using BGE embeddings, top-5 chunks passed to LLM.

**Enhanced** — Three-stage pipeline:
1. Hybrid retrieval fusing dense (BGE) and sparse (BM25) results using Reciprocal Rank Fusion
2. Cross-encoder reranking over the fused candidate pool
3. Top-5 reranked chunks passed to LLM

The two pipelines use the same prompt, model and temperature so any performance difference is attributable to retrieval alone.

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.3 70B via Groq API |
| Orchestration | Python, Google Colab |
| Data Sources | Wikipedia (MediaWiki API), Cricbuzz |

## Corpus

- 66 chunks from 9 Wikipedia articles (main IPL 2026 article + 8 team season pages)
- 74 chunks from Cricbuzz covering all 70 league matches and 4 playoff fixtures
- 140 chunks total, built entirely from post-cutoff sources

## Results

| Pipeline | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| Baseline | 0.45 | 0.21 | 0.43 |
| Enhanced | 0.75 | 0.56 | 0.69 |

The most significant finding was that the baseline scored 0.00 Hit@5 on simple fact questions despite these being the easiest category. All 74 match result chunks share near-identical boilerplate prose, making them semantically indistinguishable to a dense embedding model. BM25's exact lexical match on ordinal tokens like "16th match" fixes this directly — simple fact Hit@5 rose from 0.00 to 0.71 after adding the hybrid pipeline.

False-refusal rate (refusing answerable questions) also dropped from 0.55 to 0.25 as a direct consequence of better retrieval.

## Project Structure

```
ipl_rag/
├── iplrag.ipynb   # Full pipeline notebook
├── data/
│   ├── raw/
│   │   ├── wikipedia/       # Raw Wikipedia article text
│   │   └── cricbuzz/        # Raw Cricbuzz match data
│   └── chunks.json          # Final chunked corpus
├── chroma_db/               # Persisted ChromaDB vector store
└── results/
    ├── run_results.json              # Per-query answers for both pipelines
    ├── retrieval_metrics_per_query.csv
    └── generation_sample.csv        # Manual generation quality comparison
```

---

## Setup

1. Open the notebook in Google Colab
2. Mount Google Drive and set the project directory path
3. Add your Groq API key when prompted
4. Run all cells in order

The notebook includes response caching so re-runs after the first pass are near-free.

## Key Design Decisions

- **Custom BM25 tokenizer** that preserves numerics and score notation (`203/4`) since standard tokenizers strip these and they are the most discriminating tokens in match result chunks
- **RRF over score normalisation** for fusion because dense and BM25 scores are on incompatible scales
- **BGE-small over BGE-large** to stay within Colab free tier memory while the cross-encoder compensates for any ranking errors from the smaller bi-encoder
- **Two enhancements rather than three** to keep failure attribution clean between baseline and enhanced
