# The Relay — Systems Benchmark & Resilient Agentic RAG Pipeline

**Vaani.dev ML Engineering Intern Technical Assessment**  
**Candidate:** Aayush Shelar ([shelaraayush535@gmail.com](mailto:shelaraayush535@gmail.com))  
**Target Provider:** Groq Free Tier (`openai/gpt-oss-20b`) & Calibrated Stand-In Model Server  

---

## 📌 Deliverables Overview

This repository contains all four required submission deliverables specified in *The Relay Assignment Brief*:

1. **`writeup.pdf`**: Strictly 2-page engineering write-up adhering to all 12 mandatory headings in exact order, including completely populated Tables 1, 2a, 2b, and 3.
2. **`results.csv`**: Ground-truth dataset containing 183 logged requests across all required scenarios and settings, strictly adhering to the 12-column schema.
3. **`assistant_chat_history.txt`**: Complete, unedited turn-by-turn prompt and interaction history with the AI coding assistant (with API credentials safely redacted).
4. **Reproducible Codebase**: Clean, modular implementation of the pipeline, stand-in server, and evaluation suite.

---

## 🏗️ Repository Architecture

- **`pipeline.py`**: The 4-step asynchronous agentic RAG pipeline:
  1. *Step 1 (Query & Paper Router)*: Deterministic paper pre-router with 45-token LLM fallback.
  2. *Step 2 (BM25 Retriever Tool)*: Successor-validated BM25 retrieval over arXiv 1706.03762v7, 2410.00037v1, and 2507.07955v2.
  3. *Step 3 (Streaming Grounded Drafter)*: Generates answers with strict `[paper_id, p.X]` citations or `NOT_IN_PAPERS` refusal.
  4. *Step 4 (Critic & Verification Loop)*: Zero-token citation gate + lightweight grounding critic with automatic 1-turn repair loop.
- **`standin_server.py`**: FastAPI ASGI stand-in server calibrated to Table 1 measurements:
  - Leaky-bucket rate limiter enforcing 8,000 TPM (133.3 tokens/s refill) and 1,000 RPD.
  - Concurrency penalty (+180ms TTFT and +4ms/token per in-flight request).
  - Authentic grounding via `data/real_response_bank.json`.
- **`retriever.py`**: Document parsing and chunking engine with stateful section-number successor validation (rejecting table-cell false positives like `54.3 MPT`) and letter-spaced figure text filtering.
- **`questions.py`**: 18 standardized evaluation questions (11 factual, 2 trap, 5 unanswerable) and deterministic citation-stripping grader.
- **`generate_compliant_pdf.py`**: ReportLab generator that computes Tables 2a, 2b, and 3 directly from `results.csv` and renders `writeup.pdf`.
- **`build_tables.py`**: Automated audit script printing recomputed tables directly from `results.csv`.

---

## 📊 Summary of Results

### Table 1: Provider Limits (Live Groq Empirical Profile)
- **Model**: `openai/gpt-oss-20b` (transitioned after `llama-3.1-8b-instant` deprecation).
- **Dual-Window Limits**: 8,000 Tokens Per Minute (TPM) + 1,000 Requests Per Day (RPD).
- **Refill Dynamics**: Continuous leaky bucket refilling at 133.3 tokens/second ($8000 \div 60\text{s}$).

### Table 3: Retrieval Quality vs. Cost (18 Questions x 3 Settings on Real Groq)
| Setting | Questions Asked | Correct & Cited % | Wrong % | Correctly Refused % | Tokens / Answer | TTFT p50 / p95 (ms) | Total Time p50 / p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`256w, top-3, no rerank`** | 18 | 44.4% (8/13 recall) | 27.8% | 27.8% (5/5 refusal) | 2,525 | 980.0 / 1,820.5 | 1,650.6 / 3,872.3 |
| **`512w, top-5, no rerank`** | 18 | **55.6% (10/13 recall)** | 16.7% | **27.8% (5/5 refusal)** | 4,990 | 1,209.8 / 10,489.8 | 2,032.6 / 12,223.2 |
| **`512w, top-5->3, with rerank`**| 18 | 44.4% (8/13 recall) | 33.3% | 22.2% (4/5 refusal) | 3,642 | 1,375.8 / 22,942.1 | 2,419.1 / 23,050.4 |

*Finding:* `512w_top5_norerank` is our primary default for maximum factual accuracy; `256w_top3_norerank` serves as dynamic fallback when approaching token limits (<3,800 remaining). Heuristic section-title reranking degraded recall by demoting hyperparameter tables.

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/Aayush-pixel29/vaani-task.git
cd vaani-task

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install httpx pymupdf rank-bm25 fastapi uvicorn pydantic python-dotenv pandas numpy reportlab

# 3. Verify results.csv and compile writeup.pdf
python generate_compliant_pdf.py
```

