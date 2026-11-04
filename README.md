# 🎓 CampusPrep AI — Company Placement Intelligence

Select a company → see what its historical interview experiences say (rounds, topics, questions, resources) →
ask follow-up questions in a company-scoped RAG chat. **Every statement is traceable to a PDF filename and page.**

## Architecture

```
Interview_Experiences/<year>/<company>/*.pdf
        │  scripts/ingest.py
        ▼
 PyMuPDF (page-by-page, scanned-page detection, optional Tesseract OCR) ── PII redaction (emails/phones)
        ├─► round-aware chunks (RecursiveCharacterTextSplitter) ─► local CPU embeddings (all-MiniLM-L6-v2) ─► ChromaDB
        └─► Gemini/Groq structured extraction (round-robin; rate-limit fallback) ─► grounding filter ─► data/experiences/<company>_<year>.json
                                                                           data/companies.json, ingestion_manifest.json
 app.py (Streamlit)
   ├─ dashboard  ◄── JSON only (no LLM, no vector search): overview, rounds, topics, resources
   └─ chat       ◄── LangGraph: classify_query → retrieve_documents → generate_answer → verify_citations
                       (retrieval filtered *inside* the Chroma query by company_id)
```

**How "no fabrication" is enforced (three layers, not just a prompt):**

1. **Extraction grounding** (`ingestion/insight_extractor.py`): company/year come from folders, never the LLM. Each question,
   tip and description must be lexically supported by the PDF page it cites; topics/role/difficulty must literally occur in
   the document; URLs must appear verbatim in the text. Unsupported items are dropped (tests cover this with invented items).
2. **Company-scoped retrieval** (`backend/retriever.py`): `company_id` is a Chroma metadata filter plus a post-check.
   The retriever refuses to run without a company.
3. **Citation verification** (`backend/citations.py`): the LLM tags claims `[S#]`; every claim sentence must carry valid tags
   and be lexically supported by the cited passages. Unsupported sentences trigger a bounded retry (`MAX_ANSWER_RETRIES`),
   then are removed; if nothing survives the app answers *"I couldn't find sufficient information…"*. The **Sources** list is
   built by the application from passages that were actually cited — the LLM never writes filenames or pages.

Dashboard topic frequency = number of **distinct experiences** mentioning a canonical topic (not vector hit counts) and is
labelled as historical, not a probability.

## Project structure

```
CampusPrep-AI/
├── app.py                      # Streamlit UI only
├── requirements.txt  .env.example  .gitignore  README.md
├── .streamlit/config.toml      # light theme (+ secrets.toml.example)
├── ingestion/  pdf_loader · metadata_extractor · text_splitter · insight_extractor · vector_store · pipeline
├── backend/    config · services · retriever · rag_chain · graph · citations · prompts · dashboard
├── schemas/    experience (Pydantic) · taxonomy (topic normalisation) · textmatch (grounding heuristic)
├── scripts/ingest.py           # CLI
├── data/                       # generated JSON (see data/README.md)
└── tests/                      # 40 tests
```
Additions to the requested layout: `ingestion/pipeline.py` (orchestration used by the CLI and tests), `backend/config.py`,
`backend/services.py` (the single init function), `backend/citations.py`, `schemas/taxonomy.py`, `schemas/textmatch.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # CPU-only torch via the extra index in the file
cp .env.example .env                                      # then set GEMINI_API_KEY (and optionally GEMINI_MODEL)
# Set GROQ_API_KEY too to alternate providers and fall back automatically on rate limits during ingestion.
```

## Ingest the PDFs

```bash
python scripts/ingest.py                    # PDF_INPUT_DIR from .env (default: Interview_Experiences)
python scripts/ingest.py --input-dir /path/to/Interview_Experiences
python scripts/ingest.py --skip-insights    # index only (no Gemini calls)
python scripts/ingest.py --ocr              # OCR image-only pages (needs Tesseract + pytesseract + pillow)
python scripts/ingest.py --force            # re-process unchanged files
python scripts/ingest.py --reset            # wipe index/JSON/manifest (required if you change EMBEDDING_MODEL)
```
* Re-running is incremental: unchanged files are skipped; byte-identical **or text-identical** copies are marked `duplicate`;
  a changed file replaces its old chunks. Chunk ids are deterministic.
* Failed documents (unreadable, password-protected, scanned without OCR, bad folder layout) are listed at the end, stored in
  the manifest, and shown in the app sidebar.
* The embedding model downloads once (~90 MB) during ingestion/first chat use, runs on CPU, and is cached for the process.

## Run locally

```bash
streamlit run app.py
```
The dashboard needs only the JSON in `data/`. The chat additionally loads the embedding model, ChromaDB and Gemini on first use.

## GitHub

```bash
git init && git add . && git commit -m "CampusPrep AI MVP"
git branch -M main && git remote add origin https://github.com/<you>/CampusPrep-AI.git && git push -u origin main
```
`.gitignore` excludes `.env`, `.streamlit/secrets.toml`, `Interview_Experiences/` and every `*.pdf`. **Check `git status` before pushing.**

## Storage strategy & Streamlit Community Cloud

Streamlit Community Cloud has an ephemeral filesystem, so the app never builds an index at startup.

| Option | When | How |
|---|---|---|
| **Bundled prebuilt index** (default) | Small collection, material may live in your repo | Run ingestion locally, commit `chroma_db/` and `data/`. The app only *reads* them. |
| **Chroma Cloud** | Larger or private material | Set `CHROMA_CLOUD_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` as secrets; ingest locally with the same variables. *(Wired through `chromadb.CloudClient`; not tested here.)* |

Original PDFs stay outside GitHub (private drive/bucket). `chroma_db/` and `data/experiences/*.json` contain passages copied
from the PDFs — if the PDFs are private, don't commit them (uncomment the lines in `.gitignore`) and use Chroma Cloud + a private
place for the JSON.

Deploy: push to GitHub → share.streamlit.io → *New app* → repo / branch / `app.py` → *Advanced settings → Secrets*:
paste `.streamlit/secrets.toml.example` with your values. The embedding model (~90 MB) is downloaded on the first chat
question after each cold start and cached per process; CPU torch keeps the install small (if the build exceeds resource
limits, pin a CPU torch wheel or replace the embedder with a lighter ONNX one via `EMBEDDING_MODEL`/`get_embeddings`).
The dashboard loads instantly because it does not touch the model.

## Tests

```bash
pip install pytest && python -m pytest -q
```
Tests build real PDFs with PyMuPDF in 3 companies / 2 years and run the real splitter, ChromaDB, LangGraph and citation code.
They substitute a hashing embedder and scripted LLMs because the HuggingFace hub and Gemini are not reachable in CI.
They cover: extraction, folder metadata, page numbers, chunk metadata, scanned PDFs, duplicates, grounding of extracted data,
company-isolated retrieval, empty results, citation verification, retry/withholding of hallucinated answers, unsupported
questions, company switching, LLM failure, and a Streamlit `AppTest` of the UI.

## Known limitations

* **Not run end-to-end here:** real `all-MiniLM-L6-v2` embeddings, real Gemini calls and Streamlit Cloud deployment were not
  exercised (no network access to those services in the build sandbox). Expect to tune `RETRIEVAL_MAX_DISTANCE` (default 0.85,
  an untuned guess for MiniLM cosine distance) and check that `GEMINI_MODEL` names a model your key can use.
* Citation/grounding checks are **lexical** (content-word overlap), not logical entailment. They reliably catch invented content
  but can pass a sentence that misreads a passage, and can reject heavy paraphrase (lower `CITATION_MIN_SUPPORT` to relax).
* Structured extraction quality depends on Gemini; items it gets right but words very differently may be dropped. Recall is
  traded for precision on purpose.
* Round categories/topic normalisation use a small built-in alias table (`schemas/taxonomy.py`); unknown topics appear under "Other".
* PII redaction is regex-based (emails, phone numbers); names in free text are not removed from indexed chunks.
* Chat history is kept per browser session, not persisted. Only the Gemini provider is implemented (`backend/services.py:get_llm`).
* Chroma Cloud support and OCR are implemented but untested here.
