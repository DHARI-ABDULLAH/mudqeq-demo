# المدقق الشامل — Mudqeq AI (Public Web Demo)

A **public, hosted demo** of Mudqeq AI: upload a PDF, then search it locally on
the demo server or ask questions answered by a hosted LLM with page citations.

> This demo is **separate** from the private macOS desktop application. The
> desktop app runs fully locally (Ollama + local FAISS) and never uploads your
> documents. This web demo is for showcasing only — **do not upload sensitive
> or confidential documents.**

---

## Architecture

```
Browser
  → Streamlit (this app, bound 0.0.0.0:7860)
      → per-session temporary storage: /tmp/mudqeq_demo/<session_id>/
      → a source is either an uploaded PDF or an added link:
          PDF: validation → pdfplumber extraction (generic, no OCR)
                          → chunking (page-tagged)
          URL: SSRF-checked fetch → readable-content extraction
                          → chunking (section-tagged)
      → multilingual-e5-small embeddings (on the demo server)
      → FAISS IndexFlatIP (one index per source, per session)
      → Top-K retrieval across the selected sources
      → bounded RAG context (+ prompt-injection defense)
      → OpenAI hosted LLM (chat only)
      → answer + citations (page number for files, page title + link for URLs)
```

The embedding model runs **on the demo server**. Only the **question + a
bounded set of retrieved chunks** are sent to OpenAI (chat only). The **Search**
page works entirely on the server with **no external LLM call**.

---

## Privacy / data flow (what leaves the server)

| Action | External network destination | Data sent |
|--------|------------------------------|-----------|
| App start | Hugging Face (only if model NOT baked in image) | Model files (no user data) |
| Upload / Extract / Index / **Search** | none | none leaves the server |
| **Add link** | the page's own host, once | An HTTP GET issued by the server; no user data beyond the address the user typed |
| **Chat** | OpenAI API (`api.openai.com`) | Question + minimum Top-K retrieved chunks + page numbers |
| **Case analysis** | OpenAI API (`api.openai.com`) | Case description + structured case + bounded retrieved evidence + page numbers |

The full PDF, and the full fetched web page, are **never** sent to any LLM. See
`services/llm_service.py`.

---

## Sources: files and links

A "source" is either an uploaded PDF or a fetched web page. Both live in one
list, share one id space, and are selected together, so any combination — one
file, several files, several links, or a mix — is just a selection.

```
link
  → validate + SSRF check (scheme, port, host, resolved IPs)
  → fetch (streamed, size-capped, redirect chain re-validated hop by hop)
  → content-type check (text/html or text/plain)
  → readable-content extraction (boilerplate, scripts, nav, hidden nodes dropped)
  → chunking → embeddings → FAISS   ← the same pipeline PDFs already use
```

| Concern | How it is handled |
|---------|-------------------|
| SSRF | Only public http(s) on ports 80/443. Loopback, private, link-local, and cloud-metadata addresses are refused before connecting **and** on every redirect |
| Memory | Response bytes capped while streaming (`MAX_URL_RESPONSE_BYTES`), extracted text capped (`MAX_URL_EXTRACTED_CHARS`), chunks capped (`MAX_URL_CHUNKS`) |
| Hangs | Separate connect/read timeouts, bounded redirect count |
| Citations | The link shown is the address the server actually fetched, read from stored metadata — the model is never given a URL it could paraphrase |
| Injection | Page text is fenced as untrusted data exactly like PDF text; instructions inside a page are quoted, never executed |
| Staleness | `retrieved_at` is stored and displayed; **تحديث المحتوى** re-fetches and rebuilds that one source's index |
| Duplicates | Canonicalised URL hash per session; re-adding the same page is refused |
| Deletion | Removing a link deletes its record, chunks, and FAISS index and nothing else |

Implementation: `services/url_security_service.py` (validation/SSRF),
`services/url_fetch_service.py` (bounded fetch), `core/html_extract.py`
(readable content), `services/url_source_service.py` (ingest orchestration),
`core/source_models.py` (shared source vocabulary).

Not supported in this version: JavaScript-rendered pages (no browser engine is
used — the user gets a clear "no readable content" message), PDFs behind a link
(the user is asked to upload the file instead), and any non-text content type.

---

## Case analysis — "تحليل حالة"

A second, explicitly-chosen mode for a full real-world problem rather than a
single lookup. It layers on top of the existing RAG stack; chat, overview, and
search keep their own code paths unchanged.

```
case text
  → understand        (structured case: parties, facts, issues, gaps)
  → missing-info gate (stops and asks when something critical is absent)
  → plan research     (3–6 focused queries: rule / conditions / exceptions /
                       procedure / consequences / limits)
  → multi-step FAISS  (each query searched independently, selected docs only)
  → evidence curation (de-duplicate, merge queries, rank, label strength)
  → candidate solutions (each tied to evidence refs)
  → grounded Arabic report + citations resolved back to real chunks
```

| Concern | How it is handled |
|---------|-------------------|
| Cost | 4 provider calls max per analysis, 1 per follow-up; every stage bounded by `MAX_CASE_*` |
| Grounding | Conclusions must cite `E#` refs; unresolvable refs are dropped, not rendered |
| Conflicts | Restricting/contradicting texts are carried into the report, never silently dropped |
| Confidence | Qualitative (`قوية` / `متوسطة` / `محدودة`) computed from evidence counters — never a fabricated percentage |
| Quota | Separate `MAX_CASES_PER_SESSION` counter, charged only after a complete report |
| Injection | Document text *and* the user's case are fenced as untrusted data in every prompt |

Implementation: `core/case_models.py`, `services/case_analysis_service.py`,
`services/query_planner_service.py`, `services/evidence_service.py`.

---

## Configuration (environment variables)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | **Yes (chat)** | — | OpenAI API key (Streamlit **Secrets** or `.env` local) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Hosted model id |
| `OPENAI_MAX_OUTPUT_TOKENS` | No | `1024` | Answer length cap |
| `OPENAI_TEMPERATURE` | No | `0.2` | Empty value omits the parameter (reasoning models) |
| `OPENAI_TIMEOUT_SECONDS` | No | `60` | Per-request timeout |
| `OPENAI_MAX_RETRIES` | No | `1` | Transient 5xx/network only — **never** rate limits |
| `OPENAI_BASE_URL` | No | — | Override endpoint (Azure/proxy) |
| `MAX_FILE_SIZE_MB` | No | `50` | Max upload size |
| `MAX_PAGES` | No | `200` | Max pages per PDF |
| `MAX_FILES_PER_SESSION` | No | `1` | Live documents per session |
| `MAX_QUESTIONS_PER_SESSION` | No | `20` | Chat quota per session |
| `MAX_UPLOADS_PER_SESSION` | No | `5` | Upload attempts per session |
| `SESSION_TTL_MINUTES` | No | `30` | Auto-cleanup age |
| `TOP_K` | No | `4` | Retrieved chunks |
| `MAX_RAG_CONTEXT_CHARS` | No | `6000` | Context sent to LLM |
| `DEMO_STORAGE_ROOT` | No | `/tmp/mudqeq_demo` | Ephemeral storage root |
| `MAX_CASE_CHARS` | No | `6000` | Max case description length |
| `MAX_CASE_RESEARCH_QUERIES` | No | `6` | Research queries per case |
| `MAX_RESULTS_PER_QUERY` | No | `5` | Chunks retrieved per research query |
| `MAX_TOTAL_EVIDENCE_CHUNKS` | No | `18` | Evidence kept after de-duplication |
| `MAX_CASE_CONTEXT_CHARS` | No | `14000` | Evidence context sent per case call |
| `MAX_CASE_LLM_CALLS` | No | `5` | Provider calls per successful analysis (incl. verify) |
| `MAX_CASES_PER_SESSION` | No | `3` | Case-analysis quota per session |
| `MAX_CASE_FOLLOWUPS_PER_CASE` | No | `5` | Follow-up questions per case |
| `MAX_URL_SOURCES_PER_SESSION` | No | `5` | Live link sources per session |
| `MAX_URLS_PER_SESSION` | No | `15` | Outbound page fetches per session (adds + refreshes) |
| `MAX_URL_RESPONSE_BYTES` | No | `5000000` | Hard cap on bytes read from a page |
| `MAX_URL_EXTRACTED_CHARS` | No | `400000` | Cap on readable text kept after extraction |
| `MIN_URL_EXTRACTED_CHARS` | No | `200` | Below this a page counts as unreadable |
| `URL_CONNECT_TIMEOUT` | No | `5` | Connect timeout (seconds) |
| `URL_READ_TIMEOUT` | No | `15` | Read timeout (seconds) |
| `MAX_URL_REDIRECTS` | No | `3` | Redirect hops, each re-validated |
| `MAX_URL_CHUNKS` | No | `600` | Chunks indexed per page |

**Never** put `OPENAI_API_KEY` in source, git, or README — only:
- **Local:** `web_demo/.env` (gitignored)
- **Streamlit Cloud:** App → Settings → Secrets

---

## Deploy on Streamlit Community Cloud (recommended — free, no Docker)

### Why this platform?

| Platform | Free? | RAM | Fits this app? |
|----------|-------|-----|----------------|
| **Streamlit Community Cloud** | ✅ $0 | up to ~2.7 GB | ⚠️ tight but feasible |
| Render Free | ✅ $0 | 512 MB | ❌ too small for PyTorch |
| Railway | ❌ | — | no real free tier |
| HF Docker Space | ❌ PRO | 16 GB | requires paid plan |

**Recommended:** [Streamlit Community Cloud](https://share.streamlit.io/) — native Streamlit, HTTPS public URL, Secrets for `OPENAI_API_KEY`, **no Docker**, **no credit card**.

> **RAM note:** PyTorch (CPU) + `multilingual-e5-small` + FAISS uses ~**2–2.5 GB** at peak. Streamlit Cloud allows up to **~2.7 GB**. The demo may work for light usage; heavy concurrent traffic could hit limits. Monitor logs after deploy.

### Prerequisites

- GitHub account (public repo required on free tier)
- OpenAI API key
- Repository containing **only** `web_demo/` files (root = app files)

### Step-by-step (you deploy — we do not push)

1. **Create a new GitHub repository** (public), e.g. `mudqeq-demo`.

2. **Copy `web_demo/` contents** into the repo root (not the whole monorepo):
   ```bash
   cd /Users/daryalshmry/Desktop/shariah_advisor_offline/web_demo
   git init
   git add .
   git status          # MUST NOT list .env
   git commit -m "Mudqeq AI public demo"
   git remote add origin https://github.com/YOUR_USERNAME/mudqeq-demo.git
   git push -u origin main
   ```

3. Go to **https://share.streamlit.io/** → **Create app**.

4. Connect your GitHub repo.

5. **Main file path:** `app.py`

6. **Python version:** 3.11 (via `.python-version` in repo).

7. **Advanced settings → Secrets** — paste:
   ```toml
   OPENAI_API_KEY = "your-openai-key-here"
   OPENAI_MODEL = "gpt-4o-mini"
   ```
   (Use your real key; never commit this to git.)

8. Click **Deploy**. First build installs PyTorch + downloads embedding model (~10–20 min).

9. **Public URL:**
   ```
   https://YOUR_APP_NAME.streamlit.app
   ```
   Share this link with anyone.

10. **Verify:** consent → upload small PDF → search → chat → delete.

### Files required in the GitHub repo

```
app.py
config.py
requirements.txt
packages.txt          # libgomp1 for FAISS
.python-version       # 3.11
.streamlit/config.toml
.streamlit/secrets.toml.example
.env.example          # empty key — template only
core/  services/  ui/  tests/
```

**Never commit:** `.env`, `*.pdf`, `storage/`, `*.faiss`, user data.

### Optional: Docker / Hugging Face (legacy)

`Dockerfile` is kept for HF Docker Spaces (requires **PRO**). Not needed for Streamlit Cloud.

---

## Run locally

```bash
cd web_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A — .env file (recommended for local dev):
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your_key_here  (never commit .env)

# Option B — shell export (no .env file):
# export OPENAI_API_KEY=your_key_here

streamlit run app.py
```

**Chat** requires `OPENAI_API_KEY`. **Search** works without it. The key is read
only by Python on the server — never sent to the browser.

Open http://localhost:8501 (Streamlit default port).

> Streamlit 1.6x uses Uvicorn internally — seeing `Uvicorn server started` in
> logs is **normal** and does **not** mean a separate FastAPI backend is running.
> Do **not** set `server.port = 7860` in `.streamlit/config.toml`; that breaks
> Streamlit Community Cloud health checks on port 8501.

### Run tests

```bash
cd web_demo
pip install pytest
PYTHONPATH=. pytest -q
```

---

## Deploy to Hugging Face Spaces (optional — requires PRO)

See `Dockerfile` and build with Docker only if you have HF PRO. Not recommended for free hosting.

```bash
cd web_demo
docker build -t mudqeq-demo .
docker run --rm -p 7860:7860 -e OPENAI_API_KEY=... mudqeq-demo
```

---

## Legacy HF deploy notes (archived)

Build the image using **`web_demo/` as the context** (do NOT use the repo root):

```bash
cd web_demo
docker build -t mudqeq-demo .
docker run --rm -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-4o-mini \
  mudqeq-demo
```

Open http://localhost:7860

---

## Deploy to Hugging Face Spaces (Docker SDK)

### Streamlit Space أم Docker Space؟

**استخدم Docker Space** (مُعدّ مسبقاً في هذا المجلد):

| | Streamlit SDK | **Docker SDK (موصى به)** |
|---|---------------|---------------------------|
| torch + FAISS + embeddings | بطيء/هشّ عند كل build | ✅ model مُدمج في الصورة |
| حجم build | غير متوقع | ✅ Dockerfile ثابت |
| non-root + healthcheck | محدود | ✅ مُفعّل |
| OpenAI secrets | ✅ | ✅ |

**لا تستخدم** Streamlit SDK مباشرة لهذا المشروع — dependencies ثقيلة (PyTorch ~2 GB + embedding model ~470 MB).

---

### إعدادات إنشاء Space (Settings)

| الإعداد | القيمة |
|---------|--------|
| **SDK** | **Docker** |
| **Hardware** | **CPU basic** (16 GB RAM) — كافٍ للديمو |
| **Visibility** | Public (للحصول على Public URL) |
| **Secret** | `OPENAI_API_KEY` = مفتاح OpenAI |
| **Variable** (اختياري) | `OPENAI_MODEL` = `gpt-4o-mini` |

> قد يتطلب Hugging Face **حساب PRO** لإنشاء Docker Space (سياسة HF 2025+). Static Spaces مجانية؛ Docker/Gradio compute قد تحتاج PRO.

---

### الملفات التي ترفعها إلى Space (محتويات `web_demo/` فقط)

```
app.py
config.py
requirements.txt
Dockerfile
README.md          ← يحتوي front-matter لـ HF (sdk: docker)
.dockerignore
.gitignore
.env.example       ← بدون مفتاح حقيقي
.streamlit/config.toml
core/
services/
ui/
tests/             ← اختياري (لا تُنسخ داخل Docker image)
```

**ممنوع رفعها:**

```
.env               ← فيه OPENAI_API_KEY — NEVER
*.pdf / *.faiss / storage/ / uploads/ / temp/
__pycache__/ / .pytest_cache/ / .venv/
desktop/ / data/ / packaging/ (خارج web_demo أصلاً)
```

---

### خطوات النشر (أنت تنفّذها — لا push تلقائي)

1. **أنشئ Space** على https://huggingface.co/new-space  
   - Name: مثلاً `mudqeq-demo`  
   - SDK: **Docker**  
   - Hardware: **CPU basic**

2. **ارفع ملفات `web_demo/` فقط** (Git أو واجهة HF):
   ```bash
   cd /Users/daryalshmry/Desktop/shariah_advisor_offline/web_demo
   git init
   git add .
   git status   # تأكد أن .env غير مُضاف
   git commit -m "Mudqeq AI public demo"
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/mudqeq-demo
   git push space main
   ```
   ⚠️ قبل `git add .` تأكد: `git check-ignore -v .env` يُظهر أن `.env` مُتجاهَل.

3. **Space → Settings → Secrets** → أضف:
   - Name: `OPENAI_API_KEY`
   - Value: مفتاح OpenAI (لا تضعه في Git)

4. **(اختياري) Settings → Variables**:
   - `OPENAI_MODEL` = `gpt-4o-mini`

5. **انتظر Build** (15–30 دقيقة أول مرة — تحميل PyTorch + baking embedding model).

6. **تحقق من Logs** — يجب أن ترى:
   - `Uvicorn server started on 0.0.0.0:7860`
   - بدون أخطاء FAISS/libgomp

7. **Public URL النهائي:**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/mudqeq-demo
   ```
   هذا الرابط يمكن إرساله لأي شخص.

8. **اختبار:** consent → upload PDF صغير → search → chat → delete.

---

### English quick reference

1. Create a new Space → **SDK: Docker** → choose CPU Basic hardware.
2. Push **only** the contents of `web_demo/` to the Space repository
   (this folder is a self-contained root; do not push the desktop repo).
   ```bash
   cd web_demo
   git init && git add . && git commit -m "Mudqeq AI demo"
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```
3. In **Space → Settings → Secrets**, add:
   - `OPENAI_API_KEY` = your OpenAI key  (**Secret**)
4. In **Space → Settings → Variables** (optional):
   - `OPENAI_MODEL` = `gpt-4o-mini` (or another current OpenAI model)
5. Let the Space **build** (first build downloads/bakes the embedding model).
6. Check **build + runtime logs** for a successful start on port 7860.
7. Open the Space URL and test: consent → upload a small PDF → search → chat →
   delete document.
8. Confirm the Space repository contains **no** private files
   (`data/`, `storage/`, `*.faiss`, `app.db`, `.env`, PDFs, DMG, `.app`).

> The build context is `web_demo/` only, so desktop/production/private files
> are never part of the image.

---

## What is NOT included (by design)

- No desktop code (`desktop/`, Tauri, FastAPI sidecar, PyInstaller).
- No Ollama. The demo uses a hosted LLM (OpenAI) instead.
- No client documents, production `storage/`, `index/`, `app.db`, or reports.
- No analytics / telemetry.
