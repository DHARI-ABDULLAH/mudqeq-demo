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
      → PDF validation (untrusted input hardening)
      → pdfplumber extraction (generic, no OCR)
      → chunking (page-tagged)
      → multilingual-e5-small embeddings (on the demo server)
      → FAISS IndexFlatIP (per session)
      → Top-K retrieval
      → bounded RAG context (+ prompt-injection defense)
      → OpenAI hosted LLM (chat only)
      → answer + citations
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
| **Chat** | OpenAI API (`api.openai.com`) | Question + minimum Top-K retrieved chunks + page numbers |

The full PDF is **never** sent to any LLM. See `services/llm_service.py`.

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
| `MAX_FILE_SIZE_MB` | No | `10` | Max upload size |
| `MAX_PAGES` | No | `50` | Max pages per PDF |
| `MAX_FILES_PER_SESSION` | No | `1` | Live documents per session |
| `MAX_QUESTIONS_PER_SESSION` | No | `20` | Chat quota per session |
| `MAX_UPLOADS_PER_SESSION` | No | `5` | Upload attempts per session |
| `SESSION_TTL_MINUTES` | No | `30` | Auto-cleanup age |
| `TOP_K` | No | `4` | Retrieved chunks |
| `MAX_RAG_CONTEXT_CHARS` | No | `6000` | Context sent to LLM |
| `DEMO_STORAGE_ROOT` | No | `/tmp/mudqeq_demo` | Ephemeral storage root |

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
