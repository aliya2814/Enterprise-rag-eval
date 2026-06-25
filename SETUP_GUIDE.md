# Setup Guide — Project #1 (simple steps + why)

Follow this top to bottom. Do ONE phase, make sure it worked, then move on.
If something breaks, copy the red error text and send it to Claude.

Everything here is FREE. You run all of it on your own laptop.

---

## What this project actually is (plain English)
A small web app where you ask questions about real company annual reports
("10-K filings") and it answers using ONLY those documents, and shows you which
page it got the answer from. That "answer only from the documents + show proof"
idea is called **RAG**, and it is the #1 skill companies want in 2026.

---

## Phase 0 — Check Python is installed
**Why:** the whole project is written in Python, so your computer needs it.

Open Command Prompt (press Start, type `cmd`, Enter) and run:
```
py --version
```
- If you see something like `Python 3.11.x` (3.10 or higher) → good, continue.
- If it says "not recognized" → install Python from https://www.python.org/downloads/
  and TICK the box "Add Python to PATH" during install. Then reopen cmd and retry.

---

## Phase 1 — Go into the project folder
**Why:** every command must run "inside" the project so it can find the code.
```
d:
cd "D:\CLAUDE\Projects\01-enterprise-rag-eval"
```
Check you are in the right place:
```
dir
```
You should see `README.md`, `app.py`, `src`, etc.

---

## Phase 2 — Create a "virtual environment"
**Why:** this is a private box for THIS project's tools, so installing things here
never messes up the rest of your computer. Professionals always do this.
```
py -m venv .venv
.venv\Scripts\activate
```
After this your line should start with `(.venv)`. That means the box is active.
(You repeat ONLY `.venv\Scripts\activate` each new time you open cmd.)

---

## Phase 3 — Install the tools (lite version first)
**Why:** the code depends on libraries (ready-made tools). The "min" list is
smaller so your first run downloads less and breaks less.
```
pip install -r requirements-min.txt
```
This takes 5–15 minutes the first time. That is normal. Let it finish.

---

## Phase 4 — Get a free "brain" (Groq API key)
**Why:** RAG finds the right text, but an AI model writes the final answer in
plain English. Groq gives you a fast model for free.

1. Go to https://console.groq.com  → sign up (free).
2. Open "API Keys" → "Create API Key" → copy it.
3. In cmd, tell the project to use Groq and paste your key:
```
set LLM_BACKEND=groq
set GROQ_API_KEY=paste_your_key_here
```
Note: `set` only lasts for this cmd window. If you close it, run these two lines again.

---

## Phase 5 — Download 2 real company reports
**Why:** real data (not fake) is what makes a portfolio impressive. This grabs
Apple's and Microsoft's official annual reports from the free public SEC website.
```
set EDGAR_UA=Aliya Tarrannum nawaznaliya@gmail.com
py src\download_filings.py --tickers AAPL MSFT --out data\raw
```
You should see `[ok] AAPL ...` and `[ok] MSFT ...`. Check with `dir data\raw`.

---

## Phase 6 — "Ingest" the documents
**Why:** the app can't read a giant report directly. This step chops the reports
into small pieces, turns each piece into numbers the computer can search
("embeddings"), and saves them. This is the "memory" the app searches later.
```
py -m src.ingest --input data\raw --persist data\processed
```
You should see `Ingested NNNN chunks ...`.

---

## Phase 7 — Ask your first question (the payoff!)
**Why:** this proves the whole pipeline works end to end.

First turn off the optional reranker (it is not in the lite install yet):
```
set USE_RERANKER=false
```
```
py -m src.generate --q "What are Apple's main business risk factors?"
```
You should get an answer plus a list of SOURCES (which document piece it used).

---

## Phase 8 — Run the visual app
**Why:** recruiters click links, not code. This opens a real web page where you
type questions and see answers + evidence. (For the very first run we keep the
reranker OFF so there's nothing extra to install.)
```
set USE_RERANKER=false
streamlit run app.py
```
Your browser opens at http://localhost:8501. Try a few questions. Press Ctrl+C in
cmd to stop it.

---

## Phase 9 — Measure how good it is
**Why:** anyone can build a chatbot; showing you MEASURED its quality is what makes
you look like an engineer. This runs a set of test questions and scores retrieval.
```
py eval\run_eval.py
```
Write the numbers it prints into `eval\EVALUATION_REPORT.md`.

---

## Phase 10 — Put it on GitHub
**Why:** this is your public proof of work that you put on your resume.
```
git init
git add .
git commit -m "Day 1: working RAG pipeline over SEC filings"
```
Then create a new empty repo on github.com and follow its "push existing repo"
lines. (We'll do this together when you reach it.)

---

## Upgrades for later (after the basics work)
- Install the full tools: `pip install -r requirements.txt`
- Turn the reranker ON: `set USE_RERANKER=true` (improves answer quality)
- Run the deeper RAGAS evaluation: `set RUN_RAGAS=true` then `py eval\run_eval.py`
- Fill in the results tables in `eval\EVALUATION_REPORT.md`
- Deploy free to Hugging Face Spaces and add the live link to your README

Remember: log a sentence in `PROGRESS.md` each day and commit. A steady GitHub
history tells recruiters you really built this.
