# 🧠 Integrated Cybersecurity Assessment System

A system combining **RAG** (Retrieval-Augmented Generation) and **BERT-based classification**
for cybersecurity resilience assessments. It rates a cybersecurity setup as **Bad**, **Good**,
or **Excellent** and explains why.

## 📘 Overview

Three complementary approaches:

1. **RAG System** — Uses LLMs with FAISS vector retrieval to analyse a cybersecurity context
   against a database of example profiles.
2. **BERT Classification** — A fine-tuned `bert-mini` model that classifies a setup as
   Bad / Good / Excellent.
3. **Hybrid** — Combines both for a more reliable verdict plus improvement suggestions.

### Supported models

| Key | Model | Embeddings used |
|-----|-------|-----------------|
| `gpt-4`      | GPT-4    | OpenAI |
| `deepseek`   | DeepSeek | Gemini |
| `mistral`    | Mistral  | Gemini |
| `gemini-pro` | Gemini   | Gemini |
| `claude`     | Claude   | Gemini |

Each model is optional — the system initialises only those whose API key is present and
logs a warning for the rest.

## 📁 Project Structure

```
Cyber Resilience Project/
├── app.py                     # RAG system: PDF loading, embeddings, vector store, LLM classification
├── integrated_system.py       # Hybrid system (RAG + BERT) with the Gradio web UI — main entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys (copy to .env — never commit the real .env)
├── .gitignore
│
├── scripts/                   # One-off data-prep utilities
│   ├── create_json.py         #   PDF → JSON extraction
│   ├── dataset_ordering.py    #   Re-order dataset rows by label
│   └── merge_csv.py           #   Merge original + synthetic datasets
│
├── llm_finetune_project/      # BERT fine-tuning sub-project
│   ├── train_finetune.py      #   Fine-tune bert-mini (optionally with LoRA/PEFT)
│   ├── train_bert_classification.py  # Train + confusion-matrix report
│   ├── train_test_split.py    #   80/20 train/test split
│   ├── test.py                #   Evaluate the model on the test set
│   ├── single_test.py         #   Classify a single text input
│   └── dataset/csv_dataset/   #   Training/testing CSVs
│
└── pdfs/                      # Source profiles: bad_profile/ good_profile/ excellent_profile/
```

### Not in this repository

These are generated or downloaded locally and are deliberately git-ignored — a fresh clone
will **not** contain them:

| Path | How to obtain it |
|------|------------------|
| `venv/` | Create it (see Setup). |
| `llm_finetune_project/bert-mini-finetuned/` | Train it — see *Fine-tuning BERT*. Required before the BERT and Hybrid modes will run. |
| `vectorstore_openai/`, `vectorstore_gemini/` | Built automatically on first run from `pdfs/`. |
| `outputs/`, `metrics_log/` | Written at runtime as you use the app. |

> **Note on paths:** all scripts resolve their paths **relative to their own location**
> (via `os.path.dirname(__file__)`), so the project runs from anywhere it is cloned — no need
> to edit hardcoded absolute paths.

## 🔧 Setup

### 1. Create a virtual environment & install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

For the BERT fine-tuning sub-project you may want a separate environment:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip3 install -r llm_finetune_project/requirements.txt
```

### 2. Configure API keys

Copy the template and fill in your own keys:

```bash
cp .env.example .env
```

Only the keys for the models you intend to use are needed:

| Variable | Required for |
|----------|--------------|
| `OPENAI_API_KEY`    | GPT-4 **and** the OpenAI embeddings / vector store |
| `GOOGLE_API_KEY`    | Gemini, plus the Gemini embeddings used by DeepSeek, Mistral and Claude |
| `DEEPSEEK_API_KEY`  | DeepSeek |
| `MISTRAL_API_KEY`   | Mistral |
| `ANTHROPIC_API_KEY` | Claude (optionally `ANTHROPIC_MODEL` to pin a version) |

`.env.example` also lists `PINECONE_API_KEY`, `GEMINI_API_KEY`, `LANGCHAIN_API_KEY` and
`HF_TOKEN`. These are **not read by the current code** and can be left blank — Gemini
authenticates through `GOOGLE_API_KEY`.

> ⚠️ **Security:** `.env` is git-ignored — **never commit real keys.** If a key is ever
> committed or otherwise exposed, rotate (regenerate) it. Commit `.env.example` only.

### 3. Organise PDF profiles

Place profile PDFs under `pdfs/` in label-named folders:

- `pdfs/excellent_profile/`
- `pdfs/good_profile/`
- `pdfs/bad_profile/`

Optional path overrides via environment variables (defaults shown):

| Variable             | Default                                         |
|----------------------|-------------------------------------------------|
| `BERT_MODEL_PATH`    | `llm_finetune_project/bert-mini-finetuned`      |
| `HYBRID_METRICS_FILE`| `outputs/hybrid_metrics.csv`                    |

## ▶️ Running

Launch the integrated system (starts the Gradio web interface):

```bash
python3 integrated_system.py
```

The UI is served at **http://127.0.0.1:7860**. First startup takes a while: it parses the
PDFs, builds the FAISS index and loads the BERT weights.

The RAG-only system can also be run on its own:

```bash
python3 app.py
```

## 🚀 Features

- **Single Query Analysis** — BERT-only (fast), RAG-only (context-rich), or Hybrid.
- **Batch Processing** — Upload a CSV of queries; results are written to `outputs/batch_results_<timestamp>.csv`.
- **Analytics** — Classification distribution, model agreement rate, processing times, and confidence levels.

## 🧪 Fine-tuning BERT on new data

1. Format data as CSV with `input` and `label` (label ∈ {bad, good, excellent}) columns,
   placed in `llm_finetune_project/dataset/csv_dataset/`.
2. (Optional) Re-create the split: `python llm_finetune_project/train_test_split.py`
3. Train: `python llm_finetune_project/train_finetune.py`
   (output is saved to `llm_finetune_project/bert-mini-finetuned/`)
4. Evaluate: `python llm_finetune_project/test.py`
5. Point inference at the new model via `BERT_MODEL_PATH` if needed.

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| **API key errors** | Ensure `.env` exists and contains valid keys for the models you selected. |
| **Model loading errors** | Train the BERT model first, or set `BERT_MODEL_PATH` to an existing model directory. |
| **Vector store errors** | Delete `vectorstore_openai/` / `vectorstore_gemini/` and let the system rebuild them. |
| **Memory issues** | Reduce the `context_chunks` parameter for large documents. |
| **`OMP: Error #15: Initializing libomp.dylib…`** (macOS crash) | Two OpenMP runtimes are linked in by PyTorch and FAISS. Run with `KMP_DUPLICATE_LIB_OK=TRUE` as a stopgap, or install `faiss-cpu` and `torch` from the same channel to resolve it properly. |
| **Gradio theme/CSS not applied** | Gradio 6 moved `theme` and `css` from the `gr.Blocks()` constructor to `launch()`. |

## 🔭 Future Improvements

- Support for additional LLM providers
- Enhanced analytics visualisation
- Confidence calibration for more reliable assessments
- Active learning to improve accuracy over time
