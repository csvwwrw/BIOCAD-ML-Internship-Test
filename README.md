# 🧠 Alzheimer's Drug Target Discovery Assistant

RAG system to assist researchers in finding new potential targets for Alzheimer's disease drug development.

***

## 📋 Table of Contents

1. [Quick Start (App Launch)](#-quick-start)
2. [Full Reproduction from Scratch](#-full-reproduction-from-scratch)
3. [Technical Details](#-technical-details)

***

## 🚀 Quick Start

If only want to launch the app:

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 2: Launch Application

```bash
streamlit run app.py
```

App opens in browser at `http://localhost:8501`

### Step 3: Configure API Key in Interface

API keys are configured directly in the app:

1. Open sidebar **⚙️ API Configuration**
2. Select LLM provider: **Perplexity**, **OpenAI**, or **Anthropic**
3. **Paste your API key** in the input field
4. Select model
5. **Start asking questions!**

***

## 🔬 Full Reproduction from Scratch

Reproduce entire pipeline: data collection → processing → training → EDA → app launch.

### 1️⃣ Environment Setup

```bash
# Clone repository (or unpack archive)
cd alzheimer-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```



### 2️⃣ Configuration

Create `.env` file in project root:

```shell
# .env
PPLX_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxx
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx  # optional
```

**Note:** At least one API key required (Perplexity, OpenAI, or Anthropic), but not all three.

**Required:**

- `PUBMED_EMAIL` — your email for PubMed API
- Any `*_API_KEY`

### 3️⃣ Data Collection from PubMed

**Script:** `src/pubmed_retrieval.py`

```bash
python src/pubmed_retrieval.py
```

**What happens:**

- Search ~100 articles for queries:
    - "Alzheimer's disease targets"
    - "Alzheimer therapeutic targets"
    - "Alzheimer drug targets"
- Filter articles with full text (PMC)
- Extract metadata (PMID, DOI, authors, journal, year)
- Download and parse full texts (XML)
- Extract sections: Abstract, Intro, Conclusion, Full_text

**Result:** `data/raw/alzheimer_corpus.csv`

***

### 4️⃣ Chunking (Text Splitting)

**Script:** `src/chunking.py`

```bash
python src/chunking.py
```

**What happens:**

- Load corpus from CSV
- Split texts into chunks (800 chars size, 150 overlap)
- Save metadata for each chunk (PMID, section, year, journal, MeSH terms)

**Result:** `data/processed/chunks.pkl`

***

### 5️⃣ Vectorization (Embedding Creation)

**Script:** `src/vectorizing.py`

```bash
python src/vectorizing.py
```

**What happens:**

- Load `sentence-transformers/allenai-specter` model (scientific model)
- Generate embeddings for all chunks
- Build FAISS index for fast search
- Save vector database

**Result:**

- `data/processed/alzheimer_vector_store_faiss.index`
- `data/processed/alzheimer_vector_store_chunks.pkl`

**⏱️ Time:** 50-60 minutes (CPU)

***

### 6️⃣ EDA (Exploratory Data Analysis)

**Script:** `src/eda_eval.py`

```bash
python src/eda_eval.py
```

**What happens:**

- Data quality analysis (missing values, duplicates, language check)
- Descriptive statistics (text length, year distribution, readability)
- Lexical analysis (TF-IDF, n-grams, word clouds)
- Semantic analysis (embeddings, UMAP/t-SNE, BERTopic topic modeling)
- RAG readiness assessment (chunking, BM25 tests)
- **Optional:** Biomedical NER (if scispacy installed):
    - Extract genes, proteins, drugs, diseases
    - Build knowledge graph
    - Druggability analysis

**Result:** Reports and visualizations in `eda_report/`:

- `data_quality.txt`
- `descriptive_stats.txt`, `descriptive_stats.png`
- `lexical_analysis.txt`, `lexical_analysis.png`
- `semantic_analysis.txt`, `semantic_analysis.png`
- `pre_rag.txt`, `pre_rag.png`
- `bio_ner.json`, `bio_ner.png` (if scispacy installed)
- `mesh_terms.txt`, `mesh_terms.png`
- `citation_density.png`

**⏱️ Time:** 10-30 minutes (depends on scispacy)

***

### 7️⃣ Launch Application

```bash
streamlit run app.py
```

Open browser: `http://localhost:8501`

**App Features:**

- LLM provider selection (Perplexity, OpenAI, Anthropic)
- Search parameters:
    - Top-K results (5-20)
    - Minimum publication year
    - Hybrid search (BM25 + Dense)
- Researcher question input
- Display found sources (PMID, year, section, relevance)
- Generate answer with source citations

**Example Questions:**

```
What are potential targets for Alzheimer's disease treatment?
Are the targets druggable with small molecules, biologics, or other modalities?
What additional studies are needed to advance these targets?
```

***

## 🔧 Technical Details

### RAG Architecture

1. **Retrieval:**
    - Hybrid search: BM25 (sparse) + SPECTER embeddings (dense)
    - Weighting: α=0.6 dense, (1-α)=0.4 sparse
    - Filter by publication year and section
2. **Chunking:**
    - Size: 800 characters
    - Overlap: 150 characters
    - Preserve paragraph/sentence boundaries
3. **Generation:**
    - 3 providers: Perplexity, OpenAI, Anthropic
    - Context: up to 7 papers, 4 chunks per paper
    - Temperature: 0.3 (deterministic)
    - Max tokens: 1500

### Models

- **Embeddings:** `allenai-specter` (scientific papers, 768 dim)
- **Topic Modeling:** BERTopic + KeyBERT
- **LLM:**
    - Perplexity: `sonar-pro`
    - OpenAI: `gpt-4o-mini`
    - Anthropic: `claude-3-5-sonnet`


### Quality Metrics

- Average relevance (retrieval score)
- Silhouette score for topic modeling
- BM25 baseline comparison
- Source citation ([PMID:XXXXX])

***

## 📚 Data Sources

- **PubMed:** articles 2020-2026
- **PMC (PubMed Central):** full texts open access
- **MeSH Terms:** medical taxonomy

***

## 📝 License

MIT License (see LICENSE)

***

## 👤 Author

Project completed for ML Intern Test Task 2026.

***

