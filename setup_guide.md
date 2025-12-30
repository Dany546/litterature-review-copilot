# RAG-Anything Setup Guide for Literature Review Copilot

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Core Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install RAG-Anything with all features
pip install raganything[all]

# Install LightRAG (the RAG engine)
pip install lightrag-hku

# Install your preferred LLM provider
pip install anthropic  # For Claude (recommended)
# OR
pip install openai     # For OpenAI/GitHub Copilot
```

### Step 2: Install MinerU (High-Quality PDF Parser)

**Option A: Full Installation (Recommended)**
```bash
# Install MinerU for best document parsing
pip install magic-pdf[full]

# Verify installation
mineru --version

# Models download automatically on first use
```

**Option B: Alternative - Docling Parser**
```bash
# Lighter alternative to MinerU
pip install docling

# Set parser type to "docling" in config
```

### Step 3: Set Up Your LLM Provider

Choose ONE:

#### Claude (Recommended - Best for Research)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export LLM_PROVIDER="claude"

# Test it
python -c "from anthropic import Anthropic; print('✓ Claude configured')"
```

#### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
export LLM_PROVIDER="openai"

# Test it
python -c "from openai import OpenAI; print('✓ OpenAI configured')"
```

#### Ollama (Free, Local)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2

export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434/v1"

# Test it
ollama list
```

### Step 4: Create Project & Test

```bash
# Create project directory
mkdir literature-review-copilot
cd literature-review-copilot

# Save the copilot.py script from the artifact

# Test with a sample PDF
python copilot.py ingest sample_paper.pdf
python copilot.py search "main contributions" hybrid
```

---

## 📋 Detailed Installation Guide

### System Requirements

- **Python**: 3.10 or higher (3.12 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5-10GB for models and documents
- **OS**: Linux, macOS, or Windows with WSL2

### Complete Installation

```bash
# 1. Create clean environment
python3.12 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install RAG-Anything ecosystem
pip install raganything[all]  # Includes image, text, and all parsers
pip install lightrag-hku

# 4. Install LLM providers (install all or just what you need)
pip install anthropic openai

# 5. Install MinerU for best parsing
pip install magic-pdf[full]

# 6. Optional: Additional tools
pip install python-frontmatter  # For YAML metadata
pip install gitpython           # For Git integration (Phase 3)
```

### Verify Installation

```bash
# Check RAG-Anything
python -c "from raganything import RAGAnything; print('✓ RAG-Anything installed')"

# Check LightRAG
python -c "from lightrag import LightRAG; print('✓ LightRAG installed')"

# Check MinerU
mineru --version
python -c "from raganything import RAGAnything; rag = RAGAnything(); print('✓ MinerU:', 'OK' if rag.check_mineru_installation() else 'Missing')"

# Check LLM provider
python -c "from anthropic import Anthropic; print('✓ Anthropic SDK installed')"
```

---

## 🔧 Configuration

### Environment Variables (Recommended)

Create `.env` file:

```bash
# .env file
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Configuration Options

The system supports multiple configuration approaches:

1. **Environment Variables** (shown above)
2. **Direct in Code** (modify Config class)
3. **Config File** (create `config.json`)

---

## 🧪 Testing the System

### Test 1: Basic RAG-Anything Functionality

```bash
# Create test script: test_rag.py
cat > test_rag.py << 'EOF'
import asyncio
from raganything import RAGAnything
from lightrag.llm.openai import openai_complete_if_cache

async def test():
    print("Initializing RAG-Anything...")
    
    rag = RAGAnything(
        working_dir="./test_rag_storage",
        llm_model_func=lambda prompt, **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            api_key="your-openai-key",
            **kwargs
        )
    )
    
    print("✓ RAG-Anything initialized")
    
    # Test with sample text
    await rag.ainsert("The transformer architecture revolutionized NLP.")
    result = await rag.aquery("What revolutionized NLP?", mode="hybrid")
    print(f"Query result: {result}")

asyncio.run(test())
EOF

python test_rag.py
```

### Test 2: Document Ingestion

```bash
# Download a sample paper
wget https://arxiv.org/pdf/1706.03762.pdf -O attention.pdf

# Ingest it
python copilot.py ingest attention.pdf

# Should see:
# 📄 Ingesting: attention.pdf
# ✓ Successfully indexed: attention.pdf
```

### Test 3: Query Modes

```bash
# Test different query modes
python copilot.py search "transformer architecture" hybrid
python copilot.py search "self-attention mechanism" local
python copilot.py search "what are the key innovations" global
python copilot.py search "multi-head attention" naive
```

### Test 4: Multimodal Parsing

```bash
# Ingest a paper with figures and tables
python copilot.py ingest paper_with_figures.pdf

# Check if figures were extracted
ls rag_storage/images/  # Should see extracted images

# Query with vision enhancement (if vision model configured)
python copilot.py search "explain the architecture diagram" hybrid
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'raganything'"

**Solution:**
```bash
pip install raganything[all]
# Or if that fails:
pip install git+https://github.com/HKUDS/RAG-Anything.git
```

### Issue: MinerU Installation Fails

**Solution 1: Use Docling instead**
```bash
pip install docling
# Change parser_type to "docling" in Config class
```

**Solution 2: Manual MinerU installation**
```bash
# Install dependencies first
pip install torch torchvision
pip install magic-pdf[full] --no-deps
pip install -r requirements.txt  # From MinerU repo
```

### Issue: "CUDA out of memory" or GPU Issues

**Solution:**
```bash
# Use CPU-only versions
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or reduce batch size in config
```

### Issue: RAG-Anything Queries Return Empty Results

**Possible causes:**
1. Documents not indexed yet
2. Wrong working directory
3. LLM API key issues

**Solution:**
```bash
# Check if documents are indexed
ls -la rag_storage/
# Should see: kv_store_*.json, vdb_*.json, graph files

# Verify API key works
python -c "from anthropic import Anthropic; c = Anthropic(); print(c.models.list())"

# Re-index documents
rm -rf rag_storage/
python copilot.py ingest your_paper.pdf
```

### Issue: "Rate limit exceeded" or API Errors

**Solution:**
```bash
# Switch to Ollama for free local inference
ollama pull llama3.2
export LLM_PROVIDER="ollama"

# Or add retry logic / delays
```

### Issue: MinerU Models Not Downloading

**Solution:**
```bash
# Manually download models
python -c "
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
model = ModelSingleton()
model.get_model(model_name='DocLayout')
"

# Or set model path manually
export MINERU_MODEL_PATH="/path/to/models"
```

---

## 📊 Understanding RAG-Anything Output

### Directory Structure After Ingestion

```
rag_storage/
├── graph_chunk_entity_relation.graphml  # Knowledge graph
├── kv_store_full_docs.json              # Full document storage
├── kv_store_text_chunks.json            # Text chunks
├── kv_store_llm_response_cache.json     # LLM response cache
├── vdb_chunks.json                      # Vector embeddings
└── images/                              # Extracted images
    ├── doc1_img1.jpg
    └── doc1_img2.jpg
```

### Query Modes Explained

| Mode | Use Case | Speed | Quality |
|------|----------|-------|---------|
| **hybrid** | General queries, best balance | Medium | High |
| **local** | Related concepts, entity focus | Fast | Medium-High |
| **global** | Broad questions, summaries | Slow | High |
| **naive** | Simple keyword search | Very Fast | Medium |

### Example Query Patterns

```bash
# Conceptual questions → hybrid
python copilot.py search "What is attention mechanism?" hybrid

# Related work → local
python copilot.py search "Papers similar to transformer" local

# Broad synthesis → global
python copilot.py search "Summarize all papers on NLP" global

# Specific facts → naive
python copilot.py search "batch size 256" naive
```

---

## 🎯 Performance Optimization

### For Faster Indexing

```python
# In RAGAnythingWrapper.__init__
self.rag = RAGAnything(
    working_dir=str(self.config.rag_storage_dir),
    llm_model_func=llm_func,
    addon_params={
        "parser": "docling",  # Faster than MinerU
        "chunk_size": 1000,   # Larger chunks = faster
    }
)
```

### For Better Quality

```python
# Use MinerU with GPU
addon_params={
    "parser": "mineru",
    "device": "cuda",     # Enable GPU
    "chunk_size": 512,    # Smaller chunks = better precision
}
```

### For Cost Savings

```bash
# Use Ollama locally
export LLM_PROVIDER="ollama"
ollama pull llama3.2

# Or use smaller OpenAI models
# In Config._get_llm_model(): return "gpt-4o-mini"
```

---

## 🔍 Advanced Features

### Vision-Enhanced Queries

```python
# Enable VLM for analyzing figures
result = await rag.query(
    "Explain the architecture diagram in Figure 2",
    mode="hybrid",
    vlm_enhanced=True  # Analyzes images in context
)
```

### Batch Document Processing

```bash
# Process multiple papers
for pdf in papers/*.pdf; do
    python copilot.py ingest "$pdf"
done

# Then query across all
python copilot.py search "compare all approaches" global
```

### Custom Chunking Strategy

```python
# In RAGAnything initialization
self.rag = RAGAnything(
    working_dir="./rag_storage",
    llm_model_func=llm_func,
    addon_params={
        "chunk_token_size": 512,
        "chunk_overlap_token_size": 50,
        "entity_extract_max_tokens": 4000
    }
)
```

---

## 📚 Next Steps After Setup

1. **Ingest Your First Papers** (3-5 PDFs)
   ```bash
   python copilot.py ingest paper1.pdf
   python copilot.py ingest paper2.pdf
   ```

2. **Test Different Query Modes**
   ```bash
   python copilot.py search "your research question" hybrid
   ```

3. **Visualize Knowledge Graph** (Optional)
   ```bash
   pip install networkx matplotlib
   # Load graph_chunk_entity_relation.graphml in Gephi or NetworkX
   ```

4. **Move to Phase 2**: Add Markdown export and metadata

---

## 🤝 Getting Help

### Documentation
- [RAG-Anything Docs](https://github.com/HKUDS/RAG-Anything)
- [LightRAG Docs](https://github.com/HKUDS/LightRAG)
- [MinerU Docs](https://github.com/opendatalab/MinerU)

### Common Resources
```bash
# Check versions
pip list | grep -E "raganything|lightrag|magic-pdf"

# Get system info
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test LLM connectivity
python -c "from anthropic import Anthropic; Anthropic().messages.create(model='claude-sonnet-4-20250514', max_tokens=10, messages=[{'role':'user','content':'hi'}])"
```

### Debug Mode
```python
# Add logging to copilot.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## ✅ Installation Checklist

Before proceeding:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] RAG-Anything installed (`pip install raganything[all]`)
- [ ] LightRAG installed (`pip install lightrag-hku`)
- [ ] LLM provider SDK installed (anthropic/openai)
- [ ] MinerU or Docling installed
- [ ] Environment variables set
- [ ] API keys tested and working
- [ ] Sample PDF successfully ingested
- [ ] Query returns results
- [ ] Knowledge graph created in `rag_storage/`

**If all checked ✓ → You're ready to build! 🎉**
