
# Quick Setup Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install anthropic openai chromadb sentence-transformers pypdf markdown python-frontmatter
```

### Step 2: Set Up LLM Access

Choose ONE option:

#### Option A: Claude (Recommended)
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LLM_PROVIDER="claude"
```

#### Option B: Ollama (Free, Local)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2
export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434/v1"
```

#### Option C: OpenAI/GitHub Copilot
```bash
export OPENAI_API_KEY="your-openai-api-key"
export LLM_PROVIDER="openai"
```

### Step 3: Create Project Structure

```bash
mkdir literature-review-copilot
cd literature-review-copilot

# Create the copilot.py file (copy from MVP artifact)
# Save it as copilot.py

# Create necessary directories (auto-created by script, but good to know)
mkdir documents vector_db assets
```

### Step 4: Test with a Sample PDF

```bash
# Ingest a PDF
python copilot.py ingest path/to/your/paper.pdf

# This will:
# 1. Convert PDF → Markdown
# 2. Extract metadata
# 3. Save to documents/paper.md
# 4. Index in vector database

# Generate AI comments
python copilot.py process documents/paper.md

# Search your documents
python copilot.py search "machine learning"
```

---

## 📋 Configuration File (Optional)

Create `config.json` for easier configuration:

```json
{
  "llm_provider": "claude",
  "anthropic_api_key": "your-key-here",
  "docs_dir": "./documents",
  "db_dir": "./vector_db",
  "assets_dir": "./assets",
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 500,
  "max_tokens": 1000
}
```

**Note**: You'll need to modify the Config class to load from this file.

---

## 🧪 Testing the System

### Test 1: Basic Ingestion
```bash
# Get a sample paper (e.g., from arXiv)
wget https://arxiv.org/pdf/1706.03762.pdf -O attention_paper.pdf

# Ingest it
python copilot.py ingest attention_paper.pdf

# Check the output
cat documents/attention_paper.md
```

### Test 2: Search Functionality
```bash
# Search for concepts
python copilot.py search "transformer architecture"
python copilot.py search "attention mechanism"
```

### Test 3: AI Comments
```bash
# Generate AI comments
python copilot.py process documents/attention_paper.md

# Check the comments
grep -A 5 "AI_COMMENT" documents/attention_paper.md
```

---

## 🔧 Troubleshooting

### Issue: ChromaDB Installation Fails
```bash
# Try installing with specific version
pip install chromadb==0.4.22

# Or use FAISS as alternative (modify RAGSystem class)
pip install faiss-cpu
```

### Issue: PDF Conversion Errors
```bash
# Install additional PDF tools
pip install PyMuPDF  # Better PDF extraction
pip install pdfplumber  # Alternative PDF parser
```

### Issue: LLM API Errors
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test API directly
python -c "from anthropic import Anthropic; print(Anthropic().models.list())"
```

### Issue: Embedding Model Download
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## 📊 Expected Output Examples

### After Ingestion
```
📄 Ingesting: paper.pdf
✓ Saved: documents/paper.md
✓ Indexed 12 chunks from paper.md

✓ Document ready at: documents/paper.md
```

### Markdown File Structure
```markdown
---
title: Attention Is All You Need
source_file: paper.pdf
authors:
  - Vaswani et al.
date_added: 2024-01-15T10:30:00
tags: []
doi: ""
---

## Page 1

Abstract content here...

## Page 2

Introduction content...
```

### After AI Comment Generation
```markdown
...content...

<!-- AI_COMMENT: {
  "id": "a3f5d891",
  "timestamp": "2024-01-15T10:35:00",
  "target_section": "Introduction",
  "type": "analysis",
  "explanation": "This section introduces the transformer architecture...",
  "related_concepts": ["self-attention", "encoder-decoder"],
  "questions": ["How does this compare to RNNs?"]
} -->
```

---

## 🎯 Next Development Steps

Once the MVP is working, here's what to build next:

### 1. Enhanced PDF Parser (2-3 hours)
- Better section detection
- Extract figures and tables
- Preserve formatting

### 2. Git Integration (1-2 hours)
```python
import git

def init_git_repo(docs_dir):
    repo = git.Repo.init(docs_dir)
    repo.index.add(['*.md'])
    repo.index.commit("Initial commit")
```

### 3. Web Interface (Day 1: Basic Flask/FastAPI)
```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/ingest")
async def ingest_document(file: UploadFile):
    # Handle upload and ingestion
    pass

@app.get("/search")
async def search_documents(query: str):
    # Return search results as JSON
    pass
```

### 4. VS Code Extension (Week 1: Basic Structure)
```typescript
// extension.ts
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    let disposable = vscode.commands.registerCommand('litreview.search', () => {
        // Call Python backend API
        vscode.window.showInformationMessage('Searching...');
    });
    
    context.subscriptions.push(disposable);
}
```

---

## 🔍 Validation Checklist

Before moving to Phase 2, verify:

- [ ] Can ingest at least 3 different PDFs successfully
- [ ] Markdown files have proper frontmatter
- [ ] Vector database contains chunks (check `vector_db/` directory size)
- [ ] Search returns relevant results
- [ ] AI comments are generated and appended
- [ ] Comments have proper JSON structure
- [ ] HUMAN_COMMENT and AI_COMMENT are separate
- [ ] No errors in console output

---

## 💾 Backup & Version Control

### Initialize Git (Recommended)
```bash
cd literature-review-copilot
git init
git add copilot.py
git commit -m "Initial MVP implementation"

# Add .gitignore
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
.env
vector_db/
.DS_Store
EOF

git add .gitignore
git commit -m "Add gitignore"
```

### Backup Vector Database
```bash
# The vector_db folder contains your embeddings
# Back it up periodically
tar -czf vector_db_backup_$(date +%Y%m%d).tar.gz vector_db/
```

---

## 🤝 Getting Help

### Common Commands Reference
```bash
# Activate environment
source venv/bin/activate

# Run ingestion
python copilot.py ingest <pdf_path>

# Process documents
python copilot.py process <md_path>

# Search
python copilot.py search "<query>"

# Check Python environment
pip list | grep -E "anthropic|chromadb|sentence"

# View logs (add logging to script)
python copilot.py ingest paper.pdf 2>&1 | tee ingestion.log
```

### Debug Mode
Add to your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📚 Further Reading

- **RAG Systems**: [LangChain Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- **Vector Databases**: [ChromaDB Guide](https://docs.trychroma.com/)
- **VS Code Extensions**: [Extension API](https://code.visualstudio.com/api)
- **ORKG API**: [Integration Guide](https://orkg.org/help-center)

---

## ✅ Success Criteria

You'll know the MVP is working when:
1. You can convert 5+ papers to searchable Markdown
2. Search returns contextually relevant chunks
3. AI comments provide useful insights with references
4. The system runs without errors for 30+ minutes
5. You can find connections between papers automatically

**Once these work, you're ready for Phase 2! 🎉**
