# litterature-review-copilot
Documents annotation, with citation helper and RAG-based documents linking and description.

1️⃣ Environment Setup
- Install Python ≥ 3.12 and Node.js
- Install VS Code + GitLens + Markdown All-in-One
- Initialize local Git repository for documents
- Set up local LLM backend: Ollama or GitHub Copilot
- Clone RAG-Anything repository and configure locally
- Choose vector DB: Chroma, FAISS, or Qdrant

2️⃣ Document Ingestion
- Convert PDFs / TeX / Websites → Markdown (Pandoc or Marker)
- Normalize metadata in YAML frontmatter:
  title, authors, DOI, tags
- Save images/assets locally
- Add Markdown files to Git repository

3️⃣ Semantic Indexing & RAG Setup
- Embed documents at paragraph, section, and document level
- Store embeddings in local vector DB
- Test semantic retrieval of documents
- Expose local API for VS Code queries to RAG-Anything

4️⃣ Comment & Annotation System
- Use HTML-style comments in Markdown:
  # <!-- HUMAN_COMMENT: ... -->
  # <!-- AI_COMMENT: ... -->
- Extend metadata for linking to file hunks:
  type, target_id (section/paragraph), start_line, timestamp, linked_papers, concepts
- Parse and filter comments by type
- Ensure AI cannot overwrite HUMAN_COMMENT
- Add Git versioning support

5️⃣ Concept Lens Layer
- Extract concepts from new documents using LLM
- Embed concepts as semantic probes
- Compute activation scores on existing documents
- Generate AI_COMMENT suggestions for new concepts
- Store concept metadata with timestamps and IDs

6️⃣ AI-Generated Comment System
- Python pipeline for AI_COMMENT generation:
  explain text, suggest links, detect misunderstandings, suggest keywords
- Integrate with RAG context retrieval
- Append-only insertion into Markdown
- Manual review / accept-reject workflow in VS Code

7️⃣ ORKG Integration for Suggestions
- Map local documents to ORKG entries via DOI/title
- Query ORKG API / ORKG Ask for:
  related papers, keywords, topics, methods
- Display in VS Code UI:
  Suggested papers to read
  Suggested keywords for search
- Feed ORKG suggestions into RAG context and AI_COMMENT generation

8️⃣ VS Code Extension Features
- File watcher for new/updated Markdown files
- Command palette actions:
  Query RAG context, Generate AI comments, Show related papers/keywords
- Sidebar panel:
  Human vs AI comments, Concept lens activations, Paper links / keywords
- Inline decorations & hover tooltips for linked comments
- Accept/edit/delete AI comments UI

9️⃣ Linking Comments to chunks
- Use line numbers, target_id, or character offsets to link comments
- Assign target_id to each section or paragraph
- Highlight linked sections when selecting a comment
- Ensure AI_COMMENTs dynamically link correctly if lines shift

🔟 Git Integration
- Commit Markdown + assets
- Track AI/Human comment changes
- Optional: auto-commit AI_COMMENT suggestions
- Use GitLens to inspect history of comments

1️⃣1️⃣ Optional / Future Enhancements
- Multi-layer concept lens re-evaluation triggered by new documents
- Incremental embedding updates without reindexing entire DB
- Local caching of ORKG queries
- Drag/drop links or inline annotation enhancements
- Section-level summarization via LLM
