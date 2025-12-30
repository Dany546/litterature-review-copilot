"""
Literature Review Copilot - MVP with RAG-Anything Integration
Built on RAG-Anything (LightRAG + MinerU) for multimodal document processing
"""

import os
import json
import re
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib

# Required installations:
# pip install raganything lightrag-hku anthropic openai

try:
    from raganything import RAGAnything
except ImportError:
    print("ERROR: raganything not installed. Run: pip install raganything[all]")
    print("For full support also install: pip install lightrag-hku anthropic openai")
    exit(1)

try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
except ImportError:
    print("ERROR: lightrag-hku not installed. Run: pip install lightrag-hku")
    exit(1)


class Config:
    """Configuration for Literature Review Copilot using RAG-Anything"""
    
    def __init__(self):
        # Directory structure
        self.docs_dir = Path("./documents")
        self.rag_storage_dir = Path("./rag_storage")
        self.assets_dir = Path("./assets")
        self.imports_dir = Path("./imports")  # For PDFs before conversion
        
        # LLM Configuration - RAG-Anything supports multiple providers
        self.llm_provider = os.getenv("LLM_PROVIDER", "claude")  # claude, openai, ollama
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
        # RAG-Anything settings
        self.llm_model = self._get_llm_model()
        self.vision_model = self._get_vision_model()
        self.embedding_model = "text-embedding-3-small"  # For OpenAI
        
        # MinerU parser settings (default in RAG-Anything)
        self.parser_type = "mineru"  # or "docling"
        
        # Create directories
        for dir_path in [self.docs_dir, self.rag_storage_dir, self.assets_dir, self.imports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_llm_model(self) -> str:
        """Get LLM model name based on provider"""
        if self.llm_provider == "claude":
            return "claude-sonnet-4-20250514"
        elif self.llm_provider == "openai":
            return "gpt-4o-mini"
        elif self.llm_provider == "ollama":
            return "llama3.2"  # or your preferred Ollama model
        return "gpt-4o-mini"
    
    def _get_vision_model(self) -> str:
        """Get vision model for multimodal support"""
        if self.llm_provider == "claude":
            return "claude-sonnet-4-20250514"  # Claude supports vision
        elif self.llm_provider == "openai":
            return "gpt-4o"  # Vision model
        return "gpt-4o"


class RAGAnythingWrapper:
    """Wrapper for RAG-Anything with support for multiple LLM providers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rag = None
        self._initialize_rag()
    
    def _get_anthropic_llm_func(self):
        """Create LLM function for Anthropic Claude"""
        from anthropic import Anthropic
        client = Anthropic(api_key=self.config.claude_api_key)
        
        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            messages = []
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=kwargs.get("max_tokens", 4000),
                system=system_prompt or "",
                messages=messages
            )
            return response.content[0].text
        
        return llm_func
    
    def _get_anthropic_vision_func(self):
        """Create vision function for Claude"""
        from anthropic import Anthropic
        client = Anthropic(api_key=self.config.claude_api_key)
        
        def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            content = []
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                })
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            response = client.messages.create(
                model=self.config.vision_model,
                max_tokens=kwargs.get("max_tokens", 4000),
                system=system_prompt or "",
                messages=messages
            )
            return response.content[0].text
        
        return vision_func
    
    def _get_openai_llm_func(self):
        """Use OpenAI LLM function from LightRAG"""
        return lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            self.config.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.config.openai_api_key,
            **kwargs
        )
    
    def _get_openai_vision_func(self):
        """Create vision function for OpenAI"""
        from openai import OpenAI
        client = OpenAI(api_key=self.config.openai_api_key)
        
        def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            content = [{"type": "text", "text": prompt}]
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
            
            messages.append({"role": "user", "content": content})
            
            response = client.chat.completions.create(
                model=self.config.vision_model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4000)
            )
            return response.choices[0].message.content
        
        return vision_func
    
    def _get_ollama_llm_func(self):
        """Create LLM function for Ollama"""
        from openai import OpenAI
        client = OpenAI(
            api_key="ollama",  # Ollama doesn't need real API key
            base_url=self.config.ollama_base_url
        )
        
        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4000)
            )
            return response.choices[0].message.content
        
        return llm_func
    
    def _get_embedding_func(self):
        """Get embedding function based on provider"""
        if self.config.llm_provider in ["openai", "ollama"]:
            # Use OpenAI embeddings or Ollama embeddings
            return EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=self.config.embedding_model,
                    api_key=self.config.openai_api_key if self.config.llm_provider == "openai" else "ollama",
                    base_url=self.config.ollama_base_url if self.config.llm_provider == "ollama" else None
                )
            )
        return None  # Use default
    
    def _initialize_rag(self):
        """Initialize RAG-Anything with appropriate LLM provider"""
        print(f"🔧 Initializing RAG-Anything with {self.config.llm_provider}...")
        
        # Get LLM and vision functions based on provider
        if self.config.llm_provider == "claude":
            llm_func = self._get_anthropic_llm_func()
            vision_func = self._get_anthropic_vision_func()
        elif self.config.llm_provider == "openai":
            llm_func = self._get_openai_llm_func()
            vision_func = self._get_openai_vision_func()
        elif self.config.llm_provider == "ollama":
            llm_func = self._get_ollama_llm_func()
            vision_func = None  # Most Ollama models don't support vision yet
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        # Initialize RAG-Anything
        self.rag = RAGAnything(
            working_dir=str(self.config.rag_storage_dir),
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=self._get_embedding_func(),
            addon_params={"parser": self.config.parser_type}
        )
        
        print("✓ RAG-Anything initialized successfully")
    
    async def index_document(self, file_path: Path):
        """Index a document using RAG-Anything's multimodal parser"""
        print(f"📄 Indexing document: {file_path.name}")
        
        try:
            # RAG-Anything automatically handles PDF, DOCX, PPTX, images, etc.
            await self.rag.ainsert_file(str(file_path))
            print(f"✓ Successfully indexed: {file_path.name}")
        except Exception as e:
            print(f"✗ Error indexing {file_path.name}: {e}")
    
    async def query(self, query: str, mode: str = "hybrid", vlm_enhanced: bool = False) -> Dict:
        """Query the RAG system with different modes"""
        try:
            result = await self.rag.aquery(
                query,
                mode=mode,  # hybrid, local, global, naive
                vlm_enhanced=vlm_enhanced  # Enable vision analysis if needed
            )
            return {"result": result, "mode": mode}
        except Exception as e:
            print(f"✗ Query error: {e}")
            return {"result": "", "mode": mode, "error": str(e)}


class CommentParser:
    """Parse and manage comments in Markdown files"""
    
    HUMAN_PATTERN = r'<!--\s*HUMAN_COMMENT:\s*(.+?)\s*-->'
    AI_PATTERN = r'<!--\s*AI_COMMENT:\s*(.+?)\s*-->'
    
    @staticmethod
    def extract_comments(markdown_content: str) -> Dict[str, List[Dict]]:
        """Extract all comments from Markdown"""
        comments = {"human": [], "ai": []}
        
        for match in re.finditer(CommentParser.HUMAN_PATTERN, markdown_content, re.DOTALL):
            try:
                comment_data = json.loads(match.group(1))
                comments["human"].append(comment_data)
            except json.JSONDecodeError:
                comments["human"].append({"text": match.group(1)})
        
        for match in re.finditer(CommentParser.AI_PATTERN, markdown_content, re.DOTALL):
            try:
                comment_data = json.loads(match.group(1))
                comments["ai"].append(comment_data)
            except json.JSONDecodeError:
                comments["ai"].append({"text": match.group(1)})
        
        return comments
    
    @staticmethod
    def add_ai_comment(markdown_content: str, comment: Dict, line_number: Optional[int] = None) -> str:
        """Add AI comment to Markdown (append-only)"""
        comment_data = {
            "id": hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "line_number": line_number,
            **comment
        }
        
        comment_str = f'\n\n<!-- AI_COMMENT: {json.dumps(comment_data, indent=2)} -->\n'
        return markdown_content + comment_str
    
    @staticmethod
    def remove_ai_comments(markdown_content: str) -> str:
        """Remove all AI comments"""
        return re.sub(CommentParser.AI_PATTERN, '', markdown_content, flags=re.DOTALL)


class AICommentGenerator:
    """Generate AI comments using RAG-Anything context"""
    
    def __init__(self, rag_wrapper: RAGAnythingWrapper):
        self.rag = rag_wrapper
    
    async def generate_comments(self, doc_path: Path, content: str) -> List[Dict]:
        """Generate AI comments for a document"""
        comments = []
        sections = self._split_into_sections(content)
        
        for section in sections[:3]:  # Limit for MVP
            # Query RAG for related content
            context_result = await self.rag.query(
                f"Find related research about: {section['title']}. {section['text'][:500]}",
                mode="hybrid"
            )
            
            # Generate comment based on context
            comment_query = f"""Based on this section and related research:

SECTION: {section['title']}
{section['text'][:1000]}

RELATED RESEARCH:
{context_result['result'][:2000]}

Provide a helpful comment with:
1. Brief explanation of key concepts
2. Links to related papers
3. Potential questions or areas needing clarification

Format as JSON with keys: explanation, related_concepts, questions"""
            
            comment_result = await self.rag.query(comment_query, mode="global")
            
            try:
                comment_data = json.loads(comment_result['result'])
                comments.append({
                    "type": "analysis",
                    "target_section": section["title"],
                    **comment_data
                })
            except:
                comments.append({
                    "type": "analysis",
                    "target_section": section["title"],
                    "text": comment_result['result']
                })
        
        return comments
    
    def _split_into_sections(self, content: str) -> List[Dict]:
        """Split document into sections"""
        sections = []
        current_section = {"title": "Introduction", "text": ""}
        
        for line in content.split('\n'):
            if line.startswith('##'):
                if current_section["text"].strip():
                    sections.append(current_section)
                current_section = {"title": line.strip('#').strip(), "text": ""}
            else:
                current_section["text"] += line + "\n"
        
        if current_section["text"].strip():
            sections.append(current_section)
        
        return sections


class LiteratureReviewCopilot:
    """Main orchestrator using RAG-Anything"""
    
    def __init__(self):
        self.config = Config()
        self.rag = RAGAnythingWrapper(self.config)
        self.parser = CommentParser()
        self.comment_generator = AICommentGenerator(self.rag)
    
    async def ingest_document(self, file_path: Path) -> Path:
        """Ingest any document (PDF, DOCX, PPTX, etc.) using RAG-Anything"""
        print(f"\n📄 Ingesting: {file_path.name}")
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return None
        
        # Index with RAG-Anything (handles conversion automatically)
        await self.rag.index_document(file_path)
        
        # Copy to imports directory for reference
        import_path = self.config.imports_dir / file_path.name
        import shutil
        shutil.copy2(file_path, import_path)
        
        return import_path
    
    async def process_markdown(self, md_path: Path):
        """Process a Markdown document and generate AI comments"""
        print(f"\n🤖 Processing: {md_path.name}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove old AI comments
        content = self.parser.remove_ai_comments(content)
        
        # Generate new comments
        comments = await self.comment_generator.generate_comments(md_path, content)
        
        # Add comments
        for comment in comments:
            content = self.parser.add_ai_comment(content, comment)
        
        # Save
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Added {len(comments)} AI comments")
    
    async def search(self, query: str, mode: str = "hybrid"):
        """Search across all documents"""
        print(f"\n🔍 Searching ({mode} mode): {query}")
        result = await self.rag.query(query, mode=mode)
        print(f"\nResult:\n{result['result']}")
        return result


async def main():
    """CLI Interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("""
Literature Review Copilot - RAG-Anything Integration

Usage:
  python copilot.py ingest <file_path>        - Ingest PDF/DOCX/PPTX
  python copilot.py process <md_path>         - Add AI comments
  python copilot.py search "<query>" [mode]   - Search (modes: hybrid/local/global/naive)
  
Example:
  python copilot.py ingest paper.pdf
  python copilot.py search "transformer architecture" hybrid
        """)
        return
    
    copilot = LiteratureReviewCopilot()
    command = sys.argv[1]
    
    if command == "ingest" and len(sys.argv) > 2:
        file_path = Path(sys.argv[2])
        await copilot.ingest_document(file_path)
    
    elif command == "process" and len(sys.argv) > 2:
        md_path = Path(sys.argv[2])
        await copilot.process_markdown(md_path)
    
    elif command == "search" and len(sys.argv) > 2:
        query = sys.argv[2]
        mode = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
        await copilot.search(query, mode)
    
    else:
        print("✗ Invalid command")


if __name__ == "__main__":
    import shutil  # Add this import at the top if not present
    asyncio.run(main())
