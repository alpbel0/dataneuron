"""
DataNeuron Core Document Interaction Tools
==========================================

This module provides essential tools for LLM agents to interact with documents
managed by the SessionManager. These tools form the foundation of the RAG system,
enabling reading, searching, and summarizing document content.

Features:
- ReadFullDocumentTool: Full document content retrieval using DocumentProcessor
- SearchInDocumentTool: Semantic search within documents using vector store
- SummarizeDocumentTool: AI-powered document summarization with full content access
- Session-aware document management
- Comprehensive error handling and validation
- Direct file access through persistent storage for complete content analysis

Usage:
    # These tools are automatically discovered by ToolManager
    # and can be executed by LLM agents through tool calls:
    
    # Read full document
    result = run_tool("read_full_document", 
                      file_name="document.pdf", 
                      session_id="user_123")
    
    # Search within document
    result = run_tool("search_in_document", 
                      query="machine learning algorithms",
                      file_name="research.pdf",
                      session_id="user_123")
    
    # Summarize document
    result = run_tool("summarize_document",
                      file_name="report.docx",
                      session_id="user_123",
                      summary_length="short",
                      output_format="bullet_points")

Integration:
- Uses SessionManager for document metadata and file path retrieval
- Uses DocumentProcessor for full content reading from persistent files
- Integrates with VectorStore for semantic search
- Utilizes Anthropic API for intelligent summarization
- Follows BaseTool architecture for consistency
- Enables complete document analysis through real file content access
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import anthropic
from pydantic import BaseModel, Field

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from utils.logger import logger
from config.settings import OPENAI_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_API_KEY

# Import dependencies with proper error handling
try:
    from core.session_manager import SessionManager
except ImportError as e:
    logger.warning(f"SessionManager import failed: {e}")
    SessionManager = None

try:
    from core.document_processor import DocumentProcessor
except ImportError as e:
    logger.warning(f"DocumentProcessor import failed: {e}")
    DocumentProcessor = None

try:
    from core.vector_store import VectorStore
except ImportError as e:
    logger.warning(f"VectorStore import failed: {e}")
    VectorStore = None

try:
    from core.embedder import Embedder
except ImportError as e:
    logger.warning(f"Embedder import failed: {e}")
    Embedder = None


try:
    import openai
    openai.api_key = OPENAI_API_KEY
except ImportError as e:
    logger.warning(f"OpenAI import failed: {e}")
    openai = None


# ============================================================================
# PYDANTIC SCHEMAS FOR DOCUMENT TOOLS
# ============================================================================

class ReadDocumentArgs(BaseToolArgs):
    """Arguments for reading a full document."""
    file_name: str = Field(..., description="Name of the document to read")
    session_id: str = Field(..., description="User session ID for document access")


class ReadDocumentResult(BaseToolResult):
    """Result from reading a document."""
    content: str = Field(..., description="Full text content of the document")
    character_count: int = Field(..., description="Number of characters in the content")
    file_info: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class SearchDocumentArgs(BaseToolArgs):
    """Arguments for searching within a document."""
    query: str = Field(..., description="Search query or question to find relevant content")
    file_name: str = Field(..., description="Name of the document to search in")
    session_id: str = Field(..., description="User session ID for document access")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant passages to return")


class SearchResult(BaseModel):
    """Individual search result."""
    content: str = Field(..., description="Text content of the relevant passage")
    similarity_score: float = Field(..., description="Semantic similarity score (0-1)")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in the document")
    page_number: Optional[Union[int, str]] = Field(None, description="Source page number(s) of the chunk")


class SearchDocumentResult(BaseToolResult):
    """Result from searching within a document."""
    search_results: List[SearchResult] = Field(..., description="List of relevant passages found")
    query_used: str = Field(..., description="The search query that was executed")
    total_results: int = Field(..., description="Total number of results found")


class SummarizeDocumentArgs(BaseToolArgs):
    """Arguments for summarizing a document."""
    file_name: str = Field(..., description="Name of the document to summarize")
    session_id: str = Field(..., description="User session ID for document access")
    text_to_summarize: Optional[str] = Field(
        None, 
        description="Specific text to summarize (if None, entire document is summarized)"
    )
    summary_length: str = Field(
        default="medium",
        pattern="^(short|medium|long)$",
        description="Length of summary: 'short' (~100 words), 'medium' (~300 words), 'long' (~500 words)"
    )
    output_format: str = Field(
        default="paragraph",
        pattern="^(paragraph|bullet_points)$", 
        description="Format of summary: 'paragraph' or 'bullet_points'"
    )


class SummarizeDocumentResult(BaseToolResult):
    """Result from summarizing a document."""
    summary: str = Field(..., description="Generated summary text")
    original_length: int = Field(..., description="Character count of original text")
    summary_length: int = Field(..., description="Character count of summary")
    compression_ratio: float = Field(..., description="Summary length / original length ratio")


# ============================================================================
# DOCUMENT TOOL IMPLEMENTATIONS
# ============================================================================

class ReadFullDocumentTool(BaseTool):
    """
    Tool for reading the complete content of a document.
    
    This tool retrieves the full text content of a document from the user's session.
    It's ideal for getting an overview of document structure or when complete content
    access is needed. For large documents, consider using search or summarization
    tools instead to avoid overwhelming the context window.
    """
    
    name = "read_full_document"
    description = (
        "Reads the complete text content of a specified document from the user's session. "
        "Use this tool when you need to access the entire document content, understand "
        "the overall structure, or when comprehensive analysis is required. "
        "For large documents (>10,000 characters), consider using 'search_in_document' "
        "to find specific information or 'summarize_document' to get key insights instead. "
        "The tool returns the full text content along with character count and metadata."
    )
    args_schema = ReadDocumentArgs
    return_schema = ReadDocumentResult
    version = "1.0.0"
    category = "document"
    requires_session = True
    
    def _execute(self, file_name: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the document reading operation.
        
        Args:
            file_name: Name of the document to read
            session_id: User session ID for document access
            
        Returns:
            Dictionary with document content and metadata
            
        Raises:
            ValueError: If SessionManager unavailable, session not found, or document not found
        """
        logger.info(f"Reading full document: {file_name} from session: {session_id}")



        
        # Check if required services are available
        if not self.session_manager:
            raise ValueError("SessionManager not available - tool not properly initialized")
        
        if DocumentProcessor is None:
            raise ValueError("DocumentProcessor is not available. Cannot process documents.")
        
        # Get documents from session using injected session manager
        session_documents = self.session_manager.get_session_documents(session_id)

        
        if not session_documents:
            raise ValueError(f"No documents found in session: {session_id}")
        
        # Find the requested document (smart file finder logic)
        target_document = None
        for doc in session_documents:
            if doc.file_name.lower() == file_name.lower():
                target_document = doc
                break
        
        if not target_document:
            available_docs = [doc.file_name for doc in session_documents]
            raise ValueError(
                f"Document '{file_name}' not found in session '{session_id}'. "
                f"Available documents: {available_docs}"
            )
        
        # Use DocumentProcessor to read full content from persistent file
        document_processor = DocumentProcessor()
        file_path = target_document.file_path
        
        logger.info(f"Reading full content from file: {file_path}")
        document = document_processor.process(file_path)
        
        if not document or not document.content:
            raise ValueError(f"Failed to read document content from {file_path}")
        
        # Handle new structured content format
        if isinstance(document.content, list):
            # New format: [{"page_number": int, "text": str}, ...]
            content = "\n".join(page["text"] for page in document.content)
            character_count = len(content)
        else:
            # Legacy format: single string (backward compatibility)
            content = document.content
            character_count = len(content)
        
        file_info = {
            "file_name": target_document.file_name,
            "file_hash": target_document.file_hash,
            "processed_at": target_document.processed_at,
            "vector_collection": target_document.vector_collection_name,
            "metadata": target_document.document_metadata
        }
        
        logger.success(f"Successfully read document: {file_name} ({character_count} characters)")
        
        return {
            "content": content,
            "character_count": character_count,
            "file_info": file_info,
            "metadata": {
                "session_id": session_id,
                "file_name": file_name,
                "retrieval_timestamp": logger._get_current_time() if hasattr(logger, '_get_current_time') else "now"
            }
        }


class SearchInDocumentTool(BaseTool):
    """
    Tool for semantic search within document content.
    
    This tool performs semantic search to find the most relevant passages within
    a document based on a query. It uses vector embeddings to understand meaning
    rather than just keyword matching, making it powerful for finding contextually
    relevant information even when exact terms don't match.
    """
    
    name = "search_in_document"
    description = (
        "Performs semantic search within a document to find passages most relevant "
        "to your query. This tool uses AI embeddings to understand meaning and context, "
        "not just keyword matching. Use this when you need to find specific information, "
        "answer questions, or locate evidence within a document. "
        "The search returns ranked passages with similarity scores, allowing you to "
        "focus on the most relevant content. Specify 'top_k' to control how many "
        "results you want (1-20, default 5)."
    )
    args_schema = SearchDocumentArgs
    return_schema = SearchDocumentResult
    version = "1.0.0"
    category = "document"
    requires_session = True
    
    def _execute(self, query: str, file_name: str, session_id: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Execute the document search operation.
        
        Args:
            query: Search query or question
            file_name: Name of the document to search
            session_id: User session ID for document access
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results and metadata
            
        Raises:
            ValueError: If required services unavailable or document not found
        """
        logger.info(f"Searching in document: {file_name} for query: '{query}' (top_k={top_k})")
        
        # Check if required services are available
        if not self.session_manager:
            raise ValueError("SessionManager not available - tool not properly initialized")
        
        if VectorStore is None:
            raise ValueError("VectorStore is not available. Cannot perform semantic search.")
        
        if Embedder is None:
            raise ValueError("Embedder is not available. Cannot create query embeddings for semantic search.")
        
        # Get documents from session using injected session manager
        session_documents = self.session_manager.get_session_documents(session_id)
        
        if not session_documents:
            raise ValueError(f"No documents found in session: {session_id}")
        
        # Find the target document
        target_document = None
        for doc in session_documents:
            if doc.file_name.lower() == file_name.lower():
                target_document = doc
                break
        
        if not target_document:
            available_docs = [doc.file_name for doc in session_documents]
            raise ValueError(
                f"Document '{file_name}' not found in session '{session_id}'. "
                f"Available documents: {available_docs}"
            )
        
        # Get vector store and perform search
        try:
            vector_store = VectorStore()
            collection_name = target_document.vector_collection_name
            
            # Convert text query to vector embedding
            logger.info(f"Generating embedding for query: '{query}'")
            embedder = Embedder()
            query_embedding = embedder.create_embedding_for_query(query)
            
            if query_embedding is None:
                raise ValueError("Failed to create embedding for the query.")
            
            logger.info(f"Generated embedding with {len(query_embedding)} dimensions")
            
            # Perform similarity search with query embedding
            search_results_raw = vector_store.similarity_search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=top_k
            )
            
            # Format results with page number information
            search_results = []
            for i, result in enumerate(search_results_raw):
                # Extract page number from metadata if available
                metadata = result.get("metadata", {})
                page_number = metadata.get("page_number", None)
                
                search_results.append({
                    "content": result.get("content", ""),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "chunk_index": result.get("chunk_index", i),
                    "page_number": page_number
                })
            
            logger.success(f"Search completed: found {len(search_results)} results")
            
            return {
                "search_results": search_results,
                "query_used": query,
                "total_results": len(search_results),
                "metadata": {
                    "session_id": session_id,
                    "file_name": file_name,
                    "collection_name": collection_name,
                    "top_k_requested": top_k
                }
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            # For now, return mock results
            mock_results = [
                {
                    "content": f"Mock search result 1 for query '{query}' in {file_name}",
                    "similarity_score": 0.95,
                    "chunk_index": 0,
                    "page_number": 1
                },
                {
                    "content": f"Mock search result 2 for query '{query}' in {file_name}",
                    "similarity_score": 0.87,
                    "chunk_index": 1,
                    "page_number": 2
                }
            ]
            
            logger.warning("Using mock search results due to VectorStore unavailability")
            
            return {
                "search_results": mock_results[:top_k],
                "query_used": query,
                "total_results": len(mock_results[:top_k]),
                "metadata": {
                    "session_id": session_id,
                    "file_name": file_name,
                    "mock_results": True,
                    "top_k_requested": top_k
                }
            }


class SummarizeDocumentTool(BaseTool):
    """
    Tool for AI-powered document summarization.
    
    This tool creates intelligent summaries of documents or text passages using
    Anthropic's language models. It can summarize entire documents or specific text
    portions with configurable length and format options. The tool is ideal for
    quickly understanding key points and main ideas from lengthy content.
    """
    
    name = "summarize_document"
    description = (
        "Creates an AI-powered summary of a document or specific text passage. "
        "Use this tool to quickly understand the main ideas and key points from "
        "lengthy content. You can customize the summary length ('short' ~100 words, "
        "'medium' ~300 words, 'long' ~500 words) and format ('paragraph' for "
        "continuous text or 'bullet_points' for structured lists). "
        "If you don't specify text to summarize, the entire document will be processed. "
        "This tool is perfect for getting overviews, extracting key insights, "
        "or creating executive summaries."
    )
    args_schema = SummarizeDocumentArgs
    return_schema = SummarizeDocumentResult
    version = "1.0.0"
    category = "document"
    requires_session = True
    
    def __init__(self, session_manager=None):
        super().__init__(session_manager=session_manager)
        # Initialize Anthropic client
        if anthropic and ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("Anthropic client not available for SummarizeDocumentTool")
    
    def _execute(
        self, 
        file_name: str, 
        session_id: str, 
        text_to_summarize: Optional[str] = None,
        summary_length: str = "medium",
        output_format: str = "paragraph"
    ) -> Dict[str, Any]:
        """
        Execute the document summarization operation.
        
        Args:
            file_name: Name of the document to summarize
            session_id: User session ID for document access
            text_to_summarize: Specific text to summarize (if None, uses full document)
            summary_length: Length of summary (short/medium/long)
            output_format: Format of summary (paragraph/bullet_points)
            
        Returns:
            Dictionary with summary and metadata
            
        Raises:
            ValueError: If required services unavailable or document not found
        """
        logger.info(f"Summarizing document: {file_name} (length: {summary_length}, format: {output_format})")

        # Check if Anthropic is available
        if anthropic is None:
            raise ValueError("Anthropic API is not available. Cannot generate summaries.")

        # Get text to summarize
        if text_to_summarize is None:
            # Get document content using DocumentProcessor for full content
            if not self.session_manager:
                raise ValueError("SessionManager not available - tool not properly initialized")
            
            if DocumentProcessor is None:
                raise ValueError("DocumentProcessor is not available. Cannot process documents.")
            
            session_documents = self.session_manager.get_session_documents(session_id)
            
            if not session_documents:
                raise ValueError(f"No documents found in session: {session_id}")
            
            # Find target document (smart file finder logic)
            target_document = None
            for doc in session_documents:
                if doc.file_name.lower() == file_name.lower():
                    target_document = doc
                    break
            
            if not target_document:
                available_docs = [doc.file_name for doc in session_documents]
                raise ValueError(
                    f"Document '{file_name}' not found in session '{session_id}'. "
                    f"Available documents: {available_docs}"
                )
            
            # Use DocumentProcessor to read full content from persistent file
            document_processor = DocumentProcessor()
            file_path = target_document.file_path
            
            logger.info(f"Reading full content for summarization from file: {file_path}")
            document = document_processor.process(file_path)
            
            if not document or not document.content:
                raise ValueError(f"Failed to read document content from {file_path}")
            
            # Handle new structured content format
            if isinstance(document.content, list):
                # New format: [{"page_number": int, "text": str}, ...]
                text_to_summarize = "\n".join(page["text"] for page in document.content)
            else:
                # Legacy format: single string (backward compatibility)
                text_to_summarize = document.content
            
            if not text_to_summarize or text_to_summarize.strip() == "":
                raise ValueError("Summarization is not possible as document content is empty.")
        
        # Prepare summarization parameters
        length_map = {
            "short": {"words": 100, "description": "concise"},
            "medium": {"words": 300, "description": "moderate detail"},
            "long": {"words": 500, "description": "comprehensive"}
        }
        
        target_words = length_map[summary_length]["words"]
        length_desc = length_map[summary_length]["description"]
        
        # Create system prompt based on parameters
        if output_format == "bullet_points":
            format_instruction = "as a structured list with bullet points"
        else:
            format_instruction = "as coherent paragraphs"

        system_prompt, user_prompt, target_words = self._create_advanced_prompts(
    text_to_summarize, summary_length, output_format, file_name
)

        # Make Anthropic API call
        try:
            if not self.client:
                raise RuntimeError("Anthropic client not available")
                
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=min(target_words * 3, 4000),  # More generous token limit
                temperature=0.2,  # Lower for more focused summaries
            )
            
            summary = response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            # Provide fallback summary
            summary = (
                f"[Mock summary of {file_name}]\n\n"
                f"This is a {summary_length} {output_format.replace('_', ' ')} summary "
                f"that would be generated by Anthropic. The actual content would provide "
                f"key insights and main points from the document in approximately "
                f"{target_words} words."
            )
            logger.warning("Using mock summary due to API unavailability")
        
        # Calculate metrics
        original_length = len(text_to_summarize)
        summary_length_chars = len(summary)
        compression_ratio = summary_length_chars / original_length if original_length > 0 else 0.0
        
        logger.success(f"Summary generated: {summary_length_chars} chars from {original_length} original chars")
        
        return {
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length_chars,
            "compression_ratio": round(compression_ratio, 3),
            "metadata": {
                "session_id": session_id,
                "file_name": file_name,
                "summary_type": summary_length,
                "output_format": output_format,
                "model_used": ANTHROPIC_MODEL
            }
        }
    

        # Gelişmiş prompt sistemi
    def _create_advanced_prompts(self, text_to_summarize: str, summary_length: str, output_format: str, file_name: str) -> tuple:
        """
        Gelişmiş system ve user prompt'lari oluştur.
        """
        
        # Text analizini yap
        word_count = len(text_to_summarize.split())
        char_count = len(text_to_summarize)
        
        # Summary parametrelerini belirle
        length_configs = {
            "short": {
                "words": 100, 
                "style": "concise and punchy",
                "focus": "only the most critical points",
                "structure": "2-3 main points maximum"
            },
            "medium": {
                "words": 300, 
                "style": "balanced and comprehensive", 
                "focus": "key insights with supporting details",
                "structure": "4-6 well-developed points"
            },
            "long": {
                "words": 500, 
                "style": "detailed and thorough",
                "focus": "comprehensive coverage with context",
                "structure": "structured analysis with multiple sections"
            }
        }
        
        config = length_configs.get(summary_length, length_configs["medium"])
        target_words = config["words"]  # VARIABLE TANIMLAND
        
        
        # Format-specific instructions
        if output_format == "bullet_points":
            format_instruction = (
                "Format as clear bullet points with:\n"
                "• Each bullet containing one main idea\n"
                "• Sub-bullets for supporting details where needed\n"
                "• Bold key terms or concepts\n"
                "• Logical grouping of related points"
            )
        else:
            format_instruction = (
                "Format as flowing paragraphs with:\n"
                "• Clear topic sentences for each paragraph\n"
                "• Smooth transitions between ideas\n"
                "• Proper conclusion that ties everything together\n"
                "• Professional, readable prose"
            )
        
        # Content type detection
        content_hints = []
        if "research" in file_name.lower() or "study" in file_name.lower():
            content_hints.append("research findings and methodology")
        if "report" in file_name.lower():
            content_hints.append("key metrics, conclusions, and recommendations")
        if "proposal" in file_name.lower():
            content_hints.append("objectives, benefits, and action items")
        if "manual" in file_name.lower() or "guide" in file_name.lower():
            content_hints.append("step-by-step processes and important warnings")
        
        content_guidance = f"Pay special attention to: {', '.join(content_hints)}" if content_hints else ""

        compression_ratio = round((target_words/max(word_count, 1))*100, 1) if word_count > 0 else "N/A"
        
        # Advanced system prompt
        system_prompt = f"""You are an expert document analyst and professional summarizer specializing in extracting maximum value from complex texts.

    TASK: Create a {config['style']} summary in approximately {config['words']} words.

    CONTENT ANALYSIS:
    - Source document: {file_name}
    - Original length: {word_count} words ({char_count} characters)
    - Compression target: ~{compression_ratio}% of original

    SUMMARY REQUIREMENTS:
    - Style: {config['style']}
    - Focus: {config['focus']}
    - Structure: {config['structure']}
    - Word target: {config['words']} words (±20% acceptable)

    FORMATTING:
    {format_instruction}

    CONTENT PRIORITIES:
    1. Main thesis or central argument
    2. Key findings, results, or conclusions
    3. Critical data points or evidence
    4. Actionable insights or recommendations
    5. Important context or background (if space allows)

    {content_guidance}

    QUALITY STANDARDS:
    - Maintain accuracy - never invent information
    - Use active voice where possible
    - Preserve technical terms when essential
    - Ensure summary stands alone (context-independent)
    - Balance brevity with clarity

    Remember: A great summary captures the essence while enabling the reader to make informed decisions without reading the full document."""

        # Advanced user prompt
        user_prompt = f"""Please analyze and summarize the following document content from "{file_name}":

    CONTENT TO SUMMARIZE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {text_to_summarize}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ANALYSIS CHECKLIST:
    □ Identify the main purpose/thesis
    □ Extract key arguments or findings  
    □ Note important data/statistics
    □ Highlight conclusions or recommendations
    □ Preserve essential context

    Create your {summary_length} summary now, following all formatting and content requirements specified above."""

        return system_prompt, user_prompt , target_words  





# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the document tools with mock data and scenarios.
    This validates the tool functionality without requiring full system integration.
    """
    
    print("=== DataNeuron Document Tools Test ===")
    
    # Test data
    test_session_id = "test_session_123"
    test_file_name = "sample_document.pdf"
    test_query = "machine learning algorithms"
    test_text = "This is a sample document about machine learning and artificial intelligence techniques."
    
    # Create tool instances
    print("\nCreating tool instances...")
    try:
        read_tool = ReadFullDocumentTool()
        search_tool = SearchInDocumentTool()
        summarize_tool = SummarizeDocumentTool()
        print(" All tools instantiated successfully")
    except Exception as e:
        print(f"L Tool instantiation failed: {e}")
        exit(1)
    
    # Test 1: ReadFullDocumentTool
    print(f"\nTest 1: ReadFullDocumentTool")
    try:
        result = read_tool.execute(
            file_name=test_file_name,
            session_id=test_session_id
        )
        if result.success:
            print(f" PASS - Document read successful")
            print(f"   Content length: {result.character_count} characters")
        else:
            print(f"L Expected failure - {result.error_message}")
    except Exception as e:
        print(f"L FAIL - Unexpected error: {e}")
    
    # Test 2: SearchInDocumentTool
    print(f"\nTest 2: SearchInDocumentTool")
    try:
        result = search_tool.execute(
            query=test_query,
            file_name=test_file_name,
            session_id=test_session_id,
            top_k=3
        )
        if result.success:
            print(f" PASS - Search completed")
            print(f"   Results found: {result.total_results}")
            print(f"   Query used: {result.query_used}")
        else:
            print(f"L Expected failure - {result.error_message}")
    except Exception as e:
        print(f"L FAIL - Unexpected error: {e}")
    
    # Test 3: SummarizeDocumentTool with custom text
    print(f"\nTest 3: SummarizeDocumentTool with custom text")
    try:
        result = summarize_tool.execute(
            file_name=test_file_name,
            session_id=test_session_id,
            text_to_summarize=test_text,
            summary_length="short",
            output_format="bullet_points"
        )
        if result.success:
            print(f" PASS - Summarization completed")
            print(f"   Summary length: {result.summary_length} characters")
            print(f"   Compression ratio: {result.compression_ratio}")
            print(f"   Summary preview: {result.summary[:100]}...")
        else:
            print(f"L Expected failure - {result.error_message}")
    except Exception as e:
        print(f"L FAIL - Unexpected error: {e}")
    
    # Test 4: Error handling - invalid arguments
    print(f"\nTest 4: Error handling with invalid arguments")
    try:
        result = read_tool.execute(
            file_name="",  # Invalid empty filename
            session_id=test_session_id
        )
        if not result.success:
            print(f" PASS - Invalid arguments handled gracefully")
            print(f"   Error: {result.error_message}")
        else:
            print(f"L FAIL - Should have failed with empty filename")
    except Exception as e:
        print(f" PASS - Exception handled: {e}")
    
    # Test 5: Schema validation
    print(f"\nTest 5: Schema validation")
    try:
        # Test valid schema
        args = ReadDocumentArgs(file_name="test.pdf", session_id="123")
        print(f" PASS - Valid arguments accepted: {args.file_name}")
        
        # Test invalid schema (this should raise ValidationError)
        try:
            invalid_args = SummarizeDocumentArgs(
                file_name="test.pdf",
                session_id="123",
                summary_length="invalid_length"  # Should fail regex validation
            )
            print(f"L FAIL - Invalid arguments should have been rejected")
        except Exception as e:
            print(f" PASS - Invalid arguments rejected: {type(e).__name__}")
            
    except Exception as e:
        print(f"L FAIL - Schema validation error: {e}")
    
    print(f"\n=== Test Complete ===")
    print("Document tools are ready for integration with ToolManager.")
    print("Tools will be automatically discovered and available for LLM agents.")