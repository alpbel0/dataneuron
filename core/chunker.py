"""
DataNeuron Intelligent Text Chunking Module
===========================================

This module provides sophisticated text chunking capabilities for optimal LLM and vector
database performance. It preserves semantic integrity while splitting large documents
into properly sized, overlapping chunks optimized for embedding and retrieval.

Features:
- Token-aware chunking using tiktoken for accurate LLM cost/performance optimization
- Semantic boundary preservation (paragraphs, sentences, words)
- Configurable chunk size and overlap for optimal retrieval performance
- Rich metadata preservation linking chunks back to source documents
- RecursiveCharacterTextSplitter integration for intelligent text splitting
- Comprehensive chunk indexing and statistics
- Page number tracking for source highlighting

Usage:
    from core.chunker import TextChunker, Chunk
    from core.document_processor import Document
    
    # Create chunker with default settings
    chunker = TextChunker()
    
    # Process a document into chunks
    document = Document(content=[{"page_number": 1, "text": "..."}], metadata={...})
    chunks = chunker.create_chunks(document)
    
    # Each chunk contains content and rich metadata
    for chunk in chunks:
        print(f"Chunk {chunk.metadata['chunk_index']}: {chunk.token_count} tokens")

Architecture:
- Chunk dataclass for structured chunk representation
- TextChunker class for intelligent text splitting
- Token-based length calculation using tiktoken
- Metadata preservation and augmentation
- Page number tracking for source highlighting
- Configurable parameters via settings
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Callable
import tiktoken

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP, 
    TIKTOKEN_ENCODING_MODEL,
    TEXT_SEPARATORS,
    MAX_CHUNK_SIZE_CHARS
)

# Import dependencies with error handling
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document as LangChainDocument
except ImportError as e:
    logger.warning(f"LangChain text splitters not available: {e}")
    RecursiveCharacterTextSplitter = None
    LangChainDocument = None

try:
    from core.document_processor import Document
except ImportError as e:
    logger.warning(f"Document processor not available: {e}")
    # Create a minimal Document class for testing - UPDATED FORMAT
    @dataclass
    class Document:
        content: List[Dict[str, Any]]
        metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CHUNK DATA STRUCTURE
# ============================================================================

@dataclass
class Chunk:
    """
    Represents a single chunk of text with associated metadata.
    
    This dataclass encapsulates a portion of a larger document along with
    comprehensive metadata that maintains the connection to the source document
    and provides chunk-specific information for retrieval and analysis.
    
    Attributes:
        content (str): The actual text content of the chunk
        metadata (Dict[str, Any]): Comprehensive metadata including:
            - Source document information (file_name, file_hash, etc.)
            - Chunk-specific data (chunk_index, token_count, etc.)
            - Processing information (chunking_timestamp, etc.)
            - Page number information (page_number)
        token_count (int): Number of tokens in this chunk (calculated via tiktoken)
        character_count (int): Number of characters in this chunk
        chunk_index (int): Position of this chunk within the document (0-based)
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    character_count: int = 0
    chunk_index: int = 0
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.character_count = len(self.content)
        
        # Update metadata with chunk-specific information
        self.metadata.update({
            'chunk_index': self.chunk_index,
            'token_count': self.token_count,
            'character_count': self.character_count,
            'content_preview': self.content[:100] + "..." if len(self.content) > 100 else self.content
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this chunk for logging/debugging."""
        return {
            'chunk_index': self.chunk_index,
            'content_length': len(self.content),
            'token_count': self.token_count,
            'character_count': self.character_count,
            'content_start': self.content[:50] + "..." if len(self.content) > 50 else self.content,
            'source_file': self.metadata.get('file_name', 'unknown'),
            'page_number': self.metadata.get('page_number', 'unknown')
        }


# ============================================================================
# TEXT CHUNKER CLASS
# ============================================================================

class TextChunker:
    """
    Intelligent text chunking system optimized for LLM and vector database performance.
    
    This class provides sophisticated text splitting capabilities that preserve semantic
    boundaries while ensuring optimal chunk sizes for embedding models and retrieval.
    It uses token-based calculations for accurate cost and performance optimization.
    
    Key Features:
    - Token-aware chunking using tiktoken encoding
    - Semantic boundary preservation (paragraphs -> sentences -> words)
    - Configurable overlap for context preservation
    - Rich metadata preservation and augmentation
    - Page number tracking for source highlighting
    - Comprehensive logging and statistics
    
    Parameters are loaded from config/settings.py:
    - CHUNK_SIZE: Target chunk size in tokens
    - CHUNK_OVERLAP: Overlap between chunks in tokens  
    - TIKTOKEN_ENCODING_MODEL: Encoding model for token calculation
    - TEXT_SEPARATORS: Hierarchical separators for semantic splitting
    """
    
    def __init__(self):
        """
        Initialize the TextChunker with configuration from settings.
        
        Sets up the tiktoken encoder and RecursiveCharacterTextSplitter
        with token-based length calculation for optimal performance.
        """
        # Load configuration
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.encoding_model = TIKTOKEN_ENCODING_MODEL
        self.separators = TEXT_SEPARATORS
        self.max_chars = MAX_CHUNK_SIZE_CHARS
        
        # Initialize tiktoken encoder
        try:
            self.tokenizer = tiktoken.get_encoding(self.encoding_model)
            logger.info(f"Initialized tiktoken with encoding: {self.encoding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}")
            self.tokenizer = None
        
        # Create token counting function
        self.token_counter = self._create_token_counter()
        
        # Initialize text splitter with token-based length function
        if RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.token_counter,
                separators=self.separators,
                keep_separator=True,  # Preserve separators for better context
                is_separator_regex=False
            )
            logger.info(f"Initialized RecursiveCharacterTextSplitter:")
            logger.info(f"  - Chunk size: {self.chunk_size} tokens")
            logger.info(f"  - Overlap: {self.chunk_overlap} tokens")
            logger.info(f"  - Separators: {self.separators}")
        else:
            self.text_splitter = None
            logger.warning("RecursiveCharacterTextSplitter not available - using fallback")
        
        # Statistics tracking
        self.total_documents_processed = 0
        self.total_chunks_created = 0
        self.total_tokens_processed = 0
        
        logger.success("TextChunker initialized successfully")
    
    def _create_token_counter(self) -> Callable[[str], int]:
        """
        Create a token counting function for use with text splitter.
        
        Returns:
            Function that takes text string and returns token count
        """
        if self.tokenizer is not None:
            def count_tokens(text: str) -> int:
                """Count tokens in text using tiktoken."""
                try:
                    return len(self.tokenizer.encode(text))
                except Exception as e:
                    logger.warning(f"Token counting failed, using character fallback: {e}")
                    return len(text) // 4  # Rough approximation: ~4 chars per token
            
            return count_tokens
        else:
            # Fallback to character-based approximation
            logger.warning("Using character-based token approximation")
            return lambda text: len(text) // 4
    
    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split a document into optimally sized chunks with preserved metadata and page tracking.
        
        This is the main method that processes a Document object and returns
        a list of Chunk objects. Each chunk maintains semantic integrity while
        staying within the configured token limits and preserves page number information.
        
        Args:
            document (Document): Document object with structured content and metadata
            
        Returns:
            List[Chunk]: List of chunks with preserved and augmented metadata
            
        Raises:
            ValueError: If document is invalid or chunking fails
        """
        if not document or not document.content:
            raise ValueError("Document must have content to chunk")
        
        if not isinstance(document.content, list):
            raise ValueError("Document.content must be a list of page dictionaries")
        
        logger.info(f"Starting chunking process for document")
        logger.info(f"  - Content pages: {len(document.content)}")
        
        # Calculate total content statistics
        total_text_length = sum(len(page.get("text", "")) for page in document.content)
        combined_text = "\n".join(page.get("text", "") for page in document.content)
        logger.info(f"  - Total content length: {total_text_length} characters")
        logger.info(f"  - Estimated tokens: {self.token_counter(combined_text)}")
        
        try:
            # Create LangChain Document objects for each page
            langchain_docs = []
            for page_data in document.content:
                page_number = page_data.get("page_number", 1)
                page_text = page_data.get("text", "")
                
                if not page_text.strip():
                    continue  # Skip empty pages
                
                # Create metadata for this page
                page_metadata = dict(document.metadata) if document.metadata else {}
                page_metadata.update({
                    "page_number": page_number,
                    "page_text_length": len(page_text),
                    "source_page": True
                })
                
                # Create LangChain Document
                if LangChainDocument is not None:
                    langchain_doc = LangChainDocument(
                        page_content=page_text,
                        metadata=page_metadata
                    )
                    langchain_docs.append(langchain_doc)
                    logger.debug(f"Created LangChain document for page {page_number}: {len(page_text)} chars")
            
            if not langchain_docs:
                raise ValueError("No valid pages found in document content")
            
            # Split documents using LangChain
            if self.text_splitter is not None and LangChainDocument is not None:
                split_docs = self.text_splitter.split_documents(langchain_docs)
                logger.info(f"LangChain split into {len(split_docs)} chunks")
            else:
                # Fallback: use simple splitting
                split_docs = self._fallback_split_documents(langchain_docs)
                logger.info(f"Fallback split into {len(split_docs)} chunks")
            
            # Convert LangChain Documents to our Chunk objects
            chunks = []
            for i, split_doc in enumerate(split_docs):
                chunk_text = split_doc.page_content
                chunk_metadata = dict(split_doc.metadata) if split_doc.metadata else {}
                
                # Calculate token count for this chunk
                token_count = self.token_counter(chunk_text)
                
                # Create comprehensive metadata
                final_metadata = self._create_chunk_metadata(
                    document, i, token_count, len(split_docs), chunk_metadata
                )
                
                # Create Chunk object
                chunk = Chunk(
                    content=chunk_text.strip(),
                    metadata=final_metadata,
                    token_count=token_count,
                    character_count=len(chunk_text.strip()),
                    chunk_index=i
                )
                
                chunks.append(chunk)
                
                page_info = f", page: {final_metadata.get('page_number', 'unknown')}"
                logger.debug(f"Created chunk {i}: {token_count} tokens, {len(chunk_text)} chars{page_info}")
            
            # Update statistics
            self.total_documents_processed += 1
            self.total_chunks_created += len(chunks)
            self.total_tokens_processed += sum(chunk.token_count for chunk in chunks)
            
            # Log summary
            logger.success(f"Chunking completed successfully:")
            logger.info(f"  - Created {len(chunks)} chunks")
            logger.info(f"  - Total tokens: {sum(chunk.token_count for chunk in chunks)}")
            
            if len(chunks) > 0:
                logger.info(f"  - Avg tokens per chunk: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f}")
                logger.info(f"  - Token utilization: {(sum(chunk.token_count for chunk in chunks) / (len(chunks) * self.chunk_size)) * 100:.1f}%")
                
                # Show page distribution
                page_numbers = [c.metadata.get('page_number', 'unknown') for c in chunks]
                unique_pages = set(str(p) for p in page_numbers if p != 'unknown')
                logger.info(f"  - Pages covered: {len(unique_pages)} unique pages")
            else:
                logger.warning("  - No chunks created - document may be empty")
            
            return chunks
            
        except Exception as e:
            logger.exception(f"Chunking failed: {str(e)}")
            raise ValueError(f"Failed to chunk document: {str(e)}")
    
    def _fallback_split_documents(self, langchain_docs: List) -> List:
        """
        Fallback document splitting when LangChain is not available.
        
        Args:
            langchain_docs: List of LangChain Document objects
            
        Returns:
            List of split document objects
        """
        logger.warning("Using fallback document splitting method")
        
        split_docs = []
        
        for doc in langchain_docs:
            text = doc.page_content
            metadata = doc.metadata
            
            # Simple character-based splitting
            text_length = len(text)
            target_chars = self.chunk_size * 4  # ~4 chars per token
            overlap_chars = self.chunk_overlap * 4
            
            current_pos = 0
            chunk_index = 0
            
            while current_pos < text_length:
                # Calculate chunk end position
                chunk_end = min(current_pos + target_chars, text_length)
                
                # Try to find a good break point
                if chunk_end < text_length:
                    for separator in ["\n\n", "\n", ". ", " "]:
                        sep_pos = text.rfind(separator, current_pos, chunk_end)
                        if sep_pos > current_pos:
                            chunk_end = sep_pos + len(separator)
                            break
                
                # Extract chunk
                chunk_text = text[current_pos:chunk_end]
                if chunk_text.strip():
                    # Create a simple document-like object
                    chunk_metadata = dict(metadata)
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'fallback_split': True
                    })
                    
                    split_doc = type('Document', (), {
                        'page_content': chunk_text,
                        'metadata': chunk_metadata
                    })()
                    
                    split_docs.append(split_doc)
                    chunk_index += 1
                
                # Move to next position with overlap
                current_pos = chunk_end - overlap_chars
                if current_pos >= chunk_end:  # Prevent infinite loop
                    current_pos = chunk_end
        
        return split_docs
    
    def _create_chunk_metadata(
        self, 
        document: Document, 
        chunk_index: int, 
        token_count: int,
        total_chunks: int,
        langchain_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for a chunk.
        
        Args:
            document: Source document
            chunk_index: Index of this chunk
            token_count: Number of tokens in chunk
            total_chunks: Total number of chunks in document
            langchain_metadata: Metadata from LangChain processing
            
        Returns:
            Dictionary with comprehensive chunk metadata
        """
        # Start with original document metadata
        chunk_metadata = dict(document.metadata) if document.metadata else {}
        
        # Add LangChain metadata (includes page_number)
        if langchain_metadata:
            chunk_metadata.update(langchain_metadata)
        
        # Add chunk-specific metadata
        chunk_metadata.update({
            # Chunk identification
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_id': f"{chunk_metadata.get('file_hash', 'unknown')}_{chunk_index}",
            
            # Size information
            'token_count': token_count,
            'chunk_size_target': self.chunk_size,
            'chunk_overlap_target': self.chunk_overlap,
            
            # Processing information
            'chunking_method': 'LangChain_split_documents' if self.text_splitter and LangChainDocument else 'fallback',
            'tokenizer_model': self.encoding_model,
            'separators_used': self.separators,
            
            # Position information
            'is_first_chunk': chunk_index == 0,
            'is_last_chunk': chunk_index == total_chunks - 1,
            'relative_position': chunk_index / max(total_chunks - 1, 1),  # 0.0 to 1.0
            
            # Quality metrics
            'token_efficiency': token_count / self.chunk_size,  # How well we utilized target size
        })
        
        return chunk_metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about chunking performance.
        
        Returns:
            Dictionary with statistics and metrics
        """
        avg_chunks_per_doc = (
            self.total_chunks_created / self.total_documents_processed 
            if self.total_documents_processed > 0 else 0
        )
        
        avg_tokens_per_chunk = (
            self.total_tokens_processed / self.total_chunks_created
            if self.total_chunks_created > 0 else 0
        )
        
        return {
            'total_documents_processed': self.total_documents_processed,
            'total_chunks_created': self.total_chunks_created,
            'total_tokens_processed': self.total_tokens_processed,
            'average_chunks_per_document': round(avg_chunks_per_doc, 2),
            'average_tokens_per_chunk': round(avg_tokens_per_chunk, 1),
            'target_chunk_size': self.chunk_size,
            'target_overlap': self.chunk_overlap,
            'encoding_model': self.encoding_model,
            'splitter_available': self.text_splitter is not None,
            'tokenizer_available': self.tokenizer is not None,
            'langchain_available': LangChainDocument is not None
        }
    
    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Validate a list of chunks for quality and consistency.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        if not chunks:
            return {'valid': False, 'error': 'No chunks provided'}
        
        validation_results = {
            'valid': True,
            'total_chunks': len(chunks),
            'issues': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Check chunk sizes
        oversized_chunks = [c for c in chunks if c.token_count > self.chunk_size * 1.2]
        undersized_chunks = [c for c in chunks if c.token_count < self.chunk_size * 0.1]
        
        if oversized_chunks:
            validation_results['issues'].append(f"{len(oversized_chunks)} chunks exceed target size by >20%")
        
        if len(undersized_chunks) > 1:  # Last chunk can be small
            validation_results['issues'].append(f"{len(undersized_chunks)} chunks are very small (<10% of target)")
        
        # Check metadata consistency
        missing_metadata = [c for c in chunks if not c.metadata.get('chunk_id')]
        if missing_metadata:
            validation_results['issues'].append(f"{len(missing_metadata)} chunks missing chunk_id")
        
        # Check page number consistency
        chunks_with_pages = [c for c in chunks if 'page_number' in c.metadata]
        if len(chunks_with_pages) < len(chunks):
            validation_results['issues'].append(f"{len(chunks) - len(chunks_with_pages)} chunks missing page numbers")
        
        # Calculate statistics
        token_counts = [c.token_count for c in chunks]
        validation_results['statistics'] = {
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'total_tokens': sum(token_counts),
            'size_variance': max(token_counts) - min(token_counts),
            'chunks_with_page_numbers': len(chunks_with_pages)
        }
        
        # Generate recommendations
        if validation_results['statistics']['size_variance'] > self.chunk_size:
            validation_results['recommendations'].append("Consider adjusting chunk size for more consistent results")
        
        if validation_results['statistics']['avg_tokens'] < self.chunk_size * 0.8:
            validation_results['recommendations'].append("Average chunk size is low - consider reducing target size")
        
        if len(chunks_with_pages) < len(chunks):
            validation_results['recommendations'].append("Some chunks missing page numbers - check document processing")
        
        validation_results['valid'] = len(validation_results['issues']) == 0
        
        return validation_results


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the TextChunker with the new Document format.
    This comprehensive test ensures the chunking system works correctly.
    """
    
    print("=== DataNeuron TextChunker Test (Updated Format) ===")
    
    # Create test document with new structured format
    test_document = Document(
        content=[
            {
                "page_number": 1,
                "text": """Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
It encompasses a wide range of computational techniques designed to mimic human intelligence and decision-making processes.

Machine Learning, a subset of AI, enables systems to automatically learn and improve from experience without being 
explicitly programmed. This approach has revolutionized fields ranging from healthcare to finance, transportation to entertainment."""
            },
            {
                "page_number": 2,
                "text": """Deep Learning, which uses artificial neural networks inspired by the human brain, has achieved remarkable breakthroughs 
in computer vision, natural language processing, and speech recognition. These advances have made possible applications 
such as image classification, machine translation, and voice assistants.

Natural Language Processing (NLP) focuses on the interaction between computers and humans through natural language. 
It involves the development of algorithms that can understand, interpret, and generate human language in a valuable way."""
            },
            {
                "page_number": 3,
                "text": """Computer Vision enables machines to interpret and understand visual information from the world. Through the use of 
digital images and videos, AI systems can identify objects, recognize faces, and even understand complex scenes.

Reinforcement Learning is an area of machine learning concerned with how intelligent agents ought to take actions 
in an environment to maximize cumulative reward. This approach has been successfully applied to game playing, 
robotics, and autonomous vehicle navigation."""
            }
        ],
        metadata={
            'file_name': 'ai_overview.pdf',
            'file_hash': 'test_hash_12345',
            'file_type': '.pdf',
            'file_size_mb': 0.02,
            'creation_date': '2024-01-01T12:00:00',
            'page_count': 3,
            'source': 'test_document'
        }
    )
    
    # Test 1: Create TextChunker instance
    print(f"\nTest 1: Creating TextChunker instance")
    try:
        chunker = TextChunker()
        print(f"✓ PASS - TextChunker created successfully")
        print(f"   Chunk size: {chunker.chunk_size} tokens")
        print(f"   Overlap: {chunker.chunk_overlap} tokens")
        print(f"   Encoding: {chunker.encoding_model}")
        print(f"   Tokenizer available: {chunker.tokenizer is not None}")
        print(f"   Text splitter available: {chunker.text_splitter is not None}")
        print(f"   LangChain available: {LangChainDocument is not None}")
    except Exception as e:
        print(f"✗ FAIL - TextChunker creation failed: {e}")
        exit(1)
    
    # Test 2: Create chunks from new document format
    print(f"\nTest 2: Creating chunks from structured document")
    try:
        chunks = chunker.create_chunks(test_document)
        print(f"✓ PASS - Document chunked successfully")
        print(f"   Created {len(chunks)} chunks")
        print(f"   Total tokens: {sum(chunk.token_count for chunk in chunks)}")
        
        if chunks:
            print(f"   First chunk: {chunks[0].token_count} tokens, page: {chunks[0].metadata.get('page_number', 'unknown')}")
            print(f"   Last chunk: {chunks[-1].token_count} tokens, page: {chunks[-1].metadata.get('page_number', 'unknown')}")
            print(f"   Average chunk size: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f} tokens")
            
            # Show page distribution
            page_numbers = [c.metadata.get('page_number', 'unknown') for c in chunks]
            unique_pages = set(str(p) for p in page_numbers if p != 'unknown')
            print(f"   Pages covered: {sorted(unique_pages)}")
        
    except Exception as e:
        print(f"✗ FAIL - Chunking failed: {e}")
        chunks = []
    
    # Test 3: Validate chunk metadata
    print(f"\nTest 3: Validating chunk metadata")
    if chunks:
        first_chunk = chunks[0]
        required_metadata = ['chunk_index', 'total_chunks', 'token_count', 'file_name', 'page_number']
        missing_metadata = [key for key in required_metadata if key not in first_chunk.metadata]
        
        if not missing_metadata:
            print(f"✓ PASS - All required metadata present")
            print(f"   Chunk ID: {first_chunk.metadata.get('chunk_id', 'N/A')}")
            print(f"   Source file: {first_chunk.metadata.get('file_name', 'N/A')}")
            print(f"   Page number: {first_chunk.metadata.get('page_number', 'N/A')}")
            print(f"   Is first chunk: {first_chunk.metadata.get('is_first_chunk', 'N/A')}")
            print(f"   Token efficiency: {first_chunk.metadata.get('token_efficiency', 'N/A'):.2f}")
        else:
            print(f"✗ FAIL - Missing metadata: {missing_metadata}")
    else:
        print(f"⚠ SKIP - No chunks to validate")
    
    # Test 4: Validate chunks
    print(f"\nTest 4: Running chunk validation")
    if chunks:
        validation_result = chunker.validate_chunks(chunks)
        
        if validation_result['valid']:
            print(f"✓ PASS - All chunks are valid")
        else:
            print(f"⚠ WARNING - Validation issues found: {len(validation_result['issues'])}")
            for issue in validation_result['issues']:
                print(f"     - {issue}")
        
        stats = validation_result['statistics']
        print(f"   Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"   Average tokens: {stats['avg_tokens']:.1f}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Chunks with page numbers: {stats['chunks_with_page_numbers']}")
        
        if validation_result['recommendations']:
            print(f"   Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"     - {rec}")
    else:
        print(f"⚠ SKIP - No chunks to validate")
    
    # Test 5: Edge case - empty document
    print(f"\nTest 5: Testing edge cases")
    
    try:
        empty_doc = Document(content=[], metadata={'test': 'empty'})
        chunker.create_chunks(empty_doc)
        print(f"✗ FAIL - Empty document should raise ValueError")
    except ValueError:
        print(f"✓ PASS - Empty document correctly rejected")
    except Exception as e:
        print(f"✗ FAIL - Unexpected error with empty document: {e}")
    
    print(f"\n=== Test Complete ===")
    print("TextChunker is ready for integration with the updated document processing pipeline.")
    print("Chunks now preserve page number information for source highlighting.")