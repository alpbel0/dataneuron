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

Usage:
    from core.chunker import TextChunker, Chunk
    from core.document_processor import Document
    
    # Create chunker with default settings
    chunker = TextChunker()
    
    # Process a document into chunks
    document = Document(content="Large text...", metadata={...})
    chunks = chunker.create_chunks(document)
    
    # Each chunk contains content and rich metadata
    for chunk in chunks:
        print(f"Chunk {chunk.metadata['chunk_index']}: {chunk.token_count} tokens")

Architecture:
- Chunk dataclass for structured chunk representation
- TextChunker class for intelligent text splitting
- Token-based length calculation using tiktoken
- Metadata preservation and augmentation
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
except ImportError as e:
    logger.warning(f"LangChain text splitters not available: {e}")
    RecursiveCharacterTextSplitter = None

try:
    from core.document_processor import Document
except ImportError as e:
    logger.warning(f"Document processor not available: {e}")
    # Create a minimal Document class for testing
    @dataclass
    class Document:
        content: str
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
            'source_file': self.metadata.get('file_name', 'unknown')
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
        Split a document into optimally sized chunks with preserved metadata.
        
        This is the main method that processes a Document object and returns
        a list of Chunk objects. Each chunk maintains semantic integrity while
        staying within the configured token limits.
        
        Args:
            document (Document): Document object with content and metadata
            
        Returns:
            List[Chunk]: List of chunks with preserved and augmented metadata
            
        Raises:
            ValueError: If document is invalid or chunking fails
        """
        if not document or not document.content:
            raise ValueError("Document must have content to chunk")
        
        logger.info(f"Starting chunking process for document")
        logger.info(f"  - Content length: {len(document.content)} characters")
        logger.info(f"  - Estimated tokens: {self.token_counter(document.content)}")
        
        try:
            # Split text into chunks
            if self.text_splitter is not None:
                text_chunks = self.text_splitter.split_text(document.content)
            else:
                # Fallback chunking method
                text_chunks = self._fallback_split_text(document.content)
            
            logger.info(f"Text split into {len(text_chunks)} raw chunks")
            
            # Create Chunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Calculate token count for this chunk
                token_count = self.token_counter(chunk_text)
                
                # Create comprehensive metadata
                chunk_metadata = self._create_chunk_metadata(document, i, token_count, len(text_chunks))
                
                # Create Chunk object
                chunk = Chunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    token_count=token_count,
                    character_count=len(chunk_text.strip()),
                    chunk_index=i
                )
                
                chunks.append(chunk)
                logger.debug(f"Created chunk {i}: {token_count} tokens, {len(chunk_text)} chars")
            
            # Update statistics
            self.total_documents_processed += 1
            self.total_chunks_created += len(chunks)
            self.total_tokens_processed += sum(chunk.token_count for chunk in chunks)
            
            # Log summary
            logger.success(f"Chunking completed successfully:")
            logger.info(f"  - Created {len(chunks)} chunks")
            logger.info(f"  - Total tokens: {sum(chunk.token_count for chunk in chunks)}")
            
            # Avoid division by zero
            if len(chunks) > 0:
                logger.info(f"  - Avg tokens per chunk: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f}")
                logger.info(f"  - Token utilization: {(sum(chunk.token_count for chunk in chunks) / (len(chunks) * self.chunk_size)) * 100:.1f}%")
            else:
                logger.warning("  - No chunks created - document may be empty or contain only whitespace")
            
            return chunks
            
        except Exception as e:
            logger.exception(f"Chunking failed: {str(e)}")
            raise ValueError(f"Failed to chunk document: {str(e)}")
    
    def _create_chunk_metadata(
        self, 
        document: Document, 
        chunk_index: int, 
        token_count: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for a chunk.
        
        Args:
            document: Source document
            chunk_index: Index of this chunk
            token_count: Number of tokens in chunk
            total_chunks: Total number of chunks in document
            
        Returns:
            Dictionary with comprehensive chunk metadata
        """
        # Start with original document metadata
        chunk_metadata = dict(document.metadata) if document.metadata else {}
        
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
            'chunking_method': 'RecursiveCharacterTextSplitter' if self.text_splitter else 'fallback',
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
    
    def _fallback_split_text(self, text: str) -> List[str]:
        """
        Fallback text splitting when LangChain is not available.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        logger.warning("Using fallback text splitting method")
        
        chunks = []
        text_length = len(text)
        current_pos = 0
        
        # Use character-based approximation since we don't have token counter
        target_chars = self.chunk_size * 4  # ~4 chars per token
        overlap_chars = self.chunk_overlap * 4
        
        while current_pos < text_length:
            # Calculate chunk end position
            chunk_end = min(current_pos + target_chars, text_length)
            
            # Try to find a good break point (prefer paragraph or sentence breaks)
            if chunk_end < text_length:
                for separator in ["\n\n", "\n", ". ", " "]:
                    sep_pos = text.rfind(separator, current_pos, chunk_end)
                    if sep_pos > current_pos:
                        chunk_end = sep_pos + len(separator)
                        break
            
            # Extract chunk
            chunk = text[current_pos:chunk_end]
            if chunk.strip():
                chunks.append(chunk)
            
            # Move to next position with overlap
            current_pos = chunk_end - overlap_chars
            if current_pos >= chunk_end:  # Prevent infinite loop
                current_pos = chunk_end
        
        return chunks
    
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
            'tokenizer_available': self.tokenizer is not None
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
        
        # Calculate statistics
        token_counts = [c.token_count for c in chunks]
        validation_results['statistics'] = {
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'total_tokens': sum(token_counts),
            'size_variance': max(token_counts) - min(token_counts)
        }
        
        # Generate recommendations
        if validation_results['statistics']['size_variance'] > self.chunk_size:
            validation_results['recommendations'].append("Consider adjusting chunk size for more consistent results")
        
        if validation_results['statistics']['avg_tokens'] < self.chunk_size * 0.8:
            validation_results['recommendations'].append("Average chunk size is low - consider reducing target size")
        
        validation_results['valid'] = len(validation_results['issues']) == 0
        
        return validation_results


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the TextChunker with various scenarios and validate functionality.
    This comprehensive test ensures the chunking system works correctly.
    """
    
    print("=== DataNeuron TextChunker Test ===")
    
    # Create test document with long text
    long_text = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
    It encompasses a wide range of computational techniques designed to mimic human intelligence and decision-making processes.
    
    Machine Learning, a subset of AI, enables systems to automatically learn and improve from experience without being 
    explicitly programmed. This approach has revolutionized fields ranging from healthcare to finance, transportation to entertainment.
    
    Deep Learning, which uses artificial neural networks inspired by the human brain, has achieved remarkable breakthroughs 
    in computer vision, natural language processing, and speech recognition. These advances have made possible applications 
    such as image classification, machine translation, and voice assistants.
    
    Natural Language Processing (NLP) focuses on the interaction between computers and humans through natural language. 
    It involves the development of algorithms that can understand, interpret, and generate human language in a valuable way.
    
    Computer Vision enables machines to interpret and understand visual information from the world. Through the use of 
    digital images and videos, AI systems can identify objects, recognize faces, and even understand complex scenes.
    
    Reinforcement Learning is an area of machine learning concerned with how intelligent agents ought to take actions 
    in an environment to maximize cumulative reward. This approach has been successfully applied to game playing, 
    robotics, and autonomous vehicle navigation.
    
    The ethical implications of AI development are becoming increasingly important as these systems become more powerful 
    and pervasive. Issues such as algorithmic bias, privacy concerns, and the impact on employment require careful 
    consideration and proactive management.
    
    Looking forward, AI is expected to continue advancing rapidly, with potential breakthroughs in areas such as 
    artificial general intelligence, quantum machine learning, and brain-computer interfaces. These developments 
    will likely reshape society in profound ways, creating new opportunities while also presenting new challenges.
    
    The integration of AI into various industries has already begun to transform business processes, improve efficiency, 
    and create new value propositions. From predictive maintenance in manufacturing to personalized medicine in healthcare, 
    AI applications are becoming increasingly sophisticated and impactful.
    
    Education and training in AI and related fields have become crucial for preparing the workforce for an AI-driven future. 
    Universities and organizations worldwide are developing new curricula and programs to meet the growing demand for 
    AI expertise and ensure responsible development and deployment of these powerful technologies.
    """
    
    # Create mock document
    test_document = Document(
        content=long_text.strip(),
        metadata={
            'file_name': 'ai_overview.txt',
            'file_hash': 'test_hash_12345',
            'file_type': '.txt',
            'file_size_mb': 0.01,
            'creation_date': '2024-01-01T12:00:00',
            'page_count': 1,
            'source': 'test_document'
        }
    )
    
    # Test 1: Create TextChunker instance
    print(f"\nTest 1: Creating TextChunker instance")
    try:
        chunker = TextChunker()
        print(f" PASS - TextChunker created successfully")
        print(f"   Chunk size: {chunker.chunk_size} tokens")
        print(f"   Overlap: {chunker.chunk_overlap} tokens")
        print(f"   Encoding: {chunker.encoding_model}")
        print(f"   Tokenizer available: {chunker.tokenizer is not None}")
        print(f"   Text splitter available: {chunker.text_splitter is not None}")
    except Exception as e:
        print(f"L FAIL - TextChunker creation failed: {e}")
        exit(1)
    
    # Test 2: Create chunks from document
    print(f"\nTest 2: Creating chunks from document")
    try:
        chunks = chunker.create_chunks(test_document)
        print(f" PASS - Document chunked successfully")
        print(f"   Created {len(chunks)} chunks")
        print(f"   Total tokens: {sum(chunk.token_count for chunk in chunks)}")
        
        if chunks:
            print(f"   First chunk: {chunks[0].token_count} tokens")
            print(f"   Last chunk: {chunks[-1].token_count} tokens")
            print(f"   Average chunk size: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f} tokens")
        
    except Exception as e:
        print(f"L FAIL - Chunking failed: {e}")
        chunks = []
    
    # Test 3: Validate chunk metadata
    print(f"\nTest 3: Validating chunk metadata")
    if chunks:
        first_chunk = chunks[0]
        required_metadata = ['chunk_index', 'total_chunks', 'token_count', 'file_name']
        missing_metadata = [key for key in required_metadata if key not in first_chunk.metadata]
        
        if not missing_metadata:
            print(f" PASS - All required metadata present")
            print(f"   Chunk ID: {first_chunk.metadata.get('chunk_id', 'N/A')}")
            print(f"   Source file: {first_chunk.metadata.get('file_name', 'N/A')}")
            print(f"   Is first chunk: {first_chunk.metadata.get('is_first_chunk', 'N/A')}")
            print(f"   Token efficiency: {first_chunk.metadata.get('token_efficiency', 'N/A'):.2f}")
        else:
            print(f"L FAIL - Missing metadata: {missing_metadata}")
    else:
        print(f"L SKIP - No chunks to validate")
    
    # Test 4: Check chunk overlap
    print(f"\nTest 4: Checking chunk overlap")
    if len(chunks) > 1:
        # Check if there's overlap between consecutive chunks
        first_chunk_end = chunks[0].content[-100:]  # Last 100 chars of first chunk
        second_chunk_start = chunks[1].content[:100]  # First 100 chars of second chunk
        
        # Simple overlap detection (this is approximate)
        overlap_detected = any(word in second_chunk_start for word in first_chunk_end.split()[-10:])
        
        if overlap_detected:
            print(f" PASS - Overlap detected between chunks")
        else:
            print(f"�  WARNING - No clear overlap detected (may be normal)")
        
        print(f"   First chunk ends: ...{first_chunk_end[-50:]}")
        print(f"   Second chunk starts: {second_chunk_start[:50]}...")
    else:
        print(f"L SKIP - Need at least 2 chunks to test overlap")
    
    # Test 5: Validate chunks
    print(f"\nTest 5: Running chunk validation")
    if chunks:
        validation_result = chunker.validate_chunks(chunks)
        
        if validation_result['valid']:
            print(f" PASS - All chunks are valid")
        else:
            print(f"�  WARNING - Validation issues found: {len(validation_result['issues'])}")
            for issue in validation_result['issues']:
                print(f"     - {issue}")
        
        stats = validation_result['statistics']
        print(f"   Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"   Average tokens: {stats['avg_tokens']:.1f}")
        print(f"   Total tokens: {stats['total_tokens']}")
        
        if validation_result['recommendations']:
            print(f"   Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"     - {rec}")
    else:
        print(f"L SKIP - No chunks to validate")
    
    # Test 6: Performance statistics
    print(f"\nTest 6: Performance statistics")
    try:
        stats = chunker.get_statistics()
        print(f" PASS - Statistics retrieved")
        print(f"   Documents processed: {stats['total_documents_processed']}")
        print(f"   Chunks created: {stats['total_chunks_created']}")
        print(f"   Tokens processed: {stats['total_tokens_processed']}")
        print(f"   Avg chunks per doc: {stats['average_chunks_per_document']}")
        print(f"   Avg tokens per chunk: {stats['average_tokens_per_chunk']}")
    except Exception as e:
        print(f"L FAIL - Statistics retrieval failed: {e}")
    
    # Test 7: Edge cases
    print(f"\nTest 7: Testing edge cases")
    
    # Empty document
    try:
        empty_doc = Document(content="", metadata={'test': 'empty'})
        chunker.create_chunks(empty_doc)
        print(f"L FAIL - Empty document should raise ValueError")
    except ValueError:
        print(f" PASS - Empty document correctly rejected")
    except Exception as e:
        print(f"L FAIL - Unexpected error with empty document: {e}")
    
    # Very short document
    try:
        short_doc = Document(content="Short text.", metadata={'test': 'short'})
        short_chunks = chunker.create_chunks(short_doc)
        if len(short_chunks) == 1:
            print(f" PASS - Short document created 1 chunk")
        else:
            print(f"�  WARNING - Short document created {len(short_chunks)} chunks")
    except Exception as e:
        print(f"L FAIL - Short document processing failed: {e}")
    
    print(f"\n=== Test Complete ===")
    print("TextChunker is ready for integration with document processing pipeline.")
    print("Chunks are optimized for vector embedding and retrieval operations.")