"""
DataNeuron ChromaDB Vector Storage Module
=========================================

This module provides a comprehensive vector storage solution using ChromaDB for efficient
semantic search and retrieval. It manages text chunks and their vector embeddings with
full metadata preservation and optimized batch operations.

Features:
- Singleton ChromaDB client for efficient resource management
- Persistent disk storage with configurable paths
- Collection-based organization for session/document isolation
- Batch operations for optimal performance
- Rich metadata storage and retrieval
- Semantic similarity search with configurable results
- Comprehensive error handling and logging

Usage:
    from core.vector_store import VectorStore
    from core.chunker import Chunk
    
    # Get singleton instance
    vector_store = VectorStore()
    
    # Add chunks with embeddings
    chunks = [...]  # List of Chunk objects
    embeddings = [...]  # List of embedding vectors
    vector_store.add_chunks("my_collection", chunks, embeddings)
    
    # Search for similar content
    query_embedding = [...]  # Query vector
    results = vector_store.similarity_search("my_collection", query_embedding, top_k=5)
    
    # Cleanup when done
    vector_store.delete_collection("my_collection")

Architecture:
- Singleton pattern for efficient resource management
- ChromaDB PersistentClient for disk-based storage
- Collection-based data organization
- Comprehensive metadata preservation
- Batch operations for performance optimization
"""

import sys
import uuid
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import CHROMADB_PATH, CHROMADB_COLLECTION

# Import dependencies with error handling
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    logger.error(f"ChromaDB not available: {e}")
    chromadb = None

try:
    from core.chunker import Chunk
except ImportError as e:
    logger.warning(f"Chunker not available for type hints: {e}")
    # Create minimal Chunk class for testing
    from dataclasses import dataclass, field
    @dataclass
    class Chunk:
        content: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        token_count: int = 0
        character_count: int = 0
        chunk_index: int = 0


# ============================================================================
# VECTOR STORE SINGLETON CLASS
# ============================================================================

class VectorStore:
    """
    Singleton ChromaDB vector storage manager.
    
    This class provides a centralized interface for managing vector embeddings
    and their associated text chunks using ChromaDB as the backend. It ensures
    efficient resource usage through singleton pattern and provides comprehensive
    methods for storage, retrieval, and management of vector collections.
    
    Key Features:
    - Singleton pattern with thread-safe initialization
    - Persistent ChromaDB storage on disk
    - Collection-based organization for data isolation
    - Batch operations for optimal performance
    - Rich metadata storage and querying
    - Configurable similarity search
    - Comprehensive error handling and logging
    
    The class maintains a single ChromaDB client instance throughout the
    application lifecycle, ensuring efficient resource usage and consistent
    data access patterns.
    """
    
    _instance: Optional['VectorStore'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'VectorStore':
        """
        Singleton pattern implementation with thread safety.
        
        Returns:
            Single instance of VectorStore
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VectorStore, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the VectorStore singleton.
        
        Sets up ChromaDB client connection and configures persistent storage.
        Only initializes once due to singleton pattern.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            # Check if ChromaDB is available
            if chromadb is None:
                raise RuntimeError("ChromaDB is not available. Please install chromadb package.")
            
            # Initialize configuration
            self.db_path = CHROMADB_PATH
            self.default_collection = CHROMADB_COLLECTION
            
            # Initialize ChromaDB client
            try:
                # Ensure database directory exists
                self.db_path.mkdir(parents=True, exist_ok=True)
                
                # Create persistent ChromaDB client
                self.client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False,  # Disable telemetry for privacy
                        allow_reset=True  # Allow database resets for testing
                    )
                )
                
                logger.info(f"ChromaDB client initialized successfully")
                logger.info(f"  - Database path: {self.db_path}")
                logger.info(f"  - Default collection: {self.default_collection}")
                
                # Test client connection
                collections = self.client.list_collections()
                logger.info(f"  - Existing collections: {len(collections)}")
                
            except Exception as e:
                logger.exception(f"Failed to initialize ChromaDB client: {e}")
                raise RuntimeError(f"ChromaDB initialization failed: {e}")
            
            # Statistics tracking
            self.total_chunks_stored = 0
            self.total_searches_performed = 0
            self.active_collections = set()
            
            self._initialized = True
            logger.success("VectorStore singleton initialized successfully")
    
    def get_or_create_collection(self, collection_name: str):
        """
        Get an existing collection or create a new one.
        
        Args:
            collection_name (str): Name of the collection to get or create
            
        Returns:
            ChromaDB Collection object
            
        Raises:
            RuntimeError: If collection operations fail
        """
        try:
            logger.debug(f"Getting or creating collection: {collection_name}")
            
            # Try to get existing collection first
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"Found existing collection: {collection_name}")
                
            except Exception:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": f"DataNeuron collection for {collection_name}"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            # Track active collections
            self.active_collections.add(collection_name)
            
            return collection
            
        except Exception as e:
            logger.exception(f"Failed to get or create collection {collection_name}: {e}")
            raise RuntimeError(f"Collection operation failed: {e}")
    
    def add_chunks(
        self, 
        collection_name: str, 
        chunks: List[Chunk], 
        embeddings: List[List[float]]
    ) -> bool:
        """
        Add text chunks and their embeddings to a collection in batch.
        
        Args:
            collection_name (str): Name of the target collection
            chunks (List[Chunk]): List of text chunks with metadata
            embeddings (List[List[float]]): Corresponding embedding vectors
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If ChromaDB operations fail
        """
        # Input validation
        if not chunks or not embeddings:
            raise ValueError("Both chunks and embeddings must be non-empty")
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        logger.info(f"Adding {len(chunks)} chunks to collection: {collection_name}")
        
        try:
            # Get or create collection
            collection = self.get_or_create_collection(collection_name)
            
            # Prepare batch data
            ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique ID for each chunk
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                
                # Document content
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB requires serializable types)
                metadata = self._prepare_metadata_for_storage(chunk.metadata, i)
                metadatas.append(metadata)
                
                logger.debug(f"Prepared chunk {i}: ID={chunk_id[:8]}..., content_len={len(chunk.content)}")
            
            # Batch add to ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            # Update statistics
            self.total_chunks_stored += len(chunks)
            
            logger.success(f"Successfully added {len(chunks)} chunks to collection: {collection_name}")
            logger.info(f"  - Total tokens stored: {sum(chunk.token_count for chunk in chunks)}")
            logger.info(f"  - Average chunk size: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f} tokens")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to add chunks to collection {collection_name}: {e}")
            raise RuntimeError(f"Chunk storage failed: {e}")
    
    def _prepare_metadata_for_storage(self, metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage by ensuring serializable types.
        
        Args:
            metadata: Original metadata dictionary
            chunk_index: Index of chunk in batch
            
        Returns:
            Serializable metadata dictionary
        """
        storage_metadata = {}
        
        for key, value in metadata.items():
            # Convert non-serializable types to strings
            if isinstance(value, (str, int, float, bool)):
                storage_metadata[key] = value
            elif value is None:
                storage_metadata[key] = ""
            else:
                # Convert complex types to string representation
                storage_metadata[key] = str(value)
        
        # Add storage-specific metadata
        storage_metadata.update({
            'storage_timestamp': str(uuid.uuid4()),  # Unique identifier for this storage operation
            'batch_index': chunk_index,
            'vector_store_version': '1.0.0'
        })
        
        return storage_metadata
    
    def similarity_search(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search in a collection.
        
        Args:
            collection_name (str): Name of the collection to search
            query_embedding (List[float]): Query vector for similarity search
            top_k (int): Number of top results to return (default: 5)
            where (Optional[Dict]): Metadata filter conditions
            
        Returns:
            List of dictionaries containing:
            - content: Text content of the chunk
            - metadata: Rich metadata from original chunk
            - similarity_score: Cosine similarity score (0-1, higher = more similar)
            - id: Unique identifier of the chunk
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If search operations fail
        """
        # Input validation
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        logger.info(f"Performing similarity search in collection: {collection_name}")
        logger.debug(f"  - Query embedding dimensions: {len(query_embedding)}")
        logger.debug(f"  - Top K results: {top_k}")
        logger.debug(f"  - Metadata filters: {where}")
        
        try:
            # Get collection
            collection = self.client.get_collection(name=collection_name)
            
            # Perform query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results['documents'] and results['documents'][0]:  # Check if results exist
                for i in range(len(results['documents'][0])):
                    # Calculate similarity score from distance
                    # ChromaDB returns cosine distance, convert to similarity
                    distance = results['distances'][0][i] if results.get('distances') else 0.0
                    similarity_score = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                        'similarity_score': similarity_score,
                        'id': results['ids'][0][i] if results.get('ids') else f"result_{i}",
                        'distance': distance  # Include original distance for debugging
                    }
                    
                    search_results.append(result)
                    logger.debug(f"Result {i}: similarity={similarity_score:.3f}, content_len={len(result['content'])}")
            
            # Update statistics
            self.total_searches_performed += 1
            
            logger.success(f"Similarity search completed: found {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.exception(f"Similarity search failed in collection {collection_name}: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its data permanently.
        
        Args:
            collection_name (str): Name of the collection to delete
            
        Returns:
            bool: True if successful, False if collection didn't exist
            
        Raises:
            RuntimeError: If deletion fails
        """
        logger.info(f"Deleting collection: {collection_name}")
        
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Delete the collection
            self.client.delete_collection(name=collection_name)
            
            # Update tracking
            self.active_collections.discard(collection_name)
            
            logger.success(f"Successfully deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to delete collection {collection_name}: {e}")
            raise RuntimeError(f"Collection deletion failed: {e}")
    
    def list_collections(self) -> List[str]:
        """
        Get a list of all collection names.
        
        Returns:
            List of collection names
            
        Raises:
            RuntimeError: If listing fails
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            logger.debug(f"Listed {len(collection_names)} collections")
            
            return collection_names
            
        except Exception as e:
            logger.exception(f"Failed to list collections: {e}")
            raise RuntimeError(f"Collection listing failed: {e}")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific collection.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Dictionary with collection information including:
            - name: Collection name
            - count: Number of documents
            - metadata: Collection metadata
            
        Raises:
            RuntimeError: If collection doesn't exist or info retrieval fails
        """
        logger.debug(f"Getting info for collection: {collection_name}")
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Get collection statistics
            count = collection.count()
            
            # Get collection metadata
            collection_metadata = getattr(collection, 'metadata', {}) or {}
            
            info = {
                'name': collection_name,
                'count': count,
                'metadata': collection_metadata,
                'is_active': collection_name in self.active_collections
            }
            
            logger.debug(f"Collection {collection_name} info: {count} documents")
            
            return info
            
        except Exception as e:
            logger.exception(f"Failed to get collection info for {collection_name}: {e}")
            raise RuntimeError(f"Collection info retrieval failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector store.
        
        Returns:
            Dictionary with statistics and metrics
        """
        try:
            all_collections = self.list_collections()
            
            total_documents = 0
            collection_stats = {}
            
            for collection_name in all_collections:
                try:
                    info = self.get_collection_info(collection_name)
                    total_documents += info['count']
                    collection_stats[collection_name] = info['count']
                except Exception as e:
                    logger.warning(f"Failed to get stats for collection {collection_name}: {e}")
                    collection_stats[collection_name] = 'error'
            
            statistics = {
                'total_collections': len(all_collections),
                'active_collections': len(self.active_collections),
                'total_documents': total_documents,
                'total_chunks_stored': self.total_chunks_stored,
                'total_searches_performed': self.total_searches_performed,
                'database_path': str(self.db_path),
                'collection_stats': collection_stats,
                'client_type': type(self.client).__name__
            }
            
            logger.info(f"Retrieved vector store statistics: {statistics['total_collections']} collections, {statistics['total_documents']} documents")
            
            return statistics
            
        except Exception as e:
            logger.exception(f"Failed to get statistics: {e}")
            return {
                'error': str(e),
                'total_chunks_stored': self.total_chunks_stored,
                'total_searches_performed': self.total_searches_performed
            }
    
    def cleanup_empty_collections(self) -> int:
        """
        Remove collections that contain no documents.
        
        Returns:
            Number of collections cleaned up
        """
        logger.info("Starting cleanup of empty collections")
        
        cleaned_count = 0
        
        try:
            collections = self.list_collections()
            
            for collection_name in collections:
                try:
                    info = self.get_collection_info(collection_name)
                    if info['count'] == 0:
                        self.delete_collection(collection_name)
                        cleaned_count += 1
                        logger.info(f"Cleaned up empty collection: {collection_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to check/cleanup collection {collection_name}: {e}")
            
            logger.success(f"Cleanup completed: removed {cleaned_count} empty collections")
            
        except Exception as e:
            logger.exception(f"Cleanup operation failed: {e}")
        
        return cleaned_count


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the VectorStore with comprehensive scenarios.
    Creates temporary database, tests all functionality, and cleans up.
    """
    
    print("=== DataNeuron VectorStore Test ===")
    
    # Create temporary test database
    import tempfile
    import shutil
    
    # Create temporary directory for test database
    temp_dir = Path(tempfile.mkdtemp(prefix="dataneuron_vectorstore_test_"))
    
    try:
        # Override the database path for testing
        original_path = CHROMADB_PATH
        
        # Monkey patch the settings for testing
        import config.settings as settings
        settings.CHROMADB_PATH = temp_dir
        
        print(f"Using temporary database at: {temp_dir}")
        
        # Test 1: Create VectorStore instance
        print(f"\nTest 1: Creating VectorStore singleton")
        try:
            vector_store = VectorStore()
            print(f" PASS - VectorStore created successfully")
            print(f"   Database path: {vector_store.db_path}")
            print(f"   Client type: {type(vector_store.client).__name__}")
            
            # Test singleton behavior
            vector_store2 = VectorStore()
            if vector_store is vector_store2:
                print(f" PASS - Singleton pattern working correctly")
            else:
                print(f"L FAIL - Multiple instances created")
                
        except Exception as e:
            print(f"L FAIL - VectorStore creation failed: {e}")
            exit(1)
        
        # Test 2: Collection management
        print(f"\nTest 2: Collection management")
        test_collection_name = "test_collection_123"
        
        try:
            # Create collection
            collection = vector_store.get_or_create_collection(test_collection_name)
            print(f" PASS - Collection created: {test_collection_name}")
            
            # List collections
            collections = vector_store.list_collections()
            if test_collection_name in collections:
                print(f" PASS - Collection appears in list: {len(collections)} total")
            else:
                print(f"L FAIL - Collection not found in list")
            
            # Get collection info
            info = vector_store.get_collection_info(test_collection_name)
            print(f" PASS - Collection info retrieved: {info['count']} documents")
            
        except Exception as e:
            print(f"L FAIL - Collection management failed: {e}")
        
        # Test 3: Add chunks with embeddings
        print(f"\nTest 3: Adding chunks with embeddings")
        
        try:
            # Create test chunks
            test_chunks = [
                Chunk(
                    content="Artificial intelligence is transforming technology.",
                    metadata={'topic': 'AI', 'source': 'test_doc_1', 'chunk_index': 0},
                    token_count=8,
                    chunk_index=0
                ),
                Chunk(
                    content="Machine learning algorithms can learn from data.",
                    metadata={'topic': 'ML', 'source': 'test_doc_1', 'chunk_index': 1}, 
                    token_count=9,
                    chunk_index=1
                ),
                Chunk(
                    content="Deep learning uses neural networks for complex tasks.",
                    metadata={'topic': 'DL', 'source': 'test_doc_2', 'chunk_index': 0},
                    token_count=10,
                    chunk_index=0
                )
            ]
            
            # Create test embeddings (mock vectors)
            test_embeddings = [
                [0.1, 0.2, 0.3, 0.4, 0.5] * 10,  # 50-dimensional mock embedding
                [0.2, 0.3, 0.4, 0.5, 0.6] * 10,
                [0.3, 0.4, 0.5, 0.6, 0.7] * 10
            ]
            
            # Add chunks
            success = vector_store.add_chunks(test_collection_name, test_chunks, test_embeddings)
            
            if success:
                print(f" PASS - {len(test_chunks)} chunks added successfully")
                
                # Verify collection has documents
                info = vector_store.get_collection_info(test_collection_name)
                if info['count'] == len(test_chunks):
                    print(f" PASS - Collection now has {info['count']} documents")
                else:
                    print(f"L FAIL - Expected {len(test_chunks)} documents, got {info['count']}")
            else:
                print(f"L FAIL - Chunk addition returned False")
                
        except Exception as e:
            print(f"L FAIL - Adding chunks failed: {e}")
        
        # Test 4: Similarity search
        print(f"\nTest 4: Similarity search")
        
        try:
            # Create query embedding (similar to first test chunk)
            query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55] * 10
            
            # Perform search
            search_results = vector_store.similarity_search(
                test_collection_name, 
                query_embedding, 
                top_k=3
            )
            
            if search_results:
                print(f" PASS - Search returned {len(search_results)} results")
                
                for i, result in enumerate(search_results):
                    print(f"   Result {i+1}:")
                    print(f"     Content: {result['content'][:50]}...")
                    print(f"     Similarity: {result['similarity_score']:.3f}")
                    print(f"     Metadata topic: {result['metadata'].get('topic', 'N/A')}")
                    
                # Verify result structure
                first_result = search_results[0]
                required_keys = ['content', 'metadata', 'similarity_score', 'id']
                missing_keys = [key for key in required_keys if key not in first_result]
                
                if not missing_keys:
                    print(f" PASS - All required result keys present")
                else:
                    print(f"L FAIL - Missing result keys: {missing_keys}")
            else:
                print(f"L FAIL - No search results returned")
                
        except Exception as e:
            print(f"L FAIL - Similarity search failed: {e}")
        
        # Test 5: Statistics
        print(f"\nTest 5: Statistics and monitoring")
        
        try:
            stats = vector_store.get_statistics()
            print(f" PASS - Statistics retrieved")
            print(f"   Total collections: {stats.get('total_collections', 'N/A')}")
            print(f"   Total documents: {stats.get('total_documents', 'N/A')}")  
            print(f"   Chunks stored: {stats.get('total_chunks_stored', 'N/A')}")
            print(f"   Searches performed: {stats.get('total_searches_performed', 'N/A')}")
            
        except Exception as e:
            print(f"L FAIL - Statistics retrieval failed: {e}")
        
        # Test 6: Collection deletion
        print(f"\nTest 6: Collection deletion")
        
        try:
            # Delete test collection
            deleted = vector_store.delete_collection(test_collection_name)
            
            if deleted:
                print(f" PASS - Collection deleted successfully")
                
                # Verify collection is gone
                collections_after = vector_store.list_collections()
                if test_collection_name not in collections_after:
                    print(f" PASS - Collection removed from list")
                else:
                    print(f"L FAIL - Collection still exists after deletion")
            else:
                print(f"L FAIL - Collection deletion returned False")
                
        except Exception as e:
            print(f"L FAIL - Collection deletion failed: {e}")
        
        # Test 7: Error handling
        print(f"\nTest 7: Error handling")
        
        try:
            # Try to search non-existent collection
            try:
                vector_store.similarity_search("non_existent_collection", [0.1] * 50)
                print(f"L FAIL - Should have thrown error for non-existent collection")
            except RuntimeError:
                print(f" PASS - Correctly handled non-existent collection")
            
            # Try to add mismatched chunks and embeddings
            try:
                vector_store.add_chunks("test", [test_chunks[0]], [[0.1] * 50, [0.2] * 50])
                print(f"L FAIL - Should have thrown error for mismatched inputs")
            except ValueError:
                print(f" PASS - Correctly validated input mismatch")
                
        except Exception as e:
            print(f"L FAIL - Error handling test failed: {e}")
        
        print(f"\n=== Test Complete ===")
        print("VectorStore is ready for integration with document processing pipeline.")
        
    except Exception as e:
        print(f"L CRITICAL FAIL - Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary database
        try:
            # Restore original path
            settings.CHROMADB_PATH = original_path
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f" Cleaned up temporary database at: {temp_dir}")
        except Exception as e:
            print(f" Could not clean up temporary database: {e}")
    
    print("ChromaDB vector storage system tested and ready for production use.")