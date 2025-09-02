#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud-Compatible Vector Store for DataNeuron
===========================================

Simple vector storage that works on Streamlit Cloud without ChromaDB dependencies.
Uses in-memory storage with JSON persistence for session data.
"""

import json
import uuid
import numpy as np
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger

try:
    from core.chunker import Chunk
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class Chunk:
        content: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        token_count: int = 0
        character_count: int = 0
        chunk_index: int = 0


class CloudVectorStore:
    """
    Simple cloud-compatible vector store.
    Uses in-memory storage with optional JSON persistence.
    """
    
    _instance: Optional['CloudVectorStore'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'CloudVectorStore':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CloudVectorStore, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            # In-memory storage
            self.collections: Dict[str, Dict[str, Any]] = {}
            self.storage_path = Path("data/vector_storage.json")
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing data if available
            self._load_from_disk()
            
            # Statistics
            self.total_chunks_stored = 0
            self.total_searches_performed = 0
            self.active_collections = set()
            
            self._initialized = True
            logger.info("CloudVectorStore initialized successfully")
    
    def _load_from_disk(self):
        """Load collections from disk if available."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.collections = data.get('collections', {})
                    logger.info(f"Loaded {len(self.collections)} collections from disk")
        except Exception as e:
            logger.warning(f"Could not load from disk: {e}")
            self.collections = {}
    
    def _save_to_disk(self):
        """Save collections to disk."""
        try:
            data = {
                'collections': self.collections,
                'metadata': {
                    'total_chunks_stored': self.total_chunks_stored,
                    'total_searches_performed': self.total_searches_performed
                }
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save to disk: {e}")
    
    def get_or_create_collection(self, collection_name: str):
        """Get or create a collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                'documents': [],
                'embeddings': [],
                'metadatas': [],
                'ids': [],
                'created_at': str(uuid.uuid4())
            }
            logger.info(f"Created collection: {collection_name}")
        
        self.active_collections.add(collection_name)
        return self.collections[collection_name]
    
    def add_chunks(
        self, 
        collection_name: str, 
        chunks: List[Chunk], 
        embeddings: List[List[float]]
    ) -> bool:
        """Add chunks and embeddings to collection."""
        try:
            if not chunks or not embeddings:
                raise ValueError("Chunks and embeddings cannot be empty")
            
            if len(chunks) != len(embeddings):
                raise ValueError("Chunks and embeddings count mismatch")
            
            collection = self.get_or_create_collection(collection_name)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = str(uuid.uuid4())
                
                collection['ids'].append(chunk_id)
                collection['documents'].append(chunk.content)
                collection['embeddings'].append(embedding)
                collection['metadatas'].append(chunk.metadata)
            
            self.total_chunks_stored += len(chunks)
            self._save_to_disk()
            
            logger.success(f"Added {len(chunks)} chunks to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return False
    
    def similarity_search(
        self, 
        collection_name: str, 
        query_embedding: List[float], 
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine similarity."""
        try:
            if collection_name not in self.collections:
                raise RuntimeError(f"Collection {collection_name} not found")
            
            collection = self.collections[collection_name]
            
            if not collection['embeddings']:
                return []
            
            # Calculate cosine similarities
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, embedding in enumerate(collection['embeddings']):
                doc_vec = np.array(embedding)
                
                # Cosine similarity
                dot_product = np.dot(query_vec, doc_vec)
                query_norm = np.linalg.norm(query_vec)
                doc_norm = np.linalg.norm(doc_vec)
                
                if query_norm == 0 or doc_norm == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (query_norm * doc_norm)
                
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top_k results
            results = []
            for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
                result = {
                    'content': collection['documents'][doc_idx],
                    'metadata': collection['metadatas'][doc_idx],
                    'similarity_score': float(similarity),
                    'id': collection['ids'][doc_idx],
                    'distance': 1.0 - similarity
                }
                results.append(result)
            
            self.total_searches_performed += 1
            logger.debug(f"Search in {collection_name}: {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
                self.active_collections.discard(collection_name)
                self._save_to_disk()
                logger.info(f"Deleted collection: {collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        return list(self.collections.keys())
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        if collection_name not in self.collections:
            raise RuntimeError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        return {
            'name': collection_name,
            'count': len(collection['documents']),
            'metadata': {'created_at': collection['created_at']},
            'is_active': collection_name in self.active_collections
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        total_documents = sum(len(col['documents']) for col in self.collections.values())
        
        return {
            'total_collections': len(self.collections),
            'active_collections': len(self.active_collections),
            'total_documents': total_documents,
            'total_chunks_stored': self.total_chunks_stored,
            'total_searches_performed': self.total_searches_performed,
            'database_path': str(self.storage_path),
            'collection_stats': {name: len(col['documents']) for name, col in self.collections.items()},
            'client_type': 'CloudVectorStore'
        }
    
    def cleanup_empty_collections(self) -> int:
        """Remove empty collections."""
        empty_collections = [
            name for name, col in self.collections.items() 
            if len(col['documents']) == 0
        ]
        
        for name in empty_collections:
            self.delete_collection(name)
        
        return len(empty_collections)


# Create alias for backward compatibility
VectorStore = CloudVectorStore