"""
DataNeuron User-Isolated Session Management Module
==================================================

This module provides session management capabilities for DataNeuron, ensuring complete
user isolation and data persistence. Each user session is tracked separately to prevent
data mixing and enable resuming from previous processing points.

Features:
- User-isolated sessions with unique session IDs
- Content-based file identification using SHA256 hashing
- JSON persistence for session state across restarts
- Comprehensive metadata tracking for processed documents
- Thread-safe operations with atomic file operations
- Graceful error handling and recovery

Usage:
    from core.session_manager import SessionManager, SessionData
    
    manager = SessionManager()
    
    # Add document to specific session
    session_data = SessionData(
        file_hash="abc123...",
        file_name="document.pdf",
        file_path="/path/to/document.pdf",
        processed_at="2024-01-01T12:00:00",
        vector_collection_name="session_123_collection",
        document_metadata={"page_count": 5}
    )
    manager.add_document_to_session("user_session_123", session_data)
    
    # Retrieve session documents
    documents = manager.get_session_documents("user_session_123")
"""

import json
import hashlib
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import SESSION_STATE_FILE, SESSION_TIMEOUT_HOURS


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SessionData:
    """
    Represents metadata for a processed document within a user session.
    
    Attributes:
        file_hash: SHA256 hash of the file content for unique identification
        file_name: Original name of the file
        file_path: Full path to the original file
        processed_at: ISO timestamp when the document was processed
        vector_collection_name: Name of the vector collection storing this document
        document_metadata: Additional metadata from document processing
        file_size_mb: Size of the file in megabytes
        content_preview: First 200 characters of content for quick reference
    """
    file_hash: str
    file_name: str
    file_path: str
    processed_at: str
    vector_collection_name: str
    document_metadata: Dict[str, Any]
    file_size_mb: float = 0.0
    content_preview: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SessionData to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create SessionData from dictionary loaded from JSON."""
        return cls(**data)


# ============================================================================
# SESSION MANAGER CLASS
# ============================================================================

class SessionManager:
    """
    Manages user-isolated sessions with persistent storage.
    
    This class ensures complete data isolation between different users while
    providing efficient access to previously processed documents within each session.
    """
    
    def __init__(self):
        """
        Initialize the session manager.
        
        Loads existing session state from disk or creates a new empty state.
        """
        self.state_file = SESSION_STATE_FILE
        self.timeout_hours = SESSION_TIMEOUT_HOURS
        self._lock = threading.Lock()  # For thread-safe operations
        self.state: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing state
        self._load_state()
        
        logger.info(f"SessionManager initialized with state file: {self.state_file}")
        logger.info(f"Session timeout: {self.timeout_hours} hours")
        logger.info(f"Loaded {len(self.state)} existing sessions")
    
    def _load_state(self) -> None:
        """
        Load session state from JSON file.
        
        Creates empty state if file doesn't exist or is corrupted.
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                    self.state = loaded_state if isinstance(loaded_state, dict) else {}
                    logger.info(f"Successfully loaded session state with {len(self.state)} sessions")
            else:
                self.state = {}
                logger.info("No existing session state file found, starting with empty state")
                
        except json.JSONDecodeError as e:
            logger.error(f"Session state file is corrupted: {e}")
            logger.warning("Starting with empty state")
            self.state = {}
            # Backup corrupted file
            try:
                backup_path = self.state_file.with_suffix('.json.backup')
                self.state_file.rename(backup_path)
                logger.info(f"Corrupted file backed up to: {backup_path}")
            except Exception as backup_e:
                logger.error(f"Failed to backup corrupted file: {backup_e}")
                
        except Exception as e:
            logger.exception(f"Unexpected error loading session state: {e}")
            self.state = {}
    
    def _save_state(self) -> None:
        """
        Save current session state to JSON file atomically.
        
        Uses atomic write operation to prevent data corruption during concurrent access.
        """
        try:
            # Write to temporary file first
            temp_file = self.state_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
            logger.debug(f"Session state saved successfully with {len(self.state)} sessions")
            
        except Exception as e:
            logger.exception(f"Failed to save session state: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate SHA256 hash of file content for unique identification.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            SHA256 hash string or None if file cannot be read
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found for hashing: {file_path}")
                return None
            
            hasher = hashlib.sha256()
            
            with open(path, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            logger.debug(f"Calculated hash for {path.name}: {file_hash[:16]}...")
            return file_hash
            
        except Exception as e:
            logger.exception(f"Error calculating file hash for {file_path}: {e}")
            return None
    
    def get_session_documents(self, session_id: str) -> List[SessionData]:
        """
        Retrieve all processed documents for a specific session.
        
        Args:
            session_id: Unique identifier for the user session
            
        Returns:
            List of SessionData objects for the session
        """
        with self._lock:
            try:
                if session_id not in self.state:
                    logger.debug(f"Session not found: {session_id}")
                    return []
                
                documents = []
                for doc_dict in self.state[session_id]:
                    try:
                        session_data = SessionData.from_dict(doc_dict)
                        documents.append(session_data)
                    except Exception as e:
                        logger.warning(f"Skipping corrupted document data in session {session_id}: {e}")
                
                logger.info(f"Retrieved {len(documents)} documents for session: {session_id}")
                return documents
                
            except Exception as e:
                logger.exception(f"Error retrieving session documents for {session_id}: {e}")
                return []
    
    def get_document_by_hash(self, session_id: str, file_hash: str) -> Optional[SessionData]:
        """
        Find a specific document by its hash within a session.
        
        Args:
            session_id: Unique identifier for the user session
            file_hash: SHA256 hash of the file content
            
        Returns:
            SessionData object if found, None otherwise
        """
        with self._lock:
            try:
                if session_id not in self.state:
                    logger.debug(f"Session not found: {session_id}")
                    return None
                
                for doc_dict in self.state[session_id]:
                    if doc_dict.get('file_hash') == file_hash:
                        logger.info(f"Found document with hash {file_hash[:16]}... in session {session_id}")
                        return SessionData.from_dict(doc_dict)
                
                logger.debug(f"Document with hash {file_hash[:16]}... not found in session {session_id}")
                return None
                
            except Exception as e:
                logger.exception(f"Error finding document by hash in session {session_id}: {e}")
                return None
    
    def is_document_processed(self, session_id: str, file_path: str) -> Optional[SessionData]:
        """
        Check if a document has already been processed in a session.
        
        Args:
            session_id: Unique identifier for the user session
            file_path: Path to the file to check
            
        Returns:
            SessionData if document was already processed, None otherwise
        """
        file_hash = self._calculate_file_hash(file_path)
        if not file_hash:
            return None
        
        return self.get_document_by_hash(session_id, file_hash)
    
    def add_document_to_session(self, session_id: str, session_data: SessionData) -> bool:
        """
        Add a processed document to a user session.
        
        Args:
            session_id: Unique identifier for the user session
            session_data: SessionData object containing document information
            
        Returns:
            True if added successfully, False otherwise
        """
        with self._lock:
            try:
                # Initialize session if it doesn't exist
                if session_id not in self.state:
                    self.state[session_id] = []
                    logger.info(f"Created new session: {session_id}")
                
                # Check if document already exists (prevent duplicates)
                # Use direct lookup to avoid recursive locking
                existing_doc = None
                for doc_dict in self.state[session_id]:
                    if doc_dict.get('file_hash') == session_data.file_hash:
                        existing_doc = SessionData.from_dict(doc_dict)
                        break
                
                if existing_doc:
                    logger.warning(f"Document with hash {session_data.file_hash[:16]}... already exists in session {session_id}")
                    return False
                
                # Add document to session
                self.state[session_id].append(session_data.to_dict())
                self._save_state()
                
                logger.success(f"Added document '{session_data.file_name}' to session {session_id}")
                return True
                
            except Exception as e:
                logger.exception(f"Error adding document to session {session_id}: {e}")
                return False
    
    def remove_document_from_session(self, session_id: str, file_hash: str) -> bool:
        """
        Remove a document from a user session.
        
        Args:
            session_id: Unique identifier for the user session
            file_hash: SHA256 hash of the file to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        with self._lock:
            try:
                if session_id not in self.state:
                    logger.warning(f"Attempted to remove document from non-existent session: {session_id}")
                    return False
                
                # Find and remove the document
                original_count = len(self.state[session_id])
                self.state[session_id] = [
                    doc for doc in self.state[session_id]
                    if doc.get('file_hash') != file_hash
                ]
                
                if len(self.state[session_id]) < original_count:
                    self._save_state()
                    logger.success(f"Removed document with hash {file_hash[:16]}... from session {session_id}")
                    
                    # Remove session if empty
                    if not self.state[session_id]:
                        del self.state[session_id]
                        self._save_state()
                        logger.info(f"Removed empty session: {session_id}")
                    
                    return True
                else:
                    logger.warning(f"Document with hash {file_hash[:16]}... not found in session {session_id}")
                    return False
                    
            except Exception as e:
                logger.exception(f"Error removing document from session {session_id}: {e}")
                return False
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all documents from a user session.
        
        Args:
            session_id: Unique identifier for the user session
            
        Returns:
            True if cleared successfully, False otherwise
        """
        with self._lock:
            try:
                if session_id in self.state:
                    document_count = len(self.state[session_id])
                    del self.state[session_id]
                    self._save_state()
                    logger.success(f"Cleared session {session_id} with {document_count} documents")
                    return True
                else:
                    logger.warning(f"Attempted to clear non-existent session: {session_id}")
                    return False
                    
            except Exception as e:
                logger.exception(f"Error clearing session {session_id}: {e}")
                return False
    
    def get_all_sessions(self) -> Dict[str, int]:
        """
        Get summary of all sessions.
        
        Returns:
            Dictionary mapping session_id to document count
        """
        with self._lock:
            try:
                sessions = {
                    session_id: len(documents)
                    for session_id, documents in self.state.items()
                }
                logger.info(f"Retrieved summary of {len(sessions)} sessions")
                return sessions
                
            except Exception as e:
                logger.exception(f"Error getting session summary: {e}")
                return {}
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions based on timeout configuration.
        
        Returns:
            Number of expired sessions removed
        """
        # This is a placeholder for future implementation
        # Would need to track last access time and compare with timeout
        logger.info("Session cleanup not yet implemented")
        return 0


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the session manager with various user isolation scenarios.
    This verifies that sessions remain completely isolated from each other.
    """
    
    print("=== DataNeuron Session Manager Test ===")
    
    # Initialize session manager
    manager = SessionManager()
    
    # Test data
    session1_id = "user_alice_session_123"
    session2_id = "user_bob_session_456"
    
    # Create test session data
    doc1_alice = SessionData(
        file_hash="abc123def456",
        file_name="alice_report.pdf",
        file_path="/fake/path/alice_report.pdf",
        processed_at=datetime.now().isoformat(),
        vector_collection_name=f"{session1_id}_collection",
        document_metadata={"page_count": 10, "author": "Alice"},
        file_size_mb=2.5,
        content_preview="This is Alice's confidential report..."
    )
    
    doc2_alice = SessionData(
        file_hash="ghi789jkl012",
        file_name="alice_notes.txt",
        file_path="/fake/path/alice_notes.txt",
        processed_at=datetime.now().isoformat(),
        vector_collection_name=f"{session1_id}_collection",
        document_metadata={"line_count": 50, "encoding": "utf-8"},
        file_size_mb=0.1,
        content_preview="Alice's personal notes and thoughts..."
    )
    
    doc1_bob = SessionData(
        file_hash="mno345pqr678",
        file_name="bob_proposal.docx",
        file_path="/fake/path/bob_proposal.docx",
        processed_at=datetime.now().isoformat(),
        vector_collection_name=f"{session2_id}_collection",
        document_metadata={"paragraph_count": 25, "author": "Bob"},
        file_size_mb=1.8,
        content_preview="Bob's business proposal for Q4..."
    )
    
    print("\nRunning user isolation tests:\n")
    
    # Test 1: Add documents to different sessions
    print("Test 1: Adding documents to different user sessions")
    success1 = manager.add_document_to_session(session1_id, doc1_alice)
    success2 = manager.add_document_to_session(session1_id, doc2_alice)
    success3 = manager.add_document_to_session(session2_id, doc1_bob)
    
    if success1 and success2 and success3:
        print(" PASS - All documents added successfully")
    else:
        print("L FAIL - Some documents failed to add")
    
    # Test 2: Verify session isolation
    print("\nTest 2: Verifying complete session isolation")
    alice_docs = manager.get_session_documents(session1_id)
    bob_docs = manager.get_session_documents(session2_id)
    
    print(f"  Alice's session ({session1_id}): {len(alice_docs)} documents")
    for doc in alice_docs:
        print(f"    - {doc.file_name} (hash: {doc.file_hash[:16]}...)")
    
    print(f"  Bob's session ({session2_id}): {len(bob_docs)} documents")
    for doc in bob_docs:
        print(f"    - {doc.file_name} (hash: {doc.file_hash[:16]}...)")
    
    if len(alice_docs) == 2 and len(bob_docs) == 1:
        print(" PASS - Sessions are properly isolated")
    else:
        print("L FAIL - Session isolation failed")
    
    # Test 3: Document lookup by hash
    print("\nTest 3: Document lookup by hash within sessions")
    alice_doc1_found = manager.get_document_by_hash(session1_id, doc1_alice.file_hash)
    bob_doc1_found = manager.get_document_by_hash(session2_id, doc1_bob.file_hash)
    
    # Cross-session lookup (should fail)
    alice_doc_in_bob_session = manager.get_document_by_hash(session2_id, doc1_alice.file_hash)
    
    if (alice_doc1_found and bob_doc1_found and not alice_doc_in_bob_session):
        print(" PASS - Document lookup working correctly with session isolation")
    else:
        print("L FAIL - Document lookup failed or broke session isolation")
    
    # Test 4: Duplicate prevention
    print("\nTest 4: Duplicate document prevention")
    duplicate_attempt = manager.add_document_to_session(session1_id, doc1_alice)
    
    if not duplicate_attempt:
        print(" PASS - Duplicate documents correctly prevented")
    else:
        print("L FAIL - Duplicate document was allowed")
    
    # Test 5: Document removal
    print("\nTest 5: Document removal from session")
    removal_success = manager.remove_document_from_session(session1_id, doc2_alice.file_hash)
    alice_docs_after_removal = manager.get_session_documents(session1_id)
    
    if removal_success and len(alice_docs_after_removal) == 1:
        print(" PASS - Document removal successful")
    else:
        print("L FAIL - Document removal failed")
    
    # Test 6: Session summary
    print("\nTest 6: Session summary")
    all_sessions = manager.get_all_sessions()
    print(f"  Total sessions: {len(all_sessions)}")
    for session_id, doc_count in all_sessions.items():
        print(f"    {session_id}: {doc_count} documents")
    
    # Test 7: Persistence (simulate restart)
    print("\nTest 7: Testing persistence across restarts")
    
    # Create new manager instance (simulates app restart)
    manager2 = SessionManager()
    alice_docs_reloaded = manager2.get_session_documents(session1_id)
    bob_docs_reloaded = manager2.get_session_documents(session2_id)
    
    if (len(alice_docs_reloaded) == 1 and len(bob_docs_reloaded) == 1):
        print(" PASS - Session state persisted correctly across restart")
    else:
        print("L FAIL - Session persistence failed")
    
    # Clean up test data
    print("\nTest 8: Session cleanup")
    manager2.clear_session(session1_id)
    manager2.clear_session(session2_id)
    
    final_sessions = manager2.get_all_sessions()
    if len(final_sessions) == 0:
        print(" PASS - Sessions cleaned up successfully")
    else:
        print("L FAIL - Session cleanup incomplete")
    
    print("\n=== Test Complete ===")
    print("All session isolation and persistence tests completed.")
    print("Check the logs above for detailed session management information.")