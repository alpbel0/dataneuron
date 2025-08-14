#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataNeuron Streamlit Frontend Application
=========================================

Complete Streamlit interface for DataNeuron project following Clean Code and SOLID principles.
Integrates all backend components with proper error handling and user feedback.

Architecture:
- Modular functions with single responsibilities
- Singleton pattern for backend components
- Proper async handling and user feedback
- Session-based data isolation
"""

import asyncio
import hashlib
import uuid
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from config.settings import UPLOADS_DIR

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import backend components with exact names from requirements
from core.document_processor import DocumentProcessor, Document
from core.chunker import TextChunker, Chunk
from core.embedder import Embedder
from core.vector_store import VectorStore
from core.session_manager import SessionManager, SessionData
from core.llm_agent import LLMAgent, CoTSession
from utils.logger import logger
from pydantic import BaseModel

def initialize_session_state():
    """Initialize all required Streamlit session state variables."""
    logger.info("Initializing Streamlit session state")
    
    # Generate unique session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated session ID: {st.session_state.session_id}")
    
    # Initialize state variables first
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Initialize backend singletons with error handling
    try:
        if 'document_processor' not in st.session_state:
            st.session_state.document_processor = DocumentProcessor()
            logger.info("DocumentProcessor initialized")
        
        if 'text_chunker' not in st.session_state:
            st.session_state.text_chunker = TextChunker()
            logger.info("TextChunker initialized")
        
        if 'embedder' not in st.session_state:
            st.session_state.embedder = Embedder()
            logger.info("Embedder initialized")
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStore()
            logger.info("VectorStore initialized")
        
        if 'session_manager' not in st.session_state:
            st.session_state.session_manager = SessionManager()
            logger.info("SessionManager initialized")
        
        if 'llm_agent' not in st.session_state:
            st.session_state.llm_agent = LLMAgent()
            logger.info("LLMAgent initialized")
            
    except Exception as e:
        logger.exception(f"Backend initialization failed: {e}")
        st.error(f"Backend initialization failed: {str(e)}")
        st.stop()
    
    # Load existing documents
    if not st.session_state.uploaded_documents:
        try:
            existing_docs = st.session_state.session_manager.get_session_documents(st.session_state.session_id)
            st.session_state.uploaded_documents = existing_docs
            logger.info(f"Loaded {len(existing_docs)} existing documents")
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")
    
    logger.success("Session state initialization completed")


def render_sidebar():
    """Render sidebar with document management and system information."""
    with st.sidebar:
        st.header("📄 Documents")
        
        # Display uploaded documents
        if st.session_state.uploaded_documents:
            st.write(f"**{len(st.session_state.uploaded_documents)} documents loaded:**")
            
            for doc in st.session_state.uploaded_documents:
                with st.expander(f"{doc.file_name} ({doc.file_size_mb:.1f}MB)"):
                    st.write(f"**Processed:** {doc.processed_at[:19]}")
                    st.write(f"**Collection:** {doc.vector_collection_name}")
                    if doc.content_preview:
                        st.write(f"**Preview:** {doc.content_preview[:100]}...")
                    
                    # Remove button (disabled during processing)
                    if st.button(
                        "Remove", 
                        key=f"remove_{doc.file_hash}",
                        disabled=st.session_state.is_processing
                    ):
                        remove_document_from_session(doc.file_hash)
        else:
            st.info("No documents uploaded yet")
        
        # System Status
        st.header("System Status")
        with st.expander("Backend Components"):
            components = [
                ("Document Processor", 'document_processor'),
                ("Text Chunker", 'text_chunker'),
                ("Embedder", 'embedder'),
                ("Vector Store", 'vector_store'),
                ("Session Manager", 'session_manager'),
                ("LLM Agent", 'llm_agent')
            ]
            
            for name, key in components:
                status = "✅" if key in st.session_state else "❌"
                st.write(f"{status} {name}")
        
        with st.expander("Session Information"):
            st.write(f"**Session ID:** `{st.session_state.session_id}`")
            st.write(f"**Chat History:** {len(st.session_state.chat_history)} messages")
            processing_status = "🔄 Processing" if st.session_state.is_processing else "✅ Ready"
            st.write(f"**Status:** {processing_status}")
        
        # Clear session button
        st.header("Session Management")
        if st.button(
            "Clear All Data", 
            type="secondary",
            disabled=st.session_state.is_processing
        ):
            clear_all_session_data()


def render_chat_interface():
    """Renders the main chat UI and handles all user interactions."""
    st.header("💬 Chat Interface")

    # 1. Display chat history from session state
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "cot_session" in message:
                with st.expander("🧠 View Agent's Reasoning"):
                    display_cot_visualization(message["cot_session"])

    # 2. Get user input using a single, keyed chat_input
    if prompt := st.chat_input("Ask about your documents...", key="chat_widget", disabled=st.session_state.is_processing):
        # Add user message to history and rerun to display it immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    # 3. If the last message is from the user, process it
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        # Set processing flag
        st.session_state.is_processing = True
        
        # Get query and history for the agent
        last_query = st.session_state.chat_history[-1]["content"]
        history_for_agent = st.session_state.chat_history[:-1]

        # Display thinking message
        with st.chat_message("assistant"):
            with st.spinner("🧠 Agent is thinking..."):
                try:
                    # Run the agent
                    cot_session = asyncio.run(
                        st.session_state.llm_agent.execute_with_cot(
                            query=last_query,
                            session_id=st.session_state.session_id,
                            chat_history=history_for_agent
                        )
                    )
                    
                    # Add agent's response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": cot_session.final_answer,
                        "cot_session": cot_session
                    })

                except Exception as e:
                    logger.exception(f"LLM processing failed: {e}")
                    error_msg = f"An error occurred: {e}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # Reset processing flag and rerun to display the final assistant message
        st.session_state.is_processing = False
        st.rerun()


def process_document_pipeline(uploaded_file) -> bool:
    """
    Process a single uploaded file through the complete pipeline.
    
    Args:
        uploaded_file: Single Streamlit UploadedFile object
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting pipeline for: {uploaded_file.name}")
    
    try:
        # Save file temporarily
        temp_file_path = Path("temp") / uploaded_file.name
        temp_file_path.parent.mkdir(exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Step 1: Process document using exact method signature
        with st.status("📖 Step 1/5: Processing document...", expanded=True) as status:
            document = st.session_state.document_processor.process(str(temp_file_path))
            if not document:
                st.error("❌ Document processing failed!")
                return False
            
            # Validate document content - check for meaningful text
            content_stripped = document.content.strip()
            if not content_stripped or len(content_stripped) < 10:
                st.error("❌ Document appears to be empty or contains no readable text!")
                logger.warning(f"Document has insufficient content: {len(content_stripped)} meaningful characters")
                return False
            
            # Check if content is mostly whitespace
            non_whitespace_chars = len([c for c in document.content if not c.isspace()])
            total_chars = len(document.content)
            if total_chars > 0 and (non_whitespace_chars / total_chars) < 0.1:
                st.error("❌ Document contains mostly whitespace - may be corrupted or unsupported format!")
                logger.warning(f"Document is {((total_chars - non_whitespace_chars) / total_chars * 100):.1f}% whitespace")
                return False
            
            status.update(label="✅ Document processed", state="complete")
            logger.info(f"Document processed: {len(document.content)} chars, {non_whitespace_chars} non-whitespace")
        
        # Step 2: Create chunks using exact method signature
        with st.status("✂️ Step 2/5: Creating chunks...", expanded=True) as status:
            chunks = st.session_state.text_chunker.create_chunks(document)
            if not chunks or len(chunks) == 0:
                st.error("❌ Text chunking failed - no meaningful chunks could be created!")
                logger.error(f"TextChunker returned {len(chunks) if chunks else 0} chunks")
                return False
            
            status.update(label="✅ Chunks created", state="complete")
            logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings using exact method signature
        with st.status("🧠 Step 3/5: Creating embeddings...", expanded=True) as status:
            embeddings = st.session_state.embedder.create_embeddings(chunks)
            if not embeddings:
                st.error("❌ Embedding creation failed!")
                return False
            
            status.update(label="✅ Embeddings created", state="complete")
            logger.info(f"Created {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database using exact method signature
        with st.status("💾 Step 4/5: Storing in vector database...", expanded=True) as status:
            collection_name = f"{st.session_state.session_id}_{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]}"
            
            success = st.session_state.vector_store.add_chunks(collection_name, chunks, embeddings)
            if not success:
                st.error("❌ Vector storage failed!")
                return False
            
            status.update(label="✅ Stored in vector database", state="complete")
            logger.info(f"Stored in collection: {collection_name}")
        
        # Step 5: Save to session using exact method signature
        with st.status("📝 Step 5/5: Saving to session...", expanded=True) as status:
            file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            
            session_data = SessionData(
                file_hash=file_hash,
                file_name=uploaded_file.name,
                file_path=str(temp_file_path),
                processed_at=datetime.now().isoformat(),
                vector_collection_name=collection_name,
                document_metadata=document.metadata,
                file_size_mb=len(uploaded_file.getbuffer()) / (1024 * 1024),
                content_preview=document.content[:200]
            )
            
            session_success = st.session_state.session_manager.add_document_to_session(
                st.session_state.session_id, session_data
            )
            
            if not session_success:
                st.error("❌ Session save failed!")
                return False
            
            # Update local list
            st.session_state.uploaded_documents.append(session_data)
            
            status.update(label="✅ Processing completed!", state="complete")
            logger.success(f"Pipeline completed: {uploaded_file.name}")
        
        # Clean up
        temp_file_path.unlink(missing_ok=True)
        return True
        
    except Exception as e:
        logger.exception(f"Pipeline failed for {uploaded_file.name}: {e}")
        st.error(f"❌ Processing error: {str(e)}")
        
        if 'temp_file_path' in locals() and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        
        return False


def handle_file_uploads(uploaded_files):
    """Handle multiple file uploads through the processing pipeline."""
    if not uploaded_files:
        return
    
    logger.info(f"Processing {len(uploaded_files)} files")
    
    # Set processing state
    st.session_state.is_processing = True
    
    successful = 0
    failed = 0
    
    try:
        for uploaded_file in uploaded_files:
            st.write(f"### 📁 Processing: {uploaded_file.name}")
            
            # Check for duplicates
            file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            existing = any(doc.file_hash == file_hash for doc in st.session_state.uploaded_documents)
            
            if existing:
                st.warning(f"Already processed: {uploaded_file.name}")
                continue
            
            # Process file
            if process_document_pipeline(uploaded_file):
                successful += 1
                st.success(f"✅ {uploaded_file.name} processed!")
            else:
                failed += 1
                st.error(f"❌ Failed: {uploaded_file.name}")
        
        # Show summary
        if successful > 0 or failed > 0:
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Successful", successful)
            with col2:
                st.metric("❌ Failed", failed)
    
    except Exception as e:
        logger.exception(f"Upload handling failed: {e}")
        st.error(f"❌ Upload failed: {str(e)}")
    
    finally:
        st.session_state.is_processing = False
        st.rerun()


def handle_chat_submission(prompt: str):
    """Handle user chat submission with immediate feedback."""
    logger.info(f"Chat submission: {prompt[:50]}...")
    
    # Add user message immediately
    st.session_state.chat_history.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Set processing state and rerun
    st.session_state.is_processing = True
    st.rerun()


def process_llm_response():
    """Process LLM response asynchronously with spinner."""
    if not st.session_state.is_processing:
        return
    
    # Get last user message
    last_message = None
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "user":
            last_message = message
            break
    
    if not last_message:
        st.session_state.is_processing = False
        return
    
    # Process with spinner as required
    with st.spinner("🤔 Assistant thinking..."):
        try:
            # Execute LLM agent with CoT using exact method and asyncio.run()
            cot_session = asyncio.run(
                st.session_state.llm_agent.execute_with_cot(
                    last_message["content"],
                    st.session_state.session_id,
                    chat_history=last_message
                )
            )
            
            # Extract final_answer as required
            final_answer = cot_session.final_answer
            
            # Add response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "timestamp": datetime.now().isoformat(),
                "cot_session": cot_session,
                "success": cot_session.success
            })
            
            logger.info(f"LLM response: {len(final_answer)} chars, success={cot_session.success}")
            
        except Exception as e:
            logger.exception(f"LLM processing failed: {e}")
            
            # Add error message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error processing request: {str(e)}\\n\\nPlease try again.",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
    
    # Reset state and rerun
    st.session_state.is_processing = False
    st.rerun()


def display_cot_visualization(cot_session: CoTSession):
    """Displays CoT process. Assumes it's already inside an expander."""
    if not cot_session:
        return
    
    try:
        # Overview metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Reasoning Steps", len(cot_session.reasoning_steps))
        col2.metric("Tool Executions", len(cot_session.tool_executions))
        col3.metric("Execution Time", f"{cot_session.total_execution_time:.2f}s")
        st.markdown("---")

        # Complexity Analysis
        if hasattr(cot_session, 'complexity_analysis') and cot_session.complexity_analysis:
            st.subheader("📊 Query Complexity Analysis")
            complexity = cot_session.complexity_analysis
            st.write(f"**Level:** `{complexity.complexity.value}` | **Confidence:** {complexity.confidence:.1%}")
            st.caption(f"Reasoning: {complexity.reasoning}")

        # Tool Executions
        if cot_session.tool_executions:
            st.subheader("🛠️ Tool Executions")
            for exec in cot_session.tool_executions:
                success_icon = "✅" if exec.success else "❌"
                with st.container(border=True):
                    st.markdown(f"{success_icon} **Tool:** `{exec.tool_name}` ({exec.execution_time:.2f}s)")
                    st.caption("Arguments:")
                    st.json(exec.arguments)
                    st.caption("Result:")
                    result_data = exec.result
                    if isinstance(result_data, BaseModel):
                        st.json(result_data.model_dump())
                    elif isinstance(result_data, dict):
                        st.json(result_data)
                    else:
                        st.code(str(result_data), language=None)
    except Exception as e:
        logger.exception(f"CoT visualization error: {e}")
        st.error(f"Could not display full reasoning: {e}")


def remove_document_from_session(file_hash: str):
    """Remove document from session and all associated data."""
    logger.info(f"Removing document: {file_hash[:16]}...")
    
    try:
        # Find document
        doc_to_remove = None
        for doc in st.session_state.uploaded_documents:
            if doc.file_hash == file_hash:
                doc_to_remove = doc
                break
        
        if not doc_to_remove:
            st.error("Document not found!")
            return
        
        # Remove using exact method signatures
        session_success = st.session_state.session_manager.remove_document_from_session(
            st.session_state.session_id, file_hash
        )
        
        vector_success = st.session_state.vector_store.delete_collection(
            doc_to_remove.vector_collection_name
        )
        
        # Remove from local list
        st.session_state.uploaded_documents = [
            doc for doc in st.session_state.uploaded_documents 
            if doc.file_hash != file_hash
        ]
        
        if session_success and vector_success:
            st.success(f"✅ {doc_to_remove.file_name} removed!")
            logger.success(f"Document removed: {doc_to_remove.file_name}")
        else:
            st.warning("⚠️ Document partially removed")
        
        st.rerun()
        
    except Exception as e:
        logger.exception(f"Remove document error: {e}")
        st.error(f"❌ Remove failed: {str(e)}")


def clear_all_session_data():
    """Clear all session data."""
    logger.info("Clearing all session data")
    
    try:
        # Clear vector collections
        for doc in st.session_state.uploaded_documents:
            try:
                st.session_state.vector_store.delete_collection(doc.vector_collection_name)
            except Exception as e:
                logger.warning(f"Collection delete failed: {e}")
        
        # Clear session manager
        st.session_state.session_manager.clear_session(st.session_state.session_id)
        
        # Clear local state
        st.session_state.uploaded_documents = []
        st.session_state.chat_history = []
        st.session_state.is_processing = False
        
        st.success("✅ All data cleared!")
        logger.success("Session cleared")
        st.rerun()
        
    except Exception as e:
        logger.exception(f"Clear session error: {e}")
        st.error(f"❌ Clear failed: {str(e)}")


def main():
    """Main application function orchestrating the Streamlit interface."""
    # Page config
    st.set_page_config(
        page_title="DataNeuron Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-ready { background-color: #28a745; }
        .status-processing { background-color: #ffc107; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🧠 DataNeuron Assistant")
    st.markdown("*AI-powered document analysis with Chain of Thought reasoning*")
    
    # Status indicator
    if st.session_state.is_processing:
        st.markdown('<span class="status-indicator status-processing"></span>**Processing...**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-ready"></span>**Ready**', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main tabs
    chat_tab, upload_tab = st.tabs(["💬 Chat", "📁 Upload Documents"])
    
    with upload_tab:
        st.header("📁 Document Upload & Processing")
        st.markdown("Upload PDF, DOCX, or TXT documents:")
        
        uploaded_files = st.file_uploader(
            "Select documents:",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            disabled=st.session_state.is_processing
        )
        
        if st.button(
            "📤 Upload and Process", 
            type="primary",
            disabled=st.session_state.is_processing or not uploaded_files
        ):
            handle_file_uploads(uploaded_files)
        
        if st.session_state.is_processing:
            st.info("🔄 Processing in progress...")
    
    with chat_tab:
        # Handle LLM response if processing
        if st.session_state.is_processing:
            process_llm_response()
        
        # Render chat
        render_chat_interface()

        

        
        # Help messages
        if not st.session_state.uploaded_documents:
            st.info("""
            💡 **Getting Started:**
            1. Go to "📁 Upload Documents" tab
            2. Upload PDF, DOCX, or TXT documents
            3. Return here to ask questions
            4. AI will analyze and provide detailed answers
            """)
        elif not st.session_state.chat_history:
            st.info(f"""
            💡 **Ready to Chat!**
            
            {len(st.session_state.uploaded_documents)} document(s) loaded.
            
            Example questions:
            - "What are the main topics?"
            - "Summarize key findings"
            - "Compare different approaches"
            """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Critical error: {e}")
        st.error(f"""
        ❌ **Critical Application Error**
        
        Error: {str(e)}
        
        Check:
        1. Required packages installed
        2. API keys in .env file
        3. Backend components initialized
        4. Application logs
        """)
        st.stop()