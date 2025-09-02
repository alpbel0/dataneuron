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
from contextlib import contextmanager
import sys
from config.settings import UPLOADS_DIR

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import backend components with exact names from requirements
from core.document_processor import DocumentProcessor, Document
from core.chunker import TextChunker, Chunk
from core.embedder import Embedder
from core.cloud_vector_store import VectorStore
from core.session_manager import SessionManager, SessionData
from core.llm_agent import LLMAgent, CoTSession
from utils.logger import logger
from utils.logo_manager import LogoManager
from pydantic import BaseModel
from core.tool_manager import ToolManager

@contextmanager
def processing_context():
    """Context manager for safe processing state management."""
    st.session_state.is_processing = True
    try:
        yield
    finally:
        st.session_state.is_processing = False

def get_session_documents_safely(session_id: str) -> List[Any]:
    """Safe wrapper for getting session documents with detailed debugging."""
    try:
        logger.info(f"Fetching documents for session: {session_id}")
        documents = st.session_state.session_manager.get_session_documents(session_id)
        logger.info(f"Retrieved {len(documents)} documents for session {session_id}")
        if not documents:
            logger.warning(f"No documents found for session {session_id}. Checking all sessions...")
            all_sessions = st.session_state.session_manager.get_all_sessions()
            logger.info(f"Available sessions: {all_sessions}")
        return documents
    except Exception as e:
        logger.error(f"Could not fetch documents for session {session_id}: {e}")
        logger.exception("Full exception details:")
        return []

def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file before processing."""
    # Size check (max 50MB)
    max_size = 50 * 1024 * 1024
    file_size = len(uploaded_file.getbuffer())
    if file_size > max_size:
        return False, f"File too large: {file_size/(1024*1024):.1f}MB > 50MB"
    
    # Type check
    allowed_types = ['.pdf', '.docx', '.txt']
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in allowed_types:
        return False, f"Unsupported file type: {file_ext}"
    
    # Name validation
    if not uploaded_file.name or len(uploaded_file.name.strip()) == 0:
        return False, "Invalid file name"
    
    return True, "Valid"

def save_uploaded_file_safely(uploaded_file, temp_path: Path, chunk_size: int = 8192) -> bool:
    """Save uploaded file with memory optimization."""
    try:
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.exception(f"File save failed: {e}")
        return False

def safe_cleanup_temp_file(file_path: Path):
    """Safely cleanup temporary files."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not cleanup temp file {file_path}: {e}")

def initialize_session_state():
    """Initialize all required Streamlit session state variables."""
    logger.info("Initializing Streamlit session state")
    
    # Generate unique session ID with better debugging
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated NEW session ID: {st.session_state.session_id}")
    else:
        logger.info(f"Using EXISTING session ID: {st.session_state.session_id}")
    
    # Initialize state variables
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
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

        if 'tool_manager' not in st.session_state:
            st.session_state.tool_manager = ToolManager(session_manager=st.session_state.session_manager)
            logger.info("ToolManager initialized with injected SessionManager")
        
        if 'llm_agent' not in st.session_state:
            st.session_state.llm_agent = LLMAgent(
                tool_manager=st.session_state.tool_manager,
                session_manager=st.session_state.session_manager
            )
            logger.info("LLMAgent initialized with shared managers")
        
        if 'logo_manager' not in st.session_state:
            st.session_state.logo_manager = LogoManager()
            logger.info("LogoManager initialized")
            
    except Exception as e:
        logger.exception(f"Backend initialization failed: {e}")
        st.error(f"Backend initialization failed: {str(e)}")
        st.stop()
    
    logger.success("Session state initialization completed")

def render_sidebar():
    """Fixed sidebar with safe document fetching."""
    with st.sidebar:
        st.header("📄 Documents")
        
        # Get documents safely
        uploaded_documents = get_session_documents_safely(st.session_state.session_id)
        
        # Display uploaded documents
        if uploaded_documents:
            st.write(f"**{len(uploaded_documents)} documents loaded:**")
            
            for doc in uploaded_documents:
                with st.expander(f"{doc.file_name} ({doc.file_size_mb:.1f}MB)"):
                    st.write(f"**Processed:** {doc.processed_at[:19]}")
                    st.write(f"**Collection:** {doc.vector_collection_name}")
                    if doc.content_preview:
                        st.write(f"**Preview:** {doc.content_preview[:100]}...")
                    
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
                ("Tool Manager", 'tool_manager'),
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
            
            # Debug session state
            try:
                all_sessions = st.session_state.session_manager.get_all_sessions()
                st.write(f"**Total sessions in system:** {len(all_sessions)}")
                if all_sessions:
                    st.write("**All sessions:**")
                    for sess_id, doc_count in all_sessions.items():
                        current_marker = "👈 **CURRENT**" if sess_id == st.session_state.session_id else ""
                        st.write(f"  - {sess_id}: {doc_count} docs {current_marker}")
            except Exception as e:
                st.write(f"**Debug error:** {e}")
        
        # Session Management
        st.header("Session Management")
        
        # Session recovery for Streamlit Cloud issues
        if st.button("🔄 Refresh Session", help="Reload documents from storage", disabled=st.session_state.is_processing):
            try:
                # Force reload session manager state
                st.session_state.session_manager._load_state()
                st.success("Session refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")
        
        # Session switching for debugging
        all_sessions = st.session_state.session_manager.get_all_sessions()
        if len(all_sessions) > 1:
            st.write("**Switch to existing session:**")
            session_options = [f"{sess_id} ({doc_count} docs)" for sess_id, doc_count in all_sessions.items()]
            selected_session = st.selectbox(
                "Available sessions:",
                session_options,
                disabled=st.session_state.is_processing
            )
            if st.button("Switch Session", disabled=st.session_state.is_processing):
                new_session_id = selected_session.split(" ")[0]
                st.session_state.session_id = new_session_id
                st.session_state.chat_history = []  # Clear chat for new session
                logger.info(f"Switched to session: {new_session_id}")
                st.success(f"Switched to session: {new_session_id}")
                st.rerun()
        
        if st.button(
            "Clear All Data", 
            type="secondary",
            disabled=st.session_state.is_processing
        ):
            clear_all_session_data()

def render_chat_interface():
    """Fixed chat interface with proper state management."""
    st.header("💬 Chat Interface")

    # Display chat history
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message["role"]):
            if message.get("is_clarification_request", False):
                st.markdown("🤔 **I need clarification:**")
                st.markdown(message["content"])
                st.info("💡 Please provide more details to help me assist you better.")
            else:
                st.markdown(message["content"])
            
            if "cot_session" in message:
                with st.expander("🧠 View Agent's Reasoning"):
                    display_cot_visualization(message["cot_session"])

    # ============================================================================
    # HEDEFLENMİŞ DOKÜMAN SORGULAMA - DOKÜMAN SEÇİM BİLEŞENİ
    # ============================================================================
    
    # Yüklü dokümanları al
    uploaded_documents = get_session_documents_safely(st.session_state.session_id)
    
    if uploaded_documents:
        # Doküman isimlerini listele
        doc_filenames = [doc.file_name for doc in uploaded_documents]
        
        st.markdown("### 📋 Doküman Seçimi")
        st.markdown("Sorgunuz için hangi dokümanları analiz etmek istiyorsunuz?")
        
        # Multiselect bileşeni
        selected_files = st.multiselect(
            "Sorgunuz için doküman seçin (birden çok seçim yapabilirsiniz):",
            options=doc_filenames,
            default=st.session_state.get('last_selected_files', doc_filenames),  # Varsayılan: tüm dosyalar
            key="document_selector",
            help="Seçili dokümanlar üzerinde analiz yapılacak. Hiç seçim yapmazsanız size sorulacak.",
            disabled=st.session_state.is_processing
        )
        
        # Seçimi session state'e kaydet
        st.session_state.last_selected_files = selected_files
        
        # Seçim durumu gösterimi
        if selected_files:
            if len(selected_files) == len(doc_filenames):
                st.success(f"✅ Tüm dokümanlar seçili ({len(selected_files)} doküman)")
            else:
                st.info(f"📑 {len(selected_files)} doküman seçili: {', '.join(selected_files)}")
        else:
            st.warning("⚠️ Hiç doküman seçilmedi. Sorguyu gönderdiğinizde size doküman seçimi sorulacak.")
        
        st.markdown("---")
    
    else:
        selected_files = []
        st.info("📄 Henüz yüklenmiş doküman bulunmuyor. Önce 'Upload Documents' sekmesinden doküman yükleyin.")
    
    # ============================================================================
    # HEDEFLENMİŞ DOKÜMAN SORGULAMA BİLEŞENİ SONU
    # ============================================================================
    
    # ============================================================================
    # AKILLI İNTERNET ARAMA ENTEGRASYONUc
    # ============================================================================
    
    st.markdown("### 🌐 İnternet Arama")
    
    # Web search toggle
    web_search_enabled = st.toggle(
        "İnternette Ara",
        value=st.session_state.get('web_search_enabled', False),
        key="web_search_toggle",
        help="Etkinleştirildiğinde, ajan yüklü dokümanlarınıza ek olarak internetten güncel bilgi arayabilir.",
        disabled=st.session_state.is_processing
    )
    
    # Update session state
    st.session_state.web_search_enabled = web_search_enabled
    
    # ### YENİ VE DOĞRU DURUM GÖSTERİMİ ###
    if web_search_enabled:
        st.success("✅ İnternet araması etkin - Ajan hem dokümanlarınızı hem de güncel web bilgilerini kullanabilir.")
        st.info("💡 **İpucu:** İnternet araması özellikle güncel haberler, şirket bilgileri ve yeni gelişmeler gibi konularda faydalıdır.")
    else:
        st.info("📄 Sadece yüklü dokümanlar - Ajan yalnızca bu oturumdaki dokümanları kullanacak.")
        
    st.markdown("---")
    
    # ============================================================================
    # AKILLI İNTERNET ARAMA ENTEGRASYONU SONU  
    # ============================================================================

    # Handle user input - FIXED VERSION (no infinite loop)
    # Dynamic chat input placeholder based on capabilities
    if web_search_enabled:
        chat_placeholder = "Ask about your documents or search the web for current information..."
    else:
        chat_placeholder = "Ask about your documents..."
    
    if prompt := st.chat_input(chat_placeholder, key="chat_widget", disabled=st.session_state.is_processing):
        # KULLANICI MESAJINI ANINDA CHAT GEÇMİŞİNE EKLE
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # --- YENİ KAPSAMLI ZENGİNLEŞTİRME MANTIĞI ---
        final_prompt_for_agent = prompt
        
        # Adım 1: Web araması açık mı? Açıksa, web arama komutu ekle.
        if web_search_enabled:
            web_prefix = "Use the web search tool to answer the following: "
            final_prompt_for_agent = f"{web_prefix}{prompt}"
            logger.info("Web search is enabled. Prepended web search command to the prompt.")

        # Adım 2: Dosya seçili mi? Seçiliyse, dosya bilgisini de ekle.
        if selected_files:
            file_names_str = ", ".join([f"'{f}'" for f in selected_files])
            # ÖNEMLİ: Web araması zaten eklenmiş olabilir, bu yüzden final_prompt_for_agent'ı kullan.
            file_prefix = f"Using the following document(s) as primary context: {file_names_str}. "
            final_prompt_for_agent = f"{file_prefix}{final_prompt_for_agent}"
            logger.info("Selected files found. Prepended document context to the prompt.")

        logger.info(f"Final prompt for agent: {final_prompt_for_agent}")
        # --- YENİ MANTIK SONU ---
        
        # Process LLM response immediately with context manager
        with processing_context():
            with st.chat_message("assistant"):
                # Enhanced spinner with dynamic progress tracking
                spinner_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                # Track current progress for smooth transitions
                if 'current_progress_percent' not in st.session_state:
                    st.session_state.current_progress_percent = 0
                
                def update_progress(current_step: int, total_steps: int, tool_name: str = "", message: str = "Processing...", is_executing: bool = False):
                    """Update progress bar and message dynamically."""
                    progress_percent = (current_step / max(total_steps, 1)) * 100
                    
                    # Tool name display mappings
                    tool_display_names = {
                        "read_full_document": "📄 Reading Document",
                        "search_in_document": "🔍 Searching Content", 
                        "summarize_document": "📝 Creating Summary",
                        "web_search": "🌐 Web Search",
                        "compare_documents": "⚖️ Comparing Docs",
                        "risk_assessment": "⚠️ Risk Analysis",
                        "synthesize_results": "🔬 Synthesizing"
                    }
                    
                    display_name = tool_display_names.get(tool_name, f"⚡ {tool_name.replace('_', ' ').title()}")
                    
                    # Choose progress bar class based on state
                    progress_class = "progress-bar-animated progress-pulse" if is_executing else "progress-bar"
                    
                    spinner_placeholder.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="display: inline-block; animation: pulse 1s infinite;">🤖</div>
                        <p><strong>{display_name}</strong></p>
                        <p style="color: #666; font-size: 0.9em;">{message}</p>
                        <div class="progress-container">
                            <div class="{progress_class}" style="width: {progress_percent}%;"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #888;">
                            Step {current_step} of {total_steps} ({progress_percent:.0f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update session state
                    st.session_state.current_progress_percent = progress_percent
                
                def smooth_progress_to(target_step: int, total_steps: int, tool_name: str = "", message: str = "Processing...", is_executing: bool = False, duration: float = 1.0):
                    """Smoothly animate progress from current to target step."""
                    import time
                    
                    current_percent = st.session_state.current_progress_percent
                    target_percent = (target_step / max(total_steps, 1)) * 100
                    
                    # Calculate increment steps
                    increment_steps = 20  # 20 small steps for smooth animation
                    increment_size = (target_percent - current_percent) / increment_steps
                    step_duration = duration / increment_steps
                    
                    for i in range(increment_steps + 1):
                        intermediate_percent = current_percent + (increment_size * i)
                        intermediate_step = (intermediate_percent / 100) * total_steps
                        
                        # Update display
                        tool_display_names = {
                            "read_full_document": "📄 Reading Document",
                            "search_in_document": "🔍 Searching Content", 
                            "summarize_document": "📝 Creating Summary",
                            "web_search": "🌐 Web Search",
                            "compare_documents": "⚖️ Comparing Docs",
                            "risk_assessment": "⚠️ Risk Analysis",
                            "synthesize_results": "🔬 Synthesizing"
                        }
                        
                        display_name = tool_display_names.get(tool_name, f"⚡ {tool_name.replace('_', ' ').title()}")
                        progress_class = "progress-bar-animated progress-pulse" if is_executing else "progress-bar"
                        
                        spinner_placeholder.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div style="display: inline-block; animation: pulse 1s infinite;">🤖</div>
                            <p><strong>{display_name}</strong></p>
                            <p style="color: #666; font-size: 0.9em;">{message}</p>
                            <div class="progress-container">
                                <div class="{progress_class}" style="width: {intermediate_percent}%;"></div>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.8em; color: #888;">
                                Step {intermediate_step:.1f} of {total_steps} ({intermediate_percent:.0f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(step_duration)
                    
                    # Update session state to final value
                    st.session_state.current_progress_percent = target_percent
                
                # Initial progress - reset progress counter
                st.session_state.current_progress_percent = 0
                update_progress(0, 10, "", "Initializing agent...")
                
                try:
                    # ============================================================================
                    # SOHBETEDİLEN HAFIZA - SON 6 MESAJİ AL (3 çift: kullanıcı + asistan)
                    # ============================================================================
                    
                    # Get recent chat history for agent (excluding current message)
                    full_history = st.session_state.chat_history[:-1]  # Son mesaj hariç
                    
                    # Son 20 mesajı al (daha uzun hafıza penceresi için)
                    # Bu şekilde maksimum 10 soru-cevap çifti hafızada tutulur
                    max_history_length = 20
                    if len(full_history) > max_history_length:
                        history_for_agent = full_history[-max_history_length:]
                        logger.info(f"🧠 Using last {len(history_for_agent)} messages from chat history for conversational memory")
                    else:
                        history_for_agent = full_history
                        if history_for_agent:
                            logger.info(f"🧠 Using all {len(history_for_agent)} messages from chat history for conversational memory")
                    
                    # Smooth progress tracking
                    import time
                    
                    smooth_progress_to(1, 10, "", "Preparing execution...", duration=0.8)
                    smooth_progress_to(2, 10, "", "Analyzing query...", is_executing=True, duration=1.0)
                    
                    # Execute agent with faster initial progress
                    smooth_progress_to(3, 10, "", "Executing tools...", is_executing=True, duration=0.8)
                    
                    # Continuous progress during LLM execution
                    import threading
                    import time
                    
                    # Progress counter for continuous animation
                    progress_counter = {"value": 5.0}
                    stop_progress = {"flag": False}
                    
                    def continuous_progress_update():
                        """Background progress updater"""
                        while not stop_progress["flag"] and progress_counter["value"] < 9.0:
                            progress_counter["value"] += 0.1
                            if progress_counter["value"] > 9.0:
                                progress_counter["value"] = 9.0
                            
                            # Update display
                            spinner_placeholder.markdown(f"""
                            <div style="text-align: center; padding: 20px;">
                                <div style="display: inline-block; animation: pulse 1s infinite;">🤖</div>
                                <p><strong>🧠 Agent Processing</strong></p>
                                <p style="color: #666; font-size: 0.9em;">Running analysis tools...</p>
                                <div class="progress-container">
                                    <div class="progress-bar-flowing" style="width: {min(progress_counter['value']/10*100, 90)}%;"></div>
                                </div>
                                <div style="margin-top: 8px; font-size: 0.8em; color: #888;">
                                    Step {progress_counter['value']:.1f} of 10 ({progress_counter['value']/10*100:.0f}%) - Agent working...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            time.sleep(1)
                    
                    # Start background progress
                    progress_thread = threading.Thread(target=continuous_progress_update, daemon=True)
                    progress_thread.start()
                    
                    try:
                        cot_session = asyncio.run(
                            st.session_state.llm_agent.execute_with_cot(
                                query=final_prompt_for_agent,
                                session_id=st.session_state.session_id,
                                chat_history=history_for_agent,
                                selected_filenames=selected_files,  # YENİ: Hedeflenmiş doküman sorgulama
                                allow_web_search=web_search_enabled  # YENİ: Akıllı İnternet Arama
                            )
                        )
                    finally:
                        # Stop background progress
                        stop_progress["flag"] = True
                    
                    # Dynamic progress based on actual tools used
                    if cot_session.tool_executions:
                        total_tools = len(cot_session.tool_executions)
                        base_progress = 6
                        tool_progress_range = 4  # From step 6 to 10
                        
                        for i, tool_exec in enumerate(cot_session.tool_executions):
                            current_progress_step = base_progress + ((i + 1) / total_tools) * tool_progress_range
                            
                            # Show tool name and what it's doing
                            tool_actions = {
                                "read_full_document": "Reading document content...",
                                "search_in_document": "Searching document...",
                                "summarize_document": "Creating summary...",
                                "web_search": "Searching web...",
                                "compare_documents": "Comparing documents...",
                                "risk_assessment": "Analyzing risks...",
                                "synthesize_results": "Synthesizing results..."
                            }
                            
                            action_message = tool_actions.get(tool_exec.tool_name, f"Running {tool_exec.tool_name}...")
                            tool_status = "✅ Completed" if tool_exec.success else "❌ Failed"
                            
                            # Smooth progress to this tool's completion
                            smooth_progress_to(int(current_progress_step), 10, tool_exec.tool_name, action_message, is_executing=True, duration=1.5)
                            
                            # Quick completion status
                            update_progress(int(current_progress_step), 10, tool_exec.tool_name, tool_status)
                            time.sleep(0.3)
                    
                    smooth_progress_to(10, 10, "", "✨ Response ready!", duration=0.5)
                    
                    # Clear spinner
                    spinner_placeholder.empty()
                    
                    # Check for clarification request - Çifte kontrol sistemi
                    is_asking_clarification = False
                    clarification_question = None
                    
                    # Birinci kontrol: CoTSession metadata'dan kontrol et
                    if cot_session.metadata.get("clarification_requested", False):
                        is_asking_clarification = True
                        clarification_question = cot_session.metadata.get("clarification_question", "Could you please clarify?")
                        logger.info("🔍 Clarification detected via CoTSession metadata")
                    
                    # İkinci kontrol: Son araç çağrısından kontrol et (fallback)
                    elif cot_session.tool_executions:
                        last_tool_execution = cot_session.tool_executions[-1]
                        if last_tool_execution.tool_name == "ask_user_for_clarification":
                            is_asking_clarification = True
                            clarification_question = last_tool_execution.arguments.get("question", "Could you please clarify?")
                            logger.info("🔍 Clarification detected via last tool execution")
                    
                    # Determine response content
                    if is_asking_clarification:
                        response_content = clarification_question
                        response_metadata = {"is_clarification_request": True}
                    else:
                        response_content = cot_session.final_answer
                        response_metadata = {}
                    
                    # Display response immediately
                    if response_metadata.get("is_clarification_request"):
                        st.markdown("🤔 **I need clarification:**")
                        st.markdown(response_content)
                        
                        # ============================================================================
                        # AKILLI YÖNLENDİRME MANTĞI - CLARIFICATION İÇİN ÖNERİLER
                        # ============================================================================
                        
                        # Dokuman durumunu kontrol et
                        has_documents = len(uploaded_documents) > 0
                        web_search_available = not web_search_enabled
                        
                        # Akıllı öneriler oluştur
                        suggestions = []
                        
                        if not has_documents:
                            suggestions.append("📁 **Doküman yükleyin:** 'Upload Documents' sekmesinden ilgili dökümanları yükleyerek daha ayrıntılı analiz alabilirsiniz.")
                        
                        if web_search_available and has_documents:
                            suggestions.append("🌐 **İnternet aramasını etkinleştirin:** Yukarıdaki 'İnternette Ara' seçeneğini açarak güncel bilgilere erişebilirsiniz.")
                        
                        if not has_documents and web_search_available:
                            suggestions.append("🌐 **İnternet aramasını deneyin:** 'İnternette Ara' seçeneğini etkinleştirerek güncel web bilgilerine ulaşabilirsiniz.")
                        
                        if has_documents and len(selected_files) == 0:
                            suggestions.append("🎯 **Doküman seçin:** Yukarıdaki listeden analiz edilecek spesifik dokümanları seçin.")
                        
                        if has_documents and web_search_enabled:
                            suggestions.append("💬 **Sorunuzu detaylandırın:** Hangi spesifik bilgiyi aradığınızı daha açık belirtin.")
                        
                        # Önerileri göster
                        if suggestions:
                            st.warning("**💡 Bu öneriler size yardımcı olabilir:**")
                            for suggestion in suggestions:
                                st.markdown(f"• {suggestion}")
                        else:
                            st.info("💡 Please provide more details to help me assist you better.")
                        
                        # ============================================================================
                        # AKILLI YÖNLENDİRME SONU
                        # ============================================================================
                    else:
                        st.markdown(response_content)
                    
                    # Show reasoning if available
                    with st.expander("🧠 View Agent's Reasoning"):
                        display_cot_visualization(cot_session)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "cot_session": cot_session,
                        **response_metadata
                    })
                    
                    # Ekranı yenile ki hem soru hem cevap görünsün
                    st.rerun()
                        
                except Exception as e:
                    logger.exception(f"LLM processing failed: {e}")
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

def process_document_pipeline(uploaded_file) -> bool:
    """Document processing pipeline with persistent file storage."""
    logger.info(f"Starting pipeline for: {uploaded_file.name}")
    
    # Validate file first
    is_valid, validation_msg = validate_uploaded_file(uploaded_file)
    if not is_valid:
        st.error(f"❌ {validation_msg}")
        return False
    
    persistent_file_path = None
    try:
        # Create session-specific upload directory
        session_upload_dir = UPLOADS_DIR / st.session_state.session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file to persistent location
        persistent_file_path = session_upload_dir / uploaded_file.name
        
        if not save_uploaded_file_safely(uploaded_file, persistent_file_path):
            st.error("❌ Failed to save uploaded file!")
            return False
        
        # Step 1: Process document
        with st.status("📖 Step 1/5: Processing document...", expanded=True) as status:
            document = st.session_state.document_processor.process(str(persistent_file_path))
            if not document:
                st.error("❌ Document processing failed!")
                return False
            
            # Enhanced content validation - UPDATED FOR NEW FORMAT
            # Combine all page texts into a single string for validation
            if isinstance(document.content, list):
                # New format: List[{"page_number": int, "text": str}]
                full_text_content = "\n".join(page.get("text", "") for page in document.content)
            else:
                # Legacy format: single string (backward compatibility)
                full_text_content = document.content
            
            content_stripped = full_text_content.strip()
            if not content_stripped or len(content_stripped) < 10:
                st.error("❌ Document appears to be empty or contains no readable text!")
                logger.warning(f"Document has insufficient content: {len(content_stripped)} meaningful characters")
                return False
            
            # Check content quality
            non_whitespace_chars = len([c for c in full_text_content if not c.isspace()])
            total_chars = len(full_text_content)
            if total_chars > 0 and (non_whitespace_chars / total_chars) < 0.1:
                st.error("❌ Document contains mostly whitespace - may be corrupted!")
                logger.warning(f"Document is {((total_chars - non_whitespace_chars) / total_chars * 100):.1f}% whitespace")
                return False
            
            status.update(label="✅ Document processed", state="complete")
            logger.info(f"Document processed: {len(full_text_content)} chars, {non_whitespace_chars} non-whitespace")
            
            # Additional logging for new format
            if isinstance(document.content, list):
                logger.info(f"Document structure: {len(document.content)} pages with page numbers")
            else:
                logger.info("Document structure: legacy single-text format")
        
        # Step 2: Create chunks
        with st.status("✂️ Step 2/5: Creating chunks...", expanded=True) as status:
            chunks = st.session_state.text_chunker.create_chunks(document)
            if not chunks or len(chunks) == 0:
                st.error("❌ Text chunking failed - no meaningful chunks could be created!")
                logger.error(f"TextChunker returned {len(chunks) if chunks else 0} chunks")
                return False
            
            status.update(label="✅ Chunks created", state="complete")
            logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings
        with st.status("🧠 Step 3/5: Creating embeddings...", expanded=True) as status:
            embeddings = st.session_state.embedder.create_embeddings(chunks)
            if not embeddings:
                st.error("❌ Embedding creation failed!")
                return False
            
            status.update(label="✅ Embeddings created", state="complete")
            logger.info(f"Created {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database
        with st.status("💾 Step 4/5: Storing in vector database...", expanded=True) as status:
            collection_name = f"{st.session_state.session_id}_{hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]}"
            
            success = st.session_state.vector_store.add_chunks(collection_name, chunks, embeddings)
            if not success:
                st.error("❌ Vector storage failed!")
                return False
            
            status.update(label="✅ Stored in vector database", state="complete")
            logger.info(f"Stored in collection: {collection_name}")
        
        # Step 5: Save to session
        with st.status("📝 Step 5/5: Saving to session...", expanded=True) as status:
            file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            
            session_data = SessionData(
                file_hash=file_hash,
                file_name=uploaded_file.name,
                file_path=str(persistent_file_path),
                processed_at=datetime.now().isoformat(),
                vector_collection_name=collection_name,
                document_metadata=document.metadata,
                file_size_mb=len(uploaded_file.getbuffer()) / (1024 * 1024),
                content_preview=full_text_content[:200]
            )
            
            session_success = st.session_state.session_manager.add_document_to_session(
                st.session_state.session_id, session_data
            )
            
            if not session_success:
                st.error("❌ Session save failed!")
                return False
            
            status.update(label="✅ Processing completed!", state="complete")
            logger.success(f"Pipeline completed: {uploaded_file.name}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Pipeline failed for {uploaded_file.name}: {e}")
        st.error(f"❌ Processing error: {str(e)}")
        return False
    
    finally:
        # Note: Files are now stored persistently, no cleanup needed
        logger.info(f"File stored persistently at: {persistent_file_path}")

def handle_file_uploads(uploaded_files):
    """Fixed file upload handler with validation."""
    if not uploaded_files:
        return
    
    logger.info(f"Processing {len(uploaded_files)} files")
    
    with processing_context():
        successful = 0
        failed = 0
        
        try:
            for uploaded_file in uploaded_files:
                st.write(f"### 📁 Processing: {uploaded_file.name}")
                
                # Validate before processing
                is_valid, validation_msg = validate_uploaded_file(uploaded_file)
                if not is_valid:
                    st.error(f"❌ Validation failed: {validation_msg}")
                    failed += 1
                    continue
                
                # Check for duplicates safely
                file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
                uploaded_documents = get_session_documents_safely(st.session_state.session_id)
                existing = any(doc.file_hash == file_hash for doc in uploaded_documents)
                
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

def get_user_friendly_tool_name(tool_name: str) -> str:
    """Convert technical tool names to user-friendly names."""
    tool_mappings = {
        "read_full_document": "Document Read",
        "search_in_document": "Document Search",
        "summarize_document": "Document Summary",
        "compare_documents": "Document Comparison",
        "risk_assessment": "Risk Analysis",
        "synthesize_results": "Results Synthesis",
        "web_search": "Web Search",
        "ask_user_for_clarification": "User Clarification"
    }
    return tool_mappings.get(tool_name, tool_name.replace("_", " ").title())

def get_tool_badge_class(success: bool, tool_name: str) -> str:
    """Get CSS class for tool execution badge."""
    if not success:
        return "badge-error"
    elif tool_name in ["ask_user_for_clarification"]:
        return "badge-warning"
    else:
        return "badge-success"



def display_cot_visualization(cot_session: CoTSession):
    """Displays CoT process. Assumes it's already inside an expander."""
    if not cot_session:
        return
    
    try:
        # Complexity Analysis
        if hasattr(cot_session, 'complexity_analysis') and cot_session.complexity_analysis:
            st.subheader("📊 Query Complexity Analysis")
            complexity = cot_session.complexity_analysis
            st.write(f"**Level:** `{complexity.complexity.value}` | **Confidence:** {complexity.confidence:.1%}")
            st.caption(f"Reasoning: {complexity.reasoning}")

        # Tool Executions - Simple list with Show Details
        if cot_session.tool_executions:
            st.subheader("🛠️ Tool Executions")
            
            # Add execution time display
            st.markdown(f"**⏱️ Total Execution Time:** {cot_session.total_execution_time:.2f}s")
            st.markdown("---")
            
            for i, exec in enumerate(cot_session.tool_executions):
                success_icon = "✅" if exec.success else "❌"
                friendly_name = get_user_friendly_tool_name(exec.tool_name)
                
                # Create unique key for each tool execution
                unique_key = f"tool_{i}_{exec.tool_name}_{exec.execution_time:.3f}_{hash(str(exec.arguments))}"
                
                # Simple tool display
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{success_icon} **{friendly_name}** ({exec.execution_time:.2f}s)")
                with col2:
                    if st.button("🔍 Show Details", key=unique_key, type="secondary"):
                        # Toggle state
                        state_key = f"details_{unique_key}"
                        st.session_state[state_key] = not st.session_state.get(state_key, False)
                        st.rerun()
                
                # Show details if button was clicked
                state_key = f"details_{unique_key}"
                if st.session_state.get(state_key, False):
                    with st.container(border=True):
                        st.caption("**Arguments:**")
                        st.json(exec.arguments)
                        st.caption("**Result:**")
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
        # Find document using SessionManager
        uploaded_documents = get_session_documents_safely(st.session_state.session_id)
        
        doc_to_remove = None
        for doc in uploaded_documents:
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
        # Get documents from SessionManager before clearing
        uploaded_documents = get_session_documents_safely(st.session_state.session_id)
        
        # Clear vector collections
        for doc in uploaded_documents:
            try:
                st.session_state.vector_store.delete_collection(doc.vector_collection_name)
            except Exception as e:
                logger.warning(f"Collection delete failed: {e}")
        
        # Clear session manager
        st.session_state.session_manager.clear_session(st.session_state.session_id)
        
        # Clear local state
        st.session_state.chat_history = []
        st.session_state.is_processing = False
        
        st.success("✅ All data cleared!")
        logger.success("Session cleared")
        st.rerun()
        
    except Exception as e:
        logger.exception(f"Clear session error: {e}")
        st.error(f"❌ Clear failed: {str(e)}")

def main():
    """Fixed main function with better error handling."""
    # Initialize session state first to get logo
    initialize_session_state()
    
    # Get session-specific logo
    session_logo = st.session_state.logo_manager.get_session_logo(st.session_state.session_id)
    
    st.set_page_config(
        page_title="DataNeuron Assistant",
        page_icon=session_logo if session_logo != "🧠" else "🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        
        # Enhanced styling with animations and visual feedback
        st.markdown("""
        <style>
            .main-header { text-align: center; padding: 1rem 0; margin-bottom: 2rem; }
            .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
            .status-ready { background-color: #28a745; }
            .status-processing { background-color: #ffc107; animation: pulse 1.5s infinite; }
            
            /* Animations */
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes slideDown {
                from { opacity: 0; max-height: 0; }
                to { opacity: 1; max-height: 500px; }
            }
            
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
            
            /* Tool execution styling */
            .tool-success { 
                animation: fadeIn 0.5s ease-in;
                border-left: 4px solid #28a745;
                padding-left: 10px;
                margin: 5px 0;
            }
            
            .tool-error { 
                animation: shake 0.5s ease-in-out;
                border-left: 4px solid #dc3545;
                padding-left: 10px;
                margin: 5px 0;
            }
            
            .tool-details {
                animation: slideDown 0.3s ease-out;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                background: #f8f9fa;
            }
            
            /* Badge styles */
            .badge-success { 
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                display: inline-block;
                margin: 2px;
            }
            
            .badge-error { 
                background: linear-gradient(135deg, #dc3545, #e74c3c);
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                display: inline-block;
                margin: 2px;
            }
            
            .badge-warning { 
                background: linear-gradient(135deg, #ffc107, #fd7e14);
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                display: inline-block;
                margin: 2px;
            }
            
            /* Mini-cards for reasoning steps */
            .reasoning-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px;
                border-radius: 10px;
                margin: 5px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            
            .reasoning-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }
            
            /* Progress bar styling */
            .progress-container {
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
                position: relative;
            }
            
            .progress-bar {
                background: linear-gradient(90deg, #28a745, #20c997);
                height: 8px;
                border-radius: 10px;
                transition: width 0.3s ease;
                position: relative;
            }
            
            .progress-bar-animated {
                background: linear-gradient(90deg, #28a745, #20c997, #28a745);
                background-size: 200% 100%;
                animation: progressSlide 0.8s linear infinite;
                height: 8px;
                border-radius: 10px;
                transition: width 0.5s ease;
            }
            
            .progress-bar-flowing {
                background: linear-gradient(90deg, #28a745, #20c997, #17a2b8, #28a745, #20c997);
                background-size: 300% 100%;
                animation: progressFlow 2s linear infinite;
                height: 8px;
                border-radius: 10px;
                position: relative;
            }
            
            @keyframes progressSlide {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            @keyframes progressFlow {
                0% { background-position: 300% 0; }
                100% { background-position: -300% 0; }
            }
            
            .progress-pulse {
                animation: progressPulse 1.5s ease-in-out infinite;
            }
            
            @keyframes progressPulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            /* Execution time chart styling */
            .time-bar {
                background: linear-gradient(90deg, #007bff, #0056b3);
                height: 20px;
                border-radius: 4px;
                margin: 2px 0;
                display: flex;
                align-items: center;
                padding: 0 8px;
                color: white;
                font-size: 0.8em;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Header with dynamic logo
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        
        # Display logo and title
        session_logo = st.session_state.logo_manager.get_session_logo(st.session_state.session_id)
        
        # Display title with properly aligned logo
        if session_logo != "🧠":  # If we have a custom logo file
            # Better aligned logo and title
            col1, col2 = st.columns([0.15, 0.85])
            with col1:
                st.image(session_logo, width=10000)
            with col2:
                st.markdown("<h1 style='margin-top: 10px;'>DataNeuron Assistant</h1>", unsafe_allow_html=True)
        else:  # Fallback to emoji
            st.title("🧠 DataNeuron Assistant")
            
        st.markdown("*AI-powered document analysis with Chain of Thought reasoning*")
        
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
            st.markdown("Upload PDF, DOCX, or TXT documents (max 50MB each):")
            
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
            # Render chat (fixed version handles everything internally)
            render_chat_interface()
            
            # Help messages
            uploaded_documents = get_session_documents_safely(st.session_state.session_id)
            doc_count = len(uploaded_documents)
            
            if doc_count == 0:
                st.info("""
                💡 **Getting Started:**
                1. **Upload Documents:** Go to "📁 Upload Documents" tab and upload PDF, DOCX, or TXT files
                2. **Enable Web Search:** Toggle "İnternette Ara" above for access to current web information  
                3. **Ask Questions:** Return here to ask about your documents or current topics
                4. **Get Smart Answers:** AI will analyze your documents and/or search the web for comprehensive responses
                """)
            elif not st.session_state.chat_history:
                web_status = "🌐 Web search enabled" if st.session_state.get('web_search_enabled', False) else "📄 Document-only mode"
                st.info(f"""
                💡 **Ready to Chat!**
                
                **Status:** {doc_count} document(s) loaded | {web_status}
                
                **Document Questions:**
                - "What are the main topics in my documents?"
                - "Summarize key findings"
                - "Compare different approaches in the files"
                
                **Web Search Questions** (if enabled):
                - "What are the latest developments in [topic]?"
                - "Find current information about [company/person]"
                - "Compare my document findings with current market trends"
                """)
    
    except Exception as e:
        logger.exception(f"Critical error: {e}")
        st.error(f"❌ **Critical Application Error**: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()