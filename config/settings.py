"""
DataNeuron RAG Project Configuration
====================================

This module contains all configuration settings for the DataNeuron RAG project.
Settings are loaded from environment variables via a .env file in the project root.
All critical settings are validated to ensure proper application startup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Determine the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")


# ============================================================================
# FILE STORAGE CONFIGURATION
# ============================================================================
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_NAME = "DataNeuron"
PROJECT_VERSION = "1.0.0"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = BASE_DIR / "logs" / "dataneuron.log"

# Ensure logs directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ANTHROPIC API CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "Error: ANTHROPIC_API_KEY environment variable not found. "
        "Please check your .env file."
    )

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


# ============================================================================
# VECTOR DATABASE (CHROMADB) CONFIGURATION
# ============================================================================

# ChromaDB path configuration
CHROMADB_PATH_STR = os.getenv("CHROMADB_PATH", "./data/chroma_db")
CHROMADB_PATH = BASE_DIR / CHROMADB_PATH_STR.lstrip("./")

# Ensure ChromaDB directory exists
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)

CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "dataneuron_docs")


# ============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# ============================================================================

# Supported file extensions for document processing
SUPPORTED_EXTENSIONS_STR = os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.txt,.docx")
SUPPORTED_EXTENSIONS = [ext.strip() for ext in SUPPORTED_EXTENSIONS_STR.split(",")]

# Maximum file size limit in MB
MAX_FILE_SIZE_MB_STR = os.getenv("MAX_FILE_SIZE_MB", "20")
try:
    MAX_FILE_SIZE_MB = int(MAX_FILE_SIZE_MB_STR)
except ValueError:
    raise ValueError(
        f"Error: MAX_FILE_SIZE_MB value is not a valid number: {MAX_FILE_SIZE_MB_STR}"
    )

# Document processing temporary directory
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SESSION MANAGEMENT CONFIGURATION
# ============================================================================

# Session state persistence file
SESSION_STATE_FILE_STR = os.getenv("SESSION_STATE_FILE", "data/session_state.json")
SESSION_STATE_FILE = BASE_DIR / SESSION_STATE_FILE_STR.lstrip("./")

# Ensure session data directory exists
SESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Session timeout in hours (optional feature)
SESSION_TIMEOUT_HOURS_STR = os.getenv("SESSION_TIMEOUT_HOURS", "24")
try:
    SESSION_TIMEOUT_HOURS = int(SESSION_TIMEOUT_HOURS_STR)
except ValueError:
    raise ValueError(
        f"Error: SESSION_TIMEOUT_HOURS value is not a valid number: {SESSION_TIMEOUT_HOURS_STR}"
    )


# ============================================================================
# TOOL MANAGEMENT CONFIGURATION
# ============================================================================

# Directory containing tool modules
TOOLS_DIR_STR = os.getenv("TOOLS_DIR", "tools")
TOOLS_DIR = BASE_DIR / TOOLS_DIR_STR

# Tool discovery settings
TOOL_AUTO_DISCOVERY = os.getenv("TOOL_AUTO_DISCOVERY", "true").lower() == "true"
TOOL_DISCOVERY_RECURSIVE = os.getenv("TOOL_DISCOVERY_RECURSIVE", "false").lower() == "true"

# Tool execution settings
TOOL_EXECUTION_TIMEOUT_SECONDS_STR = os.getenv("TOOL_EXECUTION_TIMEOUT_SECONDS", "60")
try:
    TOOL_EXECUTION_TIMEOUT_SECONDS = int(TOOL_EXECUTION_TIMEOUT_SECONDS_STR)
except ValueError:
    raise ValueError(
        f"Error: TOOL_EXECUTION_TIMEOUT_SECONDS value is not a valid number: {TOOL_EXECUTION_TIMEOUT_SECONDS_STR}"
    )


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Text chunking parameters for optimal LLM and vector database performance
CHUNK_SIZE_STR = os.getenv("CHUNK_SIZE", "1000")
try:
    CHUNK_SIZE = int(CHUNK_SIZE_STR)
except ValueError:
    raise ValueError(
        f"Error: CHUNK_SIZE value is not a valid number: {CHUNK_SIZE_STR}"
    )

CHUNK_OVERLAP_STR = os.getenv("CHUNK_OVERLAP", "200")  
try:
    CHUNK_OVERLAP = int(CHUNK_OVERLAP_STR)
except ValueError:
    raise ValueError(
        f"Error: CHUNK_OVERLAP value is not a valid number: {CHUNK_OVERLAP_STR}"
    )

# Tiktoken encoding model for token counting (compatible with OpenAI models)
TIKTOKEN_ENCODING_MODEL = os.getenv("TIKTOKEN_ENCODING_MODEL", "cl100k_base")

# Text splitting separators (preserve semantic boundaries)
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Maximum chunk size in characters as fallback
MAX_CHUNK_SIZE_CHARS_STR = os.getenv("MAX_CHUNK_SIZE_CHARS", "4000")
try:
    MAX_CHUNK_SIZE_CHARS = int(MAX_CHUNK_SIZE_CHARS_STR)  
except ValueError:
    raise ValueError(
        f"Error: MAX_CHUNK_SIZE_CHARS value is not a valid number: {MAX_CHUNK_SIZE_CHARS_STR}"
    )


# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# OpenAI Embedding model and batch processing settings
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Error: OPENAI_API_KEY environment variable not found. "
        "Please check your .env file."
    )

EMBEDDING_DIMENSION_STR = os.getenv("EMBEDDING_DIMENSION", "3072")
try:
    EMBEDDING_DIMENSION = int(EMBEDDING_DIMENSION_STR)
except ValueError:
    raise ValueError(
        f"Error: EMBEDDING_DIMENSION value is not a valid number: {EMBEDDING_DIMENSION_STR}"
    )

# Batch size for embedding API calls (optimize for rate limits and performance)
EMBEDDING_BATCH_SIZE_STR = os.getenv("EMBEDDING_BATCH_SIZE", "16")
try:
    EMBEDDING_BATCH_SIZE = int(EMBEDDING_BATCH_SIZE_STR)
except ValueError:
    raise ValueError(
        f"Error: EMBEDDING_BATCH_SIZE value is not a valid number: {EMBEDDING_BATCH_SIZE_STR}"
    )


# ============================================================================
# WEB TOOLS (OPTIONAL)
# ============================================================================

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Optional - no validation required


# ============================================================================
# STREAMLIT (UI) CONFIGURATION
# ============================================================================

STREAMLIT_PORT_STR = os.getenv("STREAMLIT_PORT", "8501")
try:
    STREAMLIT_PORT = int(STREAMLIT_PORT_STR)
except ValueError:
    raise ValueError(
        f"Error: STREAMLIT_PORT value is not a valid number: {STREAMLIT_PORT_STR}"
    )


# ============================================================================
# VALIDATION SUMMARY
# ============================================================================

def validate_settings():
    """
    Validate all critical settings and provide a summary.
    This function can be called to ensure all settings are properly configured.
    """
    print(f"✓ Project: {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"✓ Base Directory: {BASE_DIR}")
    print(f"✓ Log Level: {LOG_LEVEL}")
    print(f"✓ Log File: {LOG_FILE}")
    print(f"✓ Anthropic Model: {ANTHROPIC_MODEL}")
    print(f"✓ ChromaDB Path: {CHROMADB_PATH}")
    print(f"✓ ChromaDB Collection: {CHROMADB_COLLECTION}")
    print(f"✓ Supported Extensions: {SUPPORTED_EXTENSIONS}")
    print(f"✓ Max File Size: {MAX_FILE_SIZE_MB} MB")
    print(f"✓ Temp Directory: {TEMP_DIR}")
    print(f"✓ Session State File: {SESSION_STATE_FILE}")
    print(f"✓ Session Timeout: {SESSION_TIMEOUT_HOURS} hours")
    print(f"✓ Tools Directory: {TOOLS_DIR}")
    print(f"✓ Tool Auto Discovery: {TOOL_AUTO_DISCOVERY}")
    print(f"✓ Tool Execution Timeout: {TOOL_EXECUTION_TIMEOUT_SECONDS} seconds")
    print(f"✓ Chunk Size: {CHUNK_SIZE} tokens")
    print(f"✓ Chunk Overlap: {CHUNK_OVERLAP} tokens")
    print(f"✓ Tiktoken Encoding: {TIKTOKEN_ENCODING_MODEL}")
    print(f"✓ Max Chunk Size (chars): {MAX_CHUNK_SIZE_CHARS}")
    print(f"✓ Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"✓ Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"✓ Streamlit Port: {STREAMLIT_PORT}")
    print(f"✓ OpenAI API Key: {'Configured' if OPENAI_API_KEY else 'Not configured'}")
    print(f"✓ Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"✓ Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"✓ Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    
    if SERPAPI_KEY:
        print("✓ SerpAPI Key: Configured")
    else:
        print("! SerpAPI Key: Not configured (optional)")
    
    print("✓ All settings loaded successfully!")


if __name__ == "__main__":
    validate_settings()