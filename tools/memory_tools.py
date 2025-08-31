"""
DataNeuron Session Memory Management Tools
==========================================

This module provides tools for managing session-specific memory storage,
allowing the LLM agent to persistently store and retrieve important information
learned from conversations within each user session.

Features:
- Store key-value pairs in session memory
- Retrieve session memory contents
- Session-isolated memory storage
- Persistent memory across conversation turns

Usage:
    from tools.memory_tools import UpdateSessionMemoryTool
    
    tool = UpdateSessionMemoryTool()
    result = tool.execute(
        key="project_name",
        value="DataNeuron AI Assistant",
        session_id="user_session_123"
    )
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import Field
from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from core.session_manager import SessionManager
from utils.logger import logger


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class UpdateMemoryArgs(BaseToolArgs):
    """
    Arguments schema for updating session memory.
    """
    key: str = Field(..., description="The key for the information to be stored (e.g., 'project_name', 'user_preference').")
    value: str = Field(..., description="The value of the information to store.")
    session_id: str = Field(..., description="The user's session ID.")


class UpdateMemoryResult(BaseToolResult):
    """
    Result schema for session memory update operations.
    """
    confirmation: str = Field(..., description="Confirmation message indicating the operation status.")


class GetMemoryArgs(BaseToolArgs):
    """
    Arguments schema for retrieving session memory.
    """
    session_id: str = Field(..., description="The user's session ID.")


# ============================================================================
# SESSION MEMORY TOOLS
# ============================================================================

class UpdateSessionMemoryTool(BaseTool):
    """
    Tool for updating session memory with key-value pairs.
    
    This tool allows the LLM agent to store important information learned
    from conversations in a persistent session memory that survives across
    conversation turns.
    """
    
    name = "update_session_memory"
    description = "Updates or stores a key-value pair in the session's long-term memory. Use this to remember critical facts confirmed by the user, like a project name or a specific decision. This information will be available for all future turns in the conversation."
    args_schema = UpdateMemoryArgs
    return_schema = UpdateMemoryResult
    
    def _execute(self, key: str, value: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the session memory update operation.
        
        Args:
            key: The memory key to store the value under
            value: The value to store in session memory
            session_id: The user's session identifier
            
        Returns:
            Dictionary containing confirmation message
        """
        try:
            # Use injected session manager instead of creating new instance
            if not self.session_manager:
                raise ValueError("SessionManager not available - tool not properly initialized")
            
            # Update session memory
            success = self.session_manager.update_session_memory(session_id, key, value)
            
            if success:
                confirmation_msg = f"Successfully stored '{value}' for key '{key}' in session memory."
                logger.info(f"Memory tool executed: {key} = {value} for session {session_id}")
                return {"confirmation": confirmation_msg}
            else:
                error_msg = f"Failed to store '{value}' for key '{key}' in session memory."
                logger.error(f"Memory tool failed: {error_msg}")
                return {
                    "success": False,
                    "error_message": error_msg,
                    "confirmation": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error updating session memory: {str(e)}"
            logger.exception(f"Memory tool exception: {error_msg}")
            return {
                "success": False,
                "error_message": error_msg,
                "confirmation": f"Failed to update session memory due to error: {str(e)}"
            }


class GetSessionMemoryTool(BaseTool):
    """
    Tool for retrieving all session memory contents.
    
    This tool allows the LLM agent to access previously stored information
    from the session memory.
    """
    
    name = "get_session_memory"
    description = "Retrieves all stored key-value pairs from the session's long-term memory. Use this to recall important information that was previously stored during the conversation."
    args_schema = GetMemoryArgs
    return_schema = BaseToolResult
    
    def _execute(self, session_id: str) -> Dict[str, Any]:
        """
        Execute the session memory retrieval operation.
        
        Args:
            session_id: The user's session identifier
            
        Returns:
            Dictionary containing all session memory data
        """
        try:
            # Use injected session manager instead of creating new instance
            if not self.session_manager:
                raise ValueError("SessionManager not available - tool not properly initialized")
            
            # Get session memory
            memory_data = self.session_manager.get_session_memory(session_id)
            
            logger.info(f"Memory retrieval tool executed for session {session_id}: {len(memory_data)} items")
            
            return {
                "success": True,
                "memory_data": memory_data,
                "item_count": len(memory_data),
                "confirmation": f"Retrieved {len(memory_data)} items from session memory."
            }
                
        except Exception as e:
            error_msg = f"Error retrieving session memory: {str(e)}"
            logger.exception(f"Memory retrieval tool exception: {error_msg}")
            return {
                "success": False,
                "error_message": error_msg,
                "memory_data": {},
                "item_count": 0,
                "confirmation": f"Failed to retrieve session memory due to error: {str(e)}"
            }


# ============================================================================
# TOOL EXPORTS
# ============================================================================

# Export tools for ToolManager auto-discovery
__all__ = [
    "UpdateSessionMemoryTool",
    "GetSessionMemoryTool",
    "UpdateMemoryArgs",
    "UpdateMemoryResult",
    "GetMemoryArgs"
]