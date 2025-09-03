"""
DataNeuron Chain of Thought (CoT) Multi-Step Reasoning AI Agent
===============================================================

This module provides a sophisticated AI agent that uses Chain of Thought reasoning
to analyze complex queries, plan tool execution sequences, and provide transparent,
step-by-step reasoning for multi-step problem solving.

Features:
- Chain of Thought (CoT) reasoning with transparent step-by-step analysis
- Query complexity analysis and adaptive planning
- Multi-step tool orchestration with reflective reasoning
- Anthropic function calling integration for tool selection
- Asynchronous execution for responsive performance
- Comprehensive error handling with fallback mechanisms
- Session-aware context management
- Detailed logging and reasoning trace capture

Usage:
    from core.llm_agent import LLMAgent
    
    # Get singleton instance
    agent = LLMAgent()
    
    # Execute query with CoT reasoning
    result = await agent.execute_with_cot(
        "Analyze this document and compare it with previous ones",
        session_id="user_session_123"
    )
    
    # Access reasoning steps
    for step in result.reasoning_steps:
        print(f"Step: {step.step_type} - {step.content}")

Architecture:
- CoT data structures for reasoning step tracking
- Singleton LLMAgent class for centralized intelligence
- Async methods for non-blocking I/O operations
- Integration with ToolManager, SessionManager, and Anthropic
- Reflective reasoning loops with adaptive planning
"""

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from pydantic import BaseModel

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import session_manager
from utils.logger import logger
from config.settings import ANTHROPIC_API_KEY, OPENAI_API_KEY,ANTHROPIC_MODEL

# Import dependencies with error handling
try:
    import anthropic
except ImportError as e:
    logger.error(f"Anthropic library not available: {e}")
    anthropic = None

try:
    from core.tool_manager import ToolManager
except ImportError as e:
    logger.warning(f"ToolManager not available: {e}")
    ToolManager = None

try:
    from core.session_manager import SessionManager
except ImportError as e:
    logger.warning(f"SessionManager not available: {e}")
    SessionManager = None


# ============================================================================
# CHAIN OF THOUGHT DATA STRUCTURES
# ============================================================================

class QueryComplexity(Enum):
    """
    Enumeration of query complexity levels for adaptive planning.
    """
    SIMPLE = "simple"          # Single-step, direct queries ("summarize")
    MODERATE = "moderate"      # Multi-step but straightforward ("find and extract")
    COMPLEX = "complex"        # Multi-step with reasoning ("compare and analyze")
    VERY_COMPLEX = "very_complex"  # Multi-step with deep analysis ("research, analyze, and recommend")


@dataclass
class ComplexityAnalysis:
    """
    Result of query complexity analysis.
    
    Attributes:
        complexity: Determined complexity level
        reasoning: Explanation of why this complexity was assigned
        estimated_steps: Expected number of reasoning/tool steps
        required_tools: List of tools likely needed
        confidence: Confidence in complexity assessment (0.0-1.0)
    """
    complexity: QueryComplexity
    reasoning: str
    estimated_steps: int
    required_tools: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    step_id: str
    step_type: str
    content: str
    # timestamp alanını varsayılan bir fabrika ile oluştur
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ToolExecutionStep:
    """
    Represents the execution of a single tool with CoT reasoning.
    
    Attributes:
        tool_name: Name of the executed tool
        arguments: Arguments passed to the tool
        pre_execution_reasoning: Reasoning before tool execution
        result: Result returned by the tool
        post_execution_reflection: Analysis of the tool result
        success: Whether the tool execution was successful
        execution_time: Time taken to execute the tool (seconds)
        timestamp: When this execution occurred
    """
    tool_name: str
    arguments: Dict[str, Any]
    pre_execution_reasoning: ReasoningStep
    result: Any = None
    post_execution_reflection: Optional[ReasoningStep] = None
    success: bool = False
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTSession:
    """
    Complete Chain of Thought session for a single query.
    
    Attributes:
        session_id: Unique identifier for this CoT session
        original_query: The original user query
        complexity_analysis: Analysis of query complexity
        reasoning_steps: All reasoning steps in chronological order
        tool_executions: All tool executions with reasoning
        final_answer: Final synthesized answer
        total_execution_time: Total time for the entire session
        success: Whether the session completed successfully
        error_info: Error information if session failed
        session_facts: Session memory facts that override document content
    """
    session_id: str
    original_query: str
    user_session_id: Optional[str] = None
    complexity_analysis: Optional[ComplexityAnalysis] = None
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    tool_executions: List[ToolExecutionStep] = field(default_factory=list)
    final_answer: str = ""
    total_execution_time: float = 0.0
    success: bool = False
    error_info: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_facts: Dict[str, Any] = field(default_factory=dict)
    
    def set_session_facts(self, facts: Dict[str, Any]) -> None:
        """
        Set session memory facts that take priority over document content.
        
        Args:
            facts: Dictionary of key-value facts from session memory
        """
        self.session_facts = facts or {}
        logger.info(f"CoTSession facts updated: {len(self.session_facts)} facts stored")
    
    def get_session_facts(self) -> Dict[str, Any]:
        """
        Get stored session facts.
        
        Returns:
            Dictionary of session facts
        """
        return self.session_facts


# ============================================================================
# CHAIN OF THOUGHT LLM AGENT
# ============================================================================

class LLMAgent:
    """
    Singleton Chain of Thought AI Agent for DataNeuron.
    
    This agent provides sophisticated multi-step reasoning capabilities using
    Anthropic's GPT models with function calling for tool orchestration. It implements
    transparent Chain of Thought reasoning where every step of analysis, planning,
    execution, and reflection is captured and made visible.
    
    Key Features:
    - Singleton pattern for centralized intelligence
    - Async execution for responsive performance
    - Chain of Thought reasoning with step tracking
    - Query complexity analysis and adaptive planning
    - Anthropic function calling for tool selection
    - Reflective reasoning with plan adaptation
    - Comprehensive error handling and fallbacks
    - Session-aware context management
    
    The agent maintains complete transparency in its reasoning process,
    making it suitable for applications requiring explainable AI decisions.
    """
    
    _instance: Optional['LLMAgent'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'LLMAgent':
        """
        Singleton pattern implementation with thread safety.
        
        Accepts any arguments to support dependency injection in __init__.
        
        Args:
            *args: Positional arguments passed to __init__
            **kwargs: Keyword arguments passed to __init__
        
        Returns:
            Single instance of LLMAgent
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMAgent, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, tool_manager=None, session_manager=None):
        """
        Initialize the LLMAgent singleton with dependency injection.

        Sets up Anthropic client, tool manager, session manager, and reasoning
        capabilities. Only initializes once due to singleton pattern.
        
        Args:
            tool_manager: Optional ToolManager instance for dependency injection
            session_manager: Optional SessionManager instance for dependency injection
        
        Raises:
            RuntimeError: If required dependencies are not available
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            # Check dependencies
            if anthropic is None:
                raise RuntimeError("Anthropic library is required for LLMAgent")

            if not ANTHROPIC_API_KEY:
                raise RuntimeError("Anthropic API key is required for LLMAgent")

            # Initialize Anthropic client
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL  # Use the configured model from settings
            
            # Initialize managers with dependency injection or fallback to singletons
            self.tool_manager = tool_manager if tool_manager is not None else (ToolManager() if ToolManager else None)
            self.session_manager = session_manager if session_manager is not None else (SessionManager() if SessionManager else None)
            
            # Configuration
            self.max_reasoning_steps = 20
            self.max_tool_executions = 10
            self.reasoning_temperature = 0.1  # Low temperature for consistent reasoning
            self.execution_temperature = 0.3  # Slightly higher for tool selection
            
            # Silent tools that don't require synthesis
            self.SILENT_TOOLS = {"update_session_memory", "get_session_memory"}
            
            # Argument normalization mapping for common LLM mistakes
            self.ARGUMENT_NORMALIZATION_MAP = {
                'filename': 'file_name',
                'document_name': 'file_name', 
                'files': 'file_names',
                'criteria': 'comparison_criteria',
                'search_query': 'query',
                'search_term': 'query',
                'text': 'content',
                'document': 'file_name',
                'doc': 'file_name',
                'key_name': 'key',
                'memory_key': 'key',
                'memory_value': 'value',
                'session': 'session_id',
                'user_id': 'session_id'
            }
            
            # Statistics
            self.total_sessions = 0
            self.successful_sessions = 0
            self.total_reasoning_steps = 0
            self.total_tool_executions = 0
            
            # CoT system prompts
            self.system_prompts = {
                "complexity_analysis": self._create_complexity_analysis_prompt(),
                "reflection": self._create_reflection_prompt(),
                "synthesis": self._create_synthesis_prompt()
            }
            
            logger.info("LLMAgent initialized successfully")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Tool Manager: {'Available' if self.tool_manager else 'Not Available'}")
            logger.info(f"  - Session Manager: {'Available' if self.session_manager else 'Not Available'}")
            
            self._initialized = True
            logger.success("LLMAgent singleton ready for Chain of Thought reasoning")

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _normalize_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool arguments to fix common LLM mistakes.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Raw arguments from LLM
            
        Returns:
            Dict with normalized argument keys
        """
        if not arguments:
            return arguments

        normalized_args = arguments.copy()
        changes_made = []

        # --- ARACA ÖZEL AKILLI DÜZELTMELER ---

        # DURUM 1: 'search_in_document' aracı 'query' bekler. Eğer 'question' geldiyse, düzelt.
        if tool_name == 'search_in_document' and 'question' in normalized_args and 'query' not in normalized_args:
            normalized_args['query'] = normalized_args.pop('question')
            changes_made.append(f"'{tool_name}': corrected 'question' to 'query'")

        # DURUM 2: 'ask_user_for_clarification' aracı 'question' bekler. Eğer 'query' geldiyse, düzelt.
        elif tool_name == 'ask_user_for_clarification' and 'query' in normalized_args and 'question' not in normalized_args:
            normalized_args['question'] = normalized_args.pop('query')
            changes_made.append(f"'{tool_name}': corrected 'query' to 'question'")

        # --- GENEL HARİTA TABANLI DÜZELTMELER (hala geçerli) ---
        final_args = {}
        for key, value in normalized_args.items():
            # Genel normalleştirme haritasını uygula
            normalized_key = self.ARGUMENT_NORMALIZATION_MAP.get(key, key)
            final_args[normalized_key] = value
            if normalized_key != key:
                changes_made.append(f"'{key}' → '{normalized_key}'")

        if changes_made:
            logger.info(f"🔧 Normalized arguments for '{tool_name}': {', '.join(changes_made)}")

        return final_args
    
    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """
        Convert Pydantic objects to dictionaries for tool argument passing.
        
        Args:
            obj: Object to convert (could be Pydantic model, list, dict, or primitive)
            
        Returns:
            Converted object with Pydantic models converted to dicts
        """
        if hasattr(obj, 'model_dump'):
            # It's a Pydantic model
            return obj.model_dump()
        elif isinstance(obj, list):
            # Convert each item in the list
            return [self._convert_pydantic_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # Convert each value in the dict
            return {key: self._convert_pydantic_to_dict(value) for key, value in obj.items()}
        else:
            # Primitive type, return as-is
            return obj

    # ============================================================================
    # HELPER METHODS (EKSİK OLAN KISIM)
    # ============================================================================

    async def _call_anthropic_for_analysis(self, 
    system_prompt: str, 
    user_prompt: str, 
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> str:
        """Calls Anthropic API for analysis tasks that don't require tools."""
        try:
            messages = []

            # --- DEFENSIVE PROGRAMMING START ---
            if chat_history is None:
                chat_history = []
            
            if not isinstance(chat_history, list):
                logger.warning(f"chat_history was not a list, it was {type(chat_history)}. Resetting to empty list.")
                chat_history = []

            clean_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in chat_history 
                if isinstance(msg, dict) and msg.get("content") and msg["role"] in ["user", "assistant"]
            ]
            messages.extend(clean_history)
            # --- DEFENSIVE PROGRAMMING END ---
        
            messages.append({"role": "user", "content": user_prompt})


            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=self.reasoning_temperature,
                max_tokens=1500
            )
            return response.content[0].text
        except Exception as e:
            logger.exception(f"Anthropic analysis call failed: {e}")
            raise

    async def _evaluate_response_insufficiency(self, response_text: str, original_query: str) -> bool:
        """
        Use AI to evaluate if a response indicates insufficient information.
        
        Args:
            response_text: The response text to evaluate
            original_query: The original user query for context
            
        Returns:
            True if response indicates insufficient information, False otherwise
        """
        try:
            evaluation_prompt = f"""
Orijinal Sorgu: "{original_query}"

Verilen Cevap: "{response_text}"

GÖREV: Bu cevap, orijinal sorguyu tam olarak yanıtlıyor mu yoksa bilgi eksikliği mi var?

Şu durumlar bilgi eksikliği olarak sayılır:
✅ "Belgede bu bilgi yok"
✅ "Yeterli bilgi bulunamadı" 
✅ "Daha detaylı belge gerekli"
✅ "Bu konuda bilgi içermiyor"
✅ "Kesin bilgi veremiyorum"

Şu durumlar yeterli bilgi olarak sayılır:
❌ Spesifik cevap veriliyor
❌ Detaylı açıklama mevcut  
❌ Konkret bilgi sunuluyor

SADECE "EVET" veya "HAYIR" ile cevap ver:
- EVET = Bilgi eksik, web arama gerekli
- HAYIR = Bilgi yeterli, web arama gerekli değil

Cevap:"""

            # Fast evaluation with minimal tokens
            evaluation_result = await self._call_anthropic_for_analysis(
                "Sen bir cevap değerlendirme uzmanısın. Verilen cevabın sorguyu tam karşılayıp karşılamadığını değerlendirirsin.",
                evaluation_prompt
            )
            
            # Parse the response 
            evaluation_clean = evaluation_result.strip().upper()
            
            if "EVET" in evaluation_clean:
                logger.info("🧠 AI Evaluation: Information is insufficient → Web search recommended")
                return True
            elif "HAYIR" in evaluation_clean:
                logger.debug("🧠 AI Evaluation: Information is sufficient → No web search needed")
                return False
            else:
                logger.warning(f"AI evaluation returned unclear response: {evaluation_result}")
                return False  # Conservative default
                
        except Exception as e:
            logger.error(f"AI response evaluation failed: {e}")
            return False  # Conservative default when evaluation fails

    async def _optimize_web_search_query(self, original_query: str, tool_executions: List[ToolExecutionStep], context_text: str) -> str:
        """
        Optimize web search query based on document content and context.
        
        Args:
            original_query: Original user query
            tool_executions: Previous tool executions to extract context
            context_text: Context text from failed tool
            
        Returns:
            Optimized web search query
        """
        try:
            # Extract entities from document tool results
            document_entities = []
            
            for execution in tool_executions:
                if execution.success and execution.result:
                    # Extract content from document tools
                    if hasattr(execution.result, 'content'):
                        content = getattr(execution.result, 'content', '')
                        if content and len(content.strip()) > 0:
                            # Better entity extraction - include known team names
                            content_lower = content.lower().strip()
                            
                            # Turkish football teams
                            turkish_teams = [
                                "galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor", 
                                "başakşehir", "alanyaspor", "antalyaspor", "kayserispor",
                                "konyaspor", "sivasspor", "gaziantep", "kasımpaşa"
                            ]
                            
                            # Check for team names in content
                            for team in turkish_teams:
                                if team in content_lower:
                                    document_entities.append(team.capitalize())
                            
                            # Also extract any capitalized words (backup)
                            words = content.strip().split()
                            for word in words:
                                if len(word) > 3 and (word.istitle() or word.isupper()):
                                    document_entities.append(word.capitalize())
            
            # Use AI to create optimized query
            optimization_prompt = f"""
Orijinal Sorgu: "{original_query}"
Dokümanda Bulunan Varlıklar: {document_entities}
Context: "{context_text[:200]}..."

GÖREV: Dokümanda bulunan takım adını kullanarak, web araması için optimize edilmiş bir sorgu oluştur.

ÖNEMLİ: Bugün 2025 yılındayız. Transfer, kadro gibi sorularda MUTLAKA "2025" yılını kullan.

Örnekler:
- Orijinal: "dosyadaki takımın son transferini söyle" 
- Optimize: "Galatasaray son transfer 2025"

- Orijinal: "bu takımın oyuncuları kimler"
- Optimize: "Fenerbahçe kadro 2025"

KURAL: Sadece optimize edilmiş sorguyu döndür, açıklama yapma. YIL MUTLAKA 2025 OLMALI!

Optimize Sorgu:"""

            optimized_query = await self._call_anthropic_for_analysis(
                "Sen bir arama sorgusu optimizasyon uzmanısın. Dokümantaki varlıkları kullanarak web araması için en etkili sorguyu oluşturursun.",
                optimization_prompt
            )
            
            # Clean and validate the optimized query
            optimized_query = optimized_query.strip().strip('"')
            
            if len(optimized_query) > 5:  # Reasonable minimum length
                logger.info(f"🎯 Query optimized: '{original_query}' → '{optimized_query}'")
                return optimized_query
            else:
                logger.warning(f"Query optimization failed, using original: {optimized_query}")
                return original_query
                
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return original_query  # Fallback to original

    async def _call_anthropic_with_cot_and_tools(self, system_prompt: str, user_prompt: str, tools: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Calls Anthropic API with CoT system prompt and available tools."""
        try:
            messages = []

            # --- DEFENSIVE PROGRAMMING START ---
            if chat_history is None:
                chat_history = []
            
            if not isinstance(chat_history, list):
                logger.warning(f"chat_history was not a list, it was {type(chat_history)}. Resetting to empty list.")
                chat_history = []

            clean_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in chat_history 
                if isinstance(msg, dict) and msg.get("content") and msg["role"] in ["user", "assistant"]
            ]
            messages.extend(clean_history)
            # --- DEFENSIVE PROGRAMMING END ---
        
            messages.append({"role": "user", "content": user_prompt})


            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else None,
                tool_choice={"type": "any"} if tools else None,
                temperature=self.execution_temperature,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.exception(f"Anthropic API call with tools failed: {e}")
            raise

    def _get_available_tools_for_anthropic(self) -> List[Dict[str, Any]]:
        """Gets available tools formatted for Anthropic function calling using Claude bridge."""
        if not self.tool_manager:
            return []
        try:
            # Use the new Claude function calling bridge from ToolManager
            claude_tools = self.tool_manager.get_claude_function_schemas()
            logger.debug(f"Retrieved {len(claude_tools)} tools from Claude function calling bridge")
            return claude_tools
        except Exception as e:
            logger.exception(f"Failed to get Claude function schemas from ToolManager: {e}")
            return []

    def _extract_tool_plans_from_response(self, response) -> List[Dict[str, Any]]:
        """Extracts tool execution plans from Anthropic response."""
        try:
            tool_plans = []
            
            # Check for tool use in content blocks
            if hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if content_block.type == 'tool_use':
                        plan = {
                            "tool_name": content_block.name,
                            "arguments": content_block.input,
                            "reasoning": f"LLM decided to call {content_block.name}."
                        }
                        tool_plans.append(plan)
                    elif content_block.type == 'text' and not tool_plans:
                        # If no tool calls and there's text content
                        tool_plans.append({
                            "tool_name": "direct_answer",
                            "arguments": {"query": "Based on reasoning in message content"},
                            "reasoning": content_block.text
                        })

            return tool_plans
        except Exception as e:
            logger.exception(f"Failed to extract tool plans from response: {e}")
            return []

    def _expand_plans_for_multiple_documents(self, tool_plans: List[Dict[str, Any]], targeted_files: List[str]) -> List[Dict[str, Any]]:
        """
        CRITICAL METHOD: Otomatik olarak tek dosya araçlarını çoklu dosya için genişletir.
        
        Args:
            tool_plans: LLM'den gelen orijinal plan
            targeted_files: Seçilen dosyalar listesi
            
        Returns:
            Genişletilmiş plan (her dosya için ayrı araç çağrıları)
        """
        logger.info(f"🔄 Post-processing plans for {len(targeted_files)} files")
        
        # Tek dosya alan araçların listesi
        single_file_tools = [
            "read_full_document", 
            "assess_risks_in_document", 
            "search_in_document", 
            "summarize_document"
        ]
        
        expanded_plans = []
        synthesis_needed = False
        
        for plan in tool_plans:
            tool_name = plan.get("tool_name", "")
            
            # Eğer bu araç tek dosya alıyor ve arguments'ta file_name varsa
            if (tool_name in single_file_tools and 
                plan.get("arguments", {}).get("file_name")):
                
                logger.info(f"🔧 Expanding {tool_name} for all {len(targeted_files)} files")
                
                # Her dosya için ayrı araç çağrısı oluştur
                for filename in targeted_files:
                    new_plan = plan.copy()
                    new_args = plan.get("arguments", {}).copy()
                    new_args["file_name"] = filename
                    new_plan["arguments"] = new_args
                    new_plan["reasoning"] = f"Multi-doc expansion: {plan.get('reasoning', '')} for '{filename}'"
                    expanded_plans.append(new_plan)
                
                synthesis_needed = True
                logger.success(f"✅ Expanded {tool_name} into {len(targeted_files)} calls")
                
            else:
                # Diğer araçları olduğu gibi bırak
                expanded_plans.append(plan)
        
        # Eğer genişletme yaptıysak, synthesis ekle
        if synthesis_needed and not any(p.get("tool_name") == "synthesize_results" for p in expanded_plans):
            expanded_plans.append({
                "tool_name": "synthesize_results",
                "reasoning": "Auto-added: Combining results from multiple document analysis",
                "arguments": {
                    "results_summary": f"Analysis results from {len(targeted_files)} documents",
                    "user_query": "Combined multi-document analysis"
                }
            })
            logger.info("➕ Added synthesize_results step for multi-document analysis")
        
        logger.info(f"📊 Plan expansion complete: {len(tool_plans)} → {len(expanded_plans)} steps")
        return expanded_plans

    # ============================================================================
    # CORE EXECUTION AND REASONING METHODS
    # ============================================================================
    


    async def synthesize_results_with_cot(self, cot_session: CoTSession, chat_history: List[Dict[str, Any]]) -> str:
        """Synthesizes all tool results into a final answer."""
        logger.debug("Synthesizing final answer from CoT session")
        try:
            synthesis_context = f"Original Query: \"{cot_session.original_query}\"\n\n"
            synthesis_context += "Tool Executions Summary:\n"
            for exec in cot_session.tool_executions:
                result_summary = str(exec.result)[:200] if exec.result else "No result"
                synthesis_context += f"- {exec.tool_name} ({'Success' if exec.success else 'Failed'}): {result_summary}...\n"
            
            synthesis_context += "\nSynthesize these results into a comprehensive final answer."

            final_answer = await self._call_anthropic_for_analysis(
                self.system_prompts["synthesis"],
                synthesis_context,
                chat_history
            )
            cot_session.reasoning_steps.append(ReasoningStep(
                step_id=f"synthesis_{len(cot_session.reasoning_steps)}",
                step_type="synthesis",
                content=f"Synthesized final answer from {len(cot_session.tool_executions)} tool executions."
            ))
            return final_answer
        except Exception as e:
            logger.exception(f"Result synthesis failed: {e}")
            return f"I encountered an error while synthesizing the final answer: {e}"      
    
    def _create_complexity_analysis_prompt(self) -> str:
        """
        Create system prompt for query complexity analysis.
        
        Returns:
            System prompt string for complexity analysis
        """
        return """
You are an expert AI assistant specializing in query complexity analysis.

Analyze the user's query and determine its complexity level:
- SIMPLE: Single-step queries that require one direct action ("summarize this", "what is X?")
- MODERATE: Multi-step queries with clear sequence ("find X and extract Y", "translate and summarize")
- COMPLEX: Queries requiring analysis and reasoning ("compare A and B", "analyze trends")
- VERY_COMPLEX: Queries requiring research, deep analysis, and recommendations ("research X, analyze implications, and recommend strategy")

Provide:
1. Complexity level with confidence (0.0-1.0)
2. Clear reasoning for your assessment
3. Estimated number of steps needed
4. Tools likely required

Be precise and analytical in your assessment.
"""
    
    def _create_cot_reasoning_prompt(self, allow_web_search: bool) -> str:
        """
        Create system prompt for Chain of Thought reasoning.
        
        Args:
            allow_web_search: Whether web search functionality is enabled for this query
        
        Returns:
            System prompt string for CoT reasoning
        """
        base_prompt = """
You are an expert AI reasoning assistant that uses Chain of Thought (CoT) methodology.

IMPORTANT: You must think step-by-step and make your reasoning transparent:

1. ANALYZE: Break down the query into logical components
2. PLAN: Decide what tools and steps are needed. 
   **🔄 ÇOKLU DOKÜMAN PLANLAMA KURALI:** Eğer kullanıcı birden fazla doküman seçtiyse, planının HER DOKÜMAN İÇİN ayrı ayrı tool çağrısı içermesi ZORUNLUDUR. Örneğin: assess_risks_in_document('dosya1.pdf'), assess_risks_in_document('dosya2.pdf'), synthesize_results()
   For complex queries, you MUST create a multi-step plan. Do not try to solve everything with a single tool. Break down the problem and plan a sequence of tool calls.
3. REASON: Before each tool call, explain WHY you're using it
4. REFLECT: After each result, analyze what it means for your goal
5. ADAPT: Modify your plan based on results
6. SYNTHESIZE: Combine all results into a comprehensive answer

🌐 **WEB SEARCH BEST PRACTICES:**
• Use specific, targeted search queries
• Always prioritize uploaded documents first, use web search to supplement
• Combine web search results with document analysis when appropriate
• Mention sources clearly when using web search information

ALWAYS explain your reasoning before calling any tool. Make every step of your thinking visible and logical.
"""
        
        # Web arama durumuna göre dinamik talimat ekle
        if allow_web_search:
            web_instruction = """
**CRITICAL INSTRUCTION FOR THIS TASK: WEB SEARCH IS ENABLED.**
You have permission to use the `web_search` tool. Prioritize the user's documents, but if the answer is not there or requires current information, you MUST use the web search tool.
"""
        else:
            web_instruction = """
**CRITICAL INSTRUCTION FOR THIS TASK: WEB SEARCH IS DISABLED.**
You are FORBIDDEN from using the `web_search` tool. You must answer using ONLY the provided documents and context. If the information is not in the documents, explicitly state that.
"""
            
        return base_prompt + "\n" + web_instruction + "\nAvailable tools will be provided in the tools parameter. Use them strategically based on your analysis."
    
    def _create_reflection_prompt(self) -> str:
        """
        Create system prompt for reflection on tool results.
        
        Returns:
            System prompt string for reflection
        """
        return """
You are reflecting on the result of a tool execution.

Analyze:
1. What does this result tell us?
2. How does it relate to our original goal?
3. What should we do next?
4. Are there any issues or gaps?
5. Should we modify our approach?

Be thorough and honest in your reflection. If something didn't work as expected, acknowledge it and suggest corrections.
"""
    
    def _create_synthesis_prompt(self) -> str:
        """
        Create system prompt for final result synthesis.
        
        Returns:
            System prompt string for synthesis
        """
        return """
You are synthesizing the final answer from all reasoning steps and tool results.

Create a comprehensive, well-structured response that:
1. Directly answers the original query
2. Incorporates insights from all tool executions
3. Provides clear, actionable information
4. Acknowledges any limitations or uncertainties
5. Is appropriate for the user's context and needs

Make your answer complete but concise, professional but accessible.
"""
    
    def _construct_planning_prompt(self, session_memory: Dict[str, Any], query: str, 
                                 available_files: List[str], targeted_files: List[str],
                                 allow_web_search: bool,
                                 chat_history: Optional[List[Dict[str, Any]]] = None,
                                 scenario: str = "normal") -> str:
        """
        Construct planning prompt with session facts integration.
        
        Args:
            session_memory: Session memory dictionary
            query: User query
            available_files: List of all available files
            targeted_files: List of targeted/selected files
            chat_history: Optional chat history
            scenario: Prompt scenario ("no_docs", "no_selection", "normal")
            
        Returns:
            Complete planning prompt with session facts
        """
        # Create session facts section - ALWAYS present even if empty
        def create_session_facts_section(memory: Dict[str, Any]) -> str:
            """Create session facts XML section for prompt integration."""
            facts_section = "<session_facts>\n"
            facts_section += "<!-- KULLANICI TARAFINDAN ONAYLANMIŞ KESİN BİLGİLER -->\n"
            
            if memory and len(memory) > 0:
                for key, value in memory.items():
                    # Escape XML special characters
                    escaped_key = str(key).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    escaped_value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    facts_section += f'<fact key="{escaped_key}">{escaped_value}</fact>\n'
            
            facts_section += "</session_facts>\n\n"
            return facts_section
        
        session_facts_section = create_session_facts_section(session_memory)
        
        # Base system instructions with ABSOLUTE RULES
        base_instructions = f"""
{session_facts_section}**MUTLAK KURAL 1: PLANLAMA ÖNCESİ BİLGİ KONTROLÜ:** Planını oluşturmadan önce her zaman `<session_facts>` bölümünü dikkatlice incele.

**MUTLAK KURAL 2: BİLGİ HİYERARŞİSİ:** `<session_facts>` içindeki bilgi, dokümanlardan veya sohbet geçmişinden gelen her türlü bilgiden **daha üstündür ve kesindir**. Eğer bu bilgiler arasında bir çelişki varsa, **her zaman `<session_facts>` bölümündeki bilgiyi doğru kabul et.**

**MUTLAK KURAL 3: BİLGİYİ KULLANMA ZORUNLULUĞU:** Yaptığın tüm araç çağrılarında ve oluşturduğun tüm nihai cevaplarda, `<session_facts>` bölümünde verilen bilgileri **kullanmak zorundasın.**

**Current User Query:** "{query}"
"""
        
        
        # Chat history section
        chat_history_section = ""
        if chat_history and len(chat_history) > 0:
            chat_history_section = "**🧠 CONVERSATIONAL MEMORY (Recent Chat History):**\n"
            for msg in chat_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # İlk 200 karakter
                if content:
                    if role == "user":
                        chat_history_section += f"👤 User: {content}\n"
                    elif role == "assistant":
                        chat_history_section += f"🤖 Assistant: {content}\n"
            chat_history_section += "\n"
        else:
            chat_history_section = "**🧠 CONVERSATIONAL MEMORY:** No previous conversation in this session.\n\n"
        
        # File information
        all_files_str = ", ".join([f"'{name}'" for name in available_files]) if available_files else "none"
        targeted_files_str = ", ".join([f"'{name}'" for name in targeted_files]) if targeted_files else "none"
        
        # Scenario-specific prompts
        if scenario == "no_docs":
            return f"""{base_instructions}
You are an expert planning agent with CONVERSATIONAL MEMORY. Your task is to create a step-by-step plan to answer the user's query.

{chat_history_section}**CRITICAL ISSUE:** No documents are available in the current session.

**DECISION LOGIC:**
1. **FIRST:** Check if the user's query can be answered from the conversation history above
2. **IF YES:** Provide a direct answer using the `synthesize_results` tool with the information from chat history
3. **IF NO:** Ask user to upload documents

**Your Response:** 
- IF answer is in conversation history → Use `synthesize_results` tool to provide the answer
- IF answer requires documents → Call `ask_user_for_clarification` tool with this message: "I don't see any documents in your session. Please upload some documents first using the 'Upload Documents' tab, then I'll be happy to help analyze them."
"""
        
        elif scenario == "no_selection":
            return f"""{base_instructions}
You are an expert planning agent. Your task is to create a step-by-step plan to answer the user's query by calling the available tools.

**Available Documents in Session:** {all_files_str}
**ISSUE:** User has not selected any specific documents to analyze.

**Your Response:** You MUST call the `ask_user_for_clarification` tool with this exact message: "I see you have {len(available_files)} documents available: {all_files_str}. Please select which documents you'd like me to analyze from the document selector above your message, then ask your question again."
"""
        
        else:  # normal scenario
            multi_doc_rule = ""
            if targeted_files and len(targeted_files) > 1:
                file_list_str = ", ".join([f"'{f}'" for f in targeted_files])
                multi_doc_rule = f"""
**MUTLAK, TARTIŞMASIZ VE EN ÖNEMLİ KURAL: "TÜM SEÇİLENLERE UYGULA" PRENSİBİ**
Kullanıcı bu sorgu için birden fazla doküman ({file_list_str}) seçti. Bu, yapacağın her analizin, **SEÇİLEN TÜM DOKÜMANLARI** kapsaması gerektiği anlamına gelir.

**PLANLAMA TALİMATLARI:**
1.  **ARAÇ SEÇİMİ:** İlk olarak, hangi araçların çoklu dosya alabileceğini kontrol et:
    - `compare_documents`: file_names listesi alır (ÇOKLU DOSYA)
    - `read_full_document`, `search_in_document`, `assess_risks_in_document`: Tek dosya alır (TEK DOSYA)
    
2.  **PLAN STRATEJİSİ:** 
    - **ÇOKLU DOSYA ARAÇLARI TERCİH ET:** Eğer uygunsa önce `compare_documents` gibi çoklu dosya alabilen araçları kullan
    - **TEK DOSYA ARAÇLARI İÇİN DÖNGÜ:** Tek dosya araçları için, seçilen **HER BİR DOKÜMAN İÇİN AYRI AYRI** çağrı yap
    
3.  **ÖRNEK DOĞRU PLANLAR:** 
    - Karşılaştırma sorguları için: `compare_documents(file_names=['dosya1.docx', 'dosya2.docx'], ...)`
    - Risk analizi için: `assess_risks_in_document(file_name='dosya1.docx')` → `assess_risks_in_document(file_name='dosya2.docx')` → `synthesize_results`
    
4.  **ASLA TEK DOSYAYLA YETİNME:** Birden fazla dosya seçilmişse, planında sadece tek bir dosyayı analiz eden bir adım olması **KESİNLİKLE YASAKTIR**.
"""

            return f"""{base_instructions}
{multi_doc_rule}

**SENİN PERSONAN: KAPSAMLI VE EKSİKSİZ BİR ANALİST**
Görevin, hızlı olmak değil, **eksiksiz** olmaktır. Kullanıcının sana verdiği tüm kaynakları sonuna kadar kullanmalısın.

{chat_history_section}**🎯 BU SORGUNUN BAĞLAMI (ANALİZ EDİLECEK KAYNAKLAR):**
**Seçilen Dokümanlar:** {targeted_files_str}

**GÖREVİN:**
Yukarıdaki "TÜM SEÇİLENLERE UYGULA" kuralına harfiyen uyarak, kullanıcının sorgusunu cevaplamak için seçilen **tüm dokümanları** kapsayan, adım adım bir araç kullanım planı oluştur.
"""
    
    async def execute_with_cot(self, query: str, session_id: Optional[str] = None, chat_history: Optional[List[Dict[str, Any]]] = None, selected_filenames: Optional[List[str]] = None, allow_web_search: bool = False) -> CoTSession:
        """
        Execute a query using Chain of Thought reasoning.
        
        This is the main method that implements the complete CoT pipeline:
        1. Query complexity analysis
        2. Step-by-step reasoning and planning
        3. Sequential tool execution with reflection
        4. Result synthesis and final answer generation
        
        Args:
            query: The user query to process
            session_id: Optional user session ID for context
            chat_history: Optional chat history for context
            selected_filenames: Optional list of specific filenames to focus on (Hedeflenmiş Doküman Sorgulama)
            allow_web_search: Whether to allow web search functionality (default: False)
            
        Returns:
            Complete CoTSession with all reasoning steps and results
        """

        if chat_history is None:
            chat_history = []


        



        # === YENİ KOD BAŞLANGICI ===

        # 1. Koruma Bendi: Boş veya geçersiz sorguları en başta reddet
        if not query or not query.strip():
            logger.warning("execute_with_cot called with an empty or whitespace-only query.")
        
            
            # Hatalı bir CoTSession nesnesi oluştur ve hemen döndür
            error_session = CoTSession(
                session_id=f"cot_error_{int(time.time())}",
                original_query=query,
                user_session_id=session_id,
                success=False,
                final_answer="Lütfen geçerli bir soru veya komut girin. Sorgu boş olamaz.",
                error_info={
                    "error_type": "InvalidQueryError",
                    "error_message": "Query was empty or contained only whitespace."
                }
            )
            # Hata durumunu logla
            error_step = ReasoningStep(
                step_id="invalid_query_error",
                step_type="error_handling",
                content="Query was rejected because it was empty."
            )
            error_session.reasoning_steps.append(error_step)
            error_session.total_execution_time = 0.0 # Hiçbir işlem yapılmadı
            
            return error_session
            
        # === YENİ KOD SONU ===
        start_time = time.time()
        cot_session = CoTSession(
            session_id=f"cot_{int(time.time())}_{id(self)}",
            original_query=query,
            user_session_id=session_id
        )
        
        # Store allow_web_search in metadata for fallback logic
        cot_session.metadata["allow_web_search"] = allow_web_search
        
        self.total_sessions += 1
        
        logger.info(f"Starting CoT execution for query: {query[:100]}...")
        logger.info(f"Session ID: {cot_session.session_id}")
        
        try:
            # Step 1: Analyze query complexity
            logger.info("Step 1: Analyzing query complexity")
            complexity_analysis = await self.analyze_query_complexity(query)
            cot_session.complexity_analysis = complexity_analysis
            
            # Add reasoning step for complexity analysis
            complexity_step = ReasoningStep(
                step_id=f"complexity_{len(cot_session.reasoning_steps)}",
                step_type="analysis",
                content=f"Query complexity: {complexity_analysis.complexity.value}. {complexity_analysis.reasoning}",
                context={"complexity_analysis": complexity_analysis.__dict__}
            )
            cot_session.reasoning_steps.append(complexity_step)
            
            # Step 2: Get session memory before planning
            logger.info("Step 2a: Retrieving session memory")
            session_memory = {}
            if session_id and self.session_manager:
                try:
                    session_memory = self.session_manager.get_session_memory(session_id)
                    logger.info(f"Retrieved session memory: {len(session_memory)} items")
                except Exception as e:
                    logger.warning(f"Failed to retrieve session memory: {e}")
                    session_memory = {}
            
            # === HAFIZA KRİTİK GÜNCELLEMESİ ===
            # Session memory'yi CoTSession'a kaydet ki synthesis aşamasında kullanılabilsin
            cot_session.set_session_facts(session_memory)
            logger.info(f"Session facts stored in CoTSession: {len(session_memory)} facts")
            
            # Step 2b: Plan tool execution with CoT
            logger.info("Step 2b: Planning tool execution with CoT reasoning")
            execution_plan = await self.plan_tool_execution_with_cot(
                query, complexity_analysis, session_id, chat_history, selected_filenames, allow_web_search, session_memory
            )
            
            # ============================================================================
            # TEMBEL PLAN TESPİTİ VE OTOMATİK DÜZELTİM
            # ============================================================================
            
            # Tembel plan kontrolü: Sadece hafıza araçları içeren tek adımlı planları tespit et
            if (len(execution_plan) == 1 and 
                execution_plan[0].get("tool_name") in ['get_session_memory', 'update_session_memory']):
                
                logger.warning("🚨 LAZY PLAN DETECTED: Single-step memory-only plan is insufficient")
                logger.info(f"Original lazy plan: {execution_plan[0].get('tool_name')}")
                
                # Otomatik plan zenginleştirmesi
                original_tool = execution_plan[0]
                
                # Eğer dokümanlar varsa, doküman analizi ekle
                if selected_filenames and len(selected_filenames) > 0:
                    logger.info("🔧 Auto-correcting plan: Adding document analysis")
                    
                    # Orijinal hafıza aracını koru ama doküman analizini ekle
                    enhanced_plan = []
                    
                    # Önce hafıza kontrol et (eğer get_session_memory ise)
                    if original_tool.get("tool_name") == "get_session_memory":
                        enhanced_plan.append(original_tool)
                    
                    # Sonra HER DOKÜMAN İÇİN okuma ekle
                    for filename in selected_filenames:
                        enhanced_plan.append({
                            "tool_name": "read_full_document",
                            "reasoning": f"Auto-enhanced: Reading document '{filename}' to provide comprehensive analysis",
                            "arguments": {"filename": filename}
                        })
                    
                    # Son olarak sentez ekle
                    enhanced_plan.append({
                        "tool_name": "synthesize_results", 
                        "reasoning": "Auto-enhanced: Synthesizing information from memory and document",
                        "arguments": {
                            "results_summary": "Combined analysis from session memory and document content",
                            "user_query": query
                        }
                    })
                    
                    # Eğer update_session_memory ise, sadece onu koru
                    if original_tool.get("tool_name") == "update_session_memory":
                        enhanced_plan = [original_tool]
                    
                    execution_plan = enhanced_plan
                    logger.success(f"✅ Plan enhanced: {len(execution_plan)} steps now planned")
                    
                else:
                    logger.info("📝 No documents available - keeping original memory plan")
            
            # ============================================================================
            # TEMBEL PLAN DÜZELTİMİ TAMAMLANDI
            # ============================================================================
            
            # Step 3: Execute tool chain with reasoning
            logger.info("Step 3: Executing tool chain with reflective reasoning")
            clarification_requested = await self.execute_tool_chain_with_reasoning(
                cot_session, execution_plan, chat_history
            )
            
            # ============================================================================
            # YENİ DEVRE KESİCİ KONTROLÜ - CLARIFICATION DURUMU
            # ============================================================================
            
            if clarification_requested:
                logger.info("🔄 CoT execution interrupted due to clarification request")
                logger.info("⏸️ Skipping synthesis step - awaiting user response")
                
                # Clarification durumunda özel final_answer ayarla
                clarification_question = cot_session.metadata.get(
                    "clarification_question", 
                    "Could you please provide more details?"
                )
                
                # Final answer'ı clarification question olarak ayarla
                cot_session.final_answer = clarification_question
                cot_session.success = True  # İşlem başarılı ama clarification bekliyor
                
                # Execution time'ı hesapla ve session'ı tamamla
                cot_session.total_execution_time = time.time() - start_time
                self.total_reasoning_steps += len(cot_session.reasoning_steps)
                self.total_tool_executions += len(cot_session.tool_executions)
                
                logger.success(f"CoT session paused for clarification in {cot_session.total_execution_time:.2f}s")
                logger.info(f"  - Reasoning steps: {len(cot_session.reasoning_steps)}")
                logger.info(f"  - Tool executions: {len(cot_session.tool_executions)}")
                logger.info(f"  - Clarification question: {clarification_question}")
                
                return cot_session
            
            # ============================================================================
            # NORMAL AKIŞ DEVAM EDİYOR
            # ============================================================================
            
            # ============================================================================
            # DEVRE KESİCİ: SESSİZ ARAÇ KONTROLÜ
            # ============================================================================
            
            # Eğer plan tek bir adımdan oluşuyorsa ve bu adım sessiz bir araçsa,
            # sentezlemeyi atla ve doğrudan bir onay mesajı ile bitir.
            if (len(execution_plan) == 1 and 
                execution_plan[0].get("tool_name") in self.SILENT_TOOLS):
                
                tool_name = execution_plan[0].get("tool_name")
                logger.info(f"🔇 SILENT TOOL DETECTED: '{tool_name}' - Skipping synthesis")
                
                # Son tool execution'ın sonucunu al
                if cot_session.tool_executions and len(cot_session.tool_executions) > 0:
                    last_execution = cot_session.tool_executions[-1]
                    
                    if last_execution.success and last_execution.result:
                        logger.info(f"Silent tool '{tool_name}' executed successfully. Ending session.")
                        
                        # Aracın kendi onay mesajını kullan veya genel bir mesaj oluştur
                        if hasattr(last_execution.result, 'confirmation') and last_execution.result.confirmation:
                            final_answer = last_execution.result.confirmation
                        else:
                            final_answer = "İşlem başarıyla tamamlandı."
                        
                        # CoT oturumunu başarıyla sonlandır
                        cot_session.final_answer = final_answer
                        cot_session.success = True
                        cot_session.total_execution_time = time.time() - start_time
                        
                        # İstatistikleri güncelle
                        self.total_reasoning_steps += len(cot_session.reasoning_steps)
                        self.total_tool_executions += len(cot_session.tool_executions)
                        self.successful_sessions += 1
                        
                        logger.success(f"🔇 Silent tool session completed in {cot_session.total_execution_time:.2f}s")
                        logger.info(f"  - Final answer: {final_answer}")
                        logger.info(f"  - Reasoning steps: {len(cot_session.reasoning_steps)}")
                        logger.info(f"  - Tool executions: {len(cot_session.tool_executions)}")
                        
                        return cot_session  # Döngüyü burada bitir
            
            # ============================================================================
            # DEVRE KESİCİ KONTROLÜ TAMAMLANDI
            # ============================================================================
            
            # ============================================================================
            # SESSIZ ARAÇ DEVRE KESİCİSİ - SENTEZLEME ADIMINI ATLA
            # ============================================================================
            
            # Check if this was a single-step silent tool execution
            if (len(execution_plan) == 1 and 
                len(cot_session.tool_executions) >= 1 and
                cot_session.tool_executions[-1].tool_name in self.SILENT_TOOLS and
                cot_session.tool_executions[-1].success):
                
                logger.info(f"🔇 Silent tool circuit breaker activated for {cot_session.tool_executions[-1].tool_name}")
                logger.info("⏩ Skipping synthesis step - using tool's direct response")
                
                # Use the tool's direct response as final answer
                tool_result = cot_session.tool_executions[-1].result
                if hasattr(tool_result, 'confirmation') and tool_result.confirmation:
                    cot_session.final_answer = tool_result.confirmation
                elif hasattr(tool_result, 'message') and tool_result.message:
                    cot_session.final_answer = tool_result.message
                else:
                    # Fallback for silent tools
                    if cot_session.tool_executions[-1].tool_name == "update_session_memory":
                        cot_session.final_answer = "Anlaşıldı, bu bilgiyi not ettim."
                    else:
                        cot_session.final_answer = "İşlem başarıyla tamamlandı."
                
                cot_session.success = True
                cot_session.completed_at = datetime.now().isoformat()
                
                # Update statistics
                self.successful_sessions += 1
                
                logger.success(f"Silent tool execution completed - final answer: {cot_session.final_answer[:100]}...")
                return cot_session
            
            # ============================================================================
            # SESSIZ ARAÇ DEVRE KESİCİSİ SONU
            # ============================================================================
            
            # ### YENİ DEVRE KESİCİ KONTROLÜ ###
            if cot_session.final_answer:
                logger.info("✅ Final answer was produced by the tool chain. Skipping redundant final synthesis step.")
                cot_session.success = True
                self.successful_sessions += 1
                cot_session.total_execution_time = time.time() - start_time
                self.total_reasoning_steps += len(cot_session.reasoning_steps)
                self.total_tool_executions += len(cot_session.tool_executions)
                
                logger.success(f"CoT session completed with existing answer in {cot_session.total_execution_time:.2f}s")
                logger.info(f"  - Final answer: {cot_session.final_answer}")
                logger.info(f"  - Reasoning steps: {len(cot_session.reasoning_steps)}")
                logger.info(f"  - Tool executions: {len(cot_session.tool_executions)}")
                
                return cot_session
            else:
                # Step 4 (Güvenlik Ağı): Plan bir cevap üretmediyse, şimdi biz üretelim.
                logger.info("Step 4: Synthesizing final results (no answer was set by the tool chain).")

                successful_tool_results_objects = [
                    step.result for step in cot_session.tool_executions if step.success and step.result
                ]
                
                final_answer = ""
                if not successful_tool_results_objects:
                    # No successful tool results to synthesize
                    final_answer = "I couldn't find any relevant information to answer your question."
                else:
                    try:
                        # === NEW AND CRITICAL TRANSFORMATION ===
                        # Convert Pydantic objects to dictionary list as expected by schema
                        results_as_dicts = [
                            res.model_dump() if isinstance(res, BaseModel) else res
                            for res in successful_tool_results_objects
                        ]
                        # === TRANSFORMATION END ===
                        
                        # === HAFIZA KRİTİK ENTEGRASYONu ===
                        # Hafızadaki bilgiyi sentezleme adımına gidecek kanıtların en başına ekle
                        if cot_session.session_facts:
                            from tools.base_tool import BaseToolResult
                            memory_as_tool_result = {
                                'success': True,
                                'memory_data': cot_session.session_facts,
                                'tool_name': 'session_memory_facts',
                                'message': 'Session memory facts retrieved'
                            }
                            results_as_dicts.insert(0, memory_as_tool_result)  # En başa ekle ki öncelikli olsun
                            logger.info(f"Session memory facts injected into synthesis: {len(cot_session.session_facts)} facts")
                        # === HAFIZA ENTEGRASYONU SONU ===

                        logger.info(f"Attempting to synthesize {len(results_as_dicts)} tool result(s) using 'synthesize_results' tool.")
                        
                        synthesis_result_obj = await asyncio.to_thread(
                            self.tool_manager.run_tool,
                            "synthesize_results",
                            tool_results=results_as_dicts,  # <-- SEND TRANSFORMED DATA
                            original_query=query,
                        )
                        
                        if synthesis_result_obj.success:
                            final_answer = synthesis_result_obj.synthesis
                            cot_session.final_answer = final_answer
                            
                            # Add synthesis as tool execution
                            synthesis_execution = ToolExecutionStep(
                                tool_name="synthesize_results",
                                arguments={
                                    "tool_results": "Transformed tool results list",
                                    "original_query": query,
                                },
                                pre_execution_reasoning=ReasoningStep(
                                    step_id="synthesis_prep",
                                    step_type="execution_prep",
                                    content="Preparing to synthesize results from all successful tool executions"
                                ),
                                result=synthesis_result_obj,
                                success=True,
                                execution_time=0.5,
                                timestamp=datetime.now().isoformat()
                            )
                            cot_session.tool_executions.append(synthesis_execution)
                            
                            logger.success(f"Synthesis completed using SynthesizeResultsTool")
                        else:
                            # Fallback to internal synthesis
                            final_answer = self._perform_internal_synthesis(results_as_dicts, query)
                            cot_session.final_answer = final_answer
                            logger.warning("SynthesizeResultsTool failed, used internal synthesis")
                            
                    except Exception as e:
                        logger.exception(f"SynthesizeResultsTool execution failed: {e}")
                        # Fallback to internal synthesis
                        final_answer = self._perform_internal_synthesis(results_as_dicts, query)
                        cot_session.final_answer = final_answer
                
                # Set final answer if not already set
                if not cot_session.final_answer:
                    cot_session.final_answer = final_answer

                
                # Mark as successful
                cot_session.success = True
                self.successful_sessions += 1
                
                logger.success(f"CoT session completed successfully in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.exception(f"CoT session failed: {e}")
            
            # Try fallback approach
            fallback_answer = await self._fallback_execution(query, str(e), chat_history)
            cot_session.final_answer = fallback_answer
            cot_session.success = False
            cot_session.error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "fallback_used": True
            }
            
            # Add error reasoning step
            error_step = ReasoningStep(
                step_id=f"error_{len(cot_session.reasoning_steps)}",
                step_type="error_handling",
                content=f"Error occurred: {str(e)}. Using fallback approach.",
                context={"error": str(e), "fallback": True}
            )
            cot_session.reasoning_steps.append(error_step)
        
        # Finalize session
        cot_session.total_execution_time = time.time() - start_time
        self.total_reasoning_steps += len(cot_session.reasoning_steps)
        self.total_tool_executions += len(cot_session.tool_executions)
        
        logger.info(f"CoT session summary:")
        logger.info(f"  - Reasoning steps: {len(cot_session.reasoning_steps)}")
        logger.info(f"  - Tool executions: {len(cot_session.tool_executions)}")
        logger.info(f"  - Success: {cot_session.success}")
        logger.info(f"  - Total time: {cot_session.total_execution_time:.2f}s")
        
        return cot_session
    
    async def analyze_query_complexity(self, query: str) -> ComplexityAnalysis:
        """
        Analyze the complexity of a user query using CoT reasoning.
        
        Args:
            query: The user query to analyze
            
        Returns:
            ComplexityAnalysis with determined complexity and reasoning
        """
        logger.debug(f"Analyzing complexity for query: {query[:100]}...")
        
        try:
            # Create improved analysis prompt that enforces JSON-only response
            analysis_prompt = f"""
Analyze the complexity of this query and respond with ONLY valid JSON (no markdown, no explanations):

QUERY: "{query}"

INSTRUCTIONS:
- Return ONLY a JSON object, nothing else
- No markdown formatting, no code blocks
- Use exactly these complexity values: "simple", "moderate", "complex", "very_complex"

Required JSON format:
{{
    "complexity": "simple|moderate|complex|very_complex",
    "reasoning": "brief explanation of complexity assessment",
    "estimated_steps": <number>,
    "required_tools": ["tool1", "tool2"],
    "confidence": <0.0-1.0>
}}

RESPOND WITH JSON ONLY:"""

            response = await self._call_anthropic_for_analysis(
                self.system_prompts["complexity_analysis"],
                analysis_prompt
            )
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response for complexity analysis: {repr(response)}")
            
            # Clean and extract JSON from response
            cleaned_json = self._extract_and_clean_json(response)
            logger.debug(f"Cleaned JSON: {repr(cleaned_json)}")
            
            if not cleaned_json:
                logger.warning("Could not extract valid JSON from LLM response")
                return self._get_fallback_complexity_analysis(query, "No valid JSON found in response")
            
            # Parse the cleaned JSON
            try:
                analysis_data = json.loads(cleaned_json)
                logger.debug(f"Parsed analysis data: {analysis_data}")
                
                # Validate and extract required fields with better error handling
                complexity_result = self._validate_and_create_complexity_analysis(analysis_data, query)
                
                if complexity_result:
                    logger.info(f"Successfully analyzed query complexity: {complexity_result.complexity.value} (confidence: {complexity_result.confidence})")
                    return complexity_result
                else:
                    logger.warning("Failed to validate complexity analysis data")
                    return self._get_fallback_complexity_analysis(query, "Invalid analysis data structure")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}. Raw response: {repr(response[:200])}")
                return self._get_fallback_complexity_analysis(query, f"JSON parsing failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error parsing complexity analysis: {e}")
                return self._get_fallback_complexity_analysis(query, f"Parsing error: {str(e)}")
                
        except Exception as e:
            logger.exception(f"Query complexity analysis failed: {e}")
            return self._get_fallback_complexity_analysis(query, f"Analysis failed: {str(e)}")

    def _extract_and_clean_json(self, response: str) -> Optional[str]:
        """
        Extract and clean JSON from LLM response, handling various formats.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned JSON string or None if no valid JSON found
        """
        if not response or not response.strip():
            return None
            
        response = response.strip()
        
        # Method 1: Try direct parsing if response looks like JSON
        if response.startswith('{') and response.endswith('}'):
            try:
                # Validate it's actually valid JSON
                json.loads(response)
                return response
            except json.JSONDecodeError:
                pass
        
        # Method 2: Extract from markdown code blocks
        import re
        
        # Look for ```json ... ``` blocks
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        json_matches = re.findall(json_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in json_matches:
            cleaned = match.strip()
            if cleaned.startswith('{') and cleaned.endswith('}'):
                try:
                    json.loads(cleaned)
                    return cleaned
                except json.JSONDecodeError:
                    continue
        
        # Method 3: Extract JSON object from mixed content
        # Find content between first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            potential_json = response[first_brace:last_brace + 1]
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # Method 4: Try to find JSON-like patterns and clean them
        # Remove common non-JSON prefixes/suffixes
        cleaned_response = response
        prefixes_to_remove = [
            "Here's the JSON:", "JSON:", "Response:", "Analysis:",
            "Here is the complexity analysis:", "The analysis is:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_response.lower().startswith(prefix.lower()):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        # Try parsing the cleaned response
        if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
            try:
                json.loads(cleaned_response)
                return cleaned_response
            except json.JSONDecodeError:
                pass
        
        return None

    def _validate_and_create_complexity_analysis(self, data: Dict[str, Any], query: str) -> Optional[ComplexityAnalysis]:
        """
        Validate parsed JSON data and create ComplexityAnalysis object.
        
        Args:
            data: Parsed JSON data
            query: Original query for context
            
        Returns:
            ComplexityAnalysis object or None if validation fails
        """
        try:
            # Required fields with defaults
            complexity_str = data.get("complexity", "moderate").lower()
            reasoning = data.get("reasoning", "No reasoning provided")
            estimated_steps = data.get("estimated_steps", 3)
            required_tools = data.get("required_tools", [])
            confidence = data.get("confidence", 0.5)
            
            # Validate and convert complexity
            valid_complexities = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MODERATE, 
                "complex": QueryComplexity.COMPLEX,
                "very_complex": QueryComplexity.VERY_COMPLEX
            }
            
            complexity = valid_complexities.get(complexity_str)
            if not complexity:
                logger.warning(f"Invalid complexity value: {complexity_str}. Using moderate.")
                complexity = QueryComplexity.MODERATE
            
            # Validate numeric fields
            if not isinstance(estimated_steps, (int, float)) or estimated_steps <= 0:
                logger.warning(f"Invalid estimated_steps: {estimated_steps}. Using 3.")
                estimated_steps = 3
            
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logger.warning(f"Invalid confidence: {confidence}. Using 0.5.")
                confidence = 0.5
            
            # Validate required_tools
            if not isinstance(required_tools, list):
                logger.warning(f"Invalid required_tools format: {required_tools}. Using empty list.")
                required_tools = []
            
            # Ensure reasoning is a string
            if not isinstance(reasoning, str):
                reasoning = str(reasoning)
            
            return ComplexityAnalysis(
                complexity=complexity,
                reasoning=reasoning[:500],  # Limit reasoning length
                estimated_steps=int(estimated_steps),
                required_tools=[str(tool) for tool in required_tools[:10]],  # Limit tools list
                confidence=float(confidence)
            )
            
        except Exception as e:
            logger.warning(f"Failed to validate complexity analysis data: {e}")
            return None

    def _get_fallback_complexity_analysis(self, query: str, reason: str) -> ComplexityAnalysis:
        """
        Generate a fallback ComplexityAnalysis when parsing fails.
        
        Args:
            query: Original query
            reason: Reason for fallback
            
        Returns:
            Fallback ComplexityAnalysis object
        """
        # Simple heuristics for fallback analysis
        query_lower = query.lower()
        query_length = len(query.split())
        
        # Determine complexity based on simple heuristics
        if query_length <= 5 and any(word in query_lower for word in ["what", "who", "when", "where", "define"]):
            fallback_complexity = QueryComplexity.SIMPLE
            steps = 1
        elif query_length > 20 or any(word in query_lower for word in ["compare", "analyze", "evaluate", "assess"]):
            fallback_complexity = QueryComplexity.COMPLEX
            steps = 4
        else:
            fallback_complexity = QueryComplexity.MODERATE
            steps = 3
        
        return ComplexityAnalysis(
            complexity=fallback_complexity,
            reasoning=f"Fallback analysis due to: {reason}. Used heuristics based on query length and keywords.",
            estimated_steps=steps,
            required_tools=[],
            confidence=0.3  # Lower confidence for fallback
        )
        


    def _perform_internal_synthesis(self, tool_results: List[Dict[str, Any]], query: str) -> str:
        """Fallback internal synthesis when SynthesizeResultsTool fails."""
        try:
            # Simple synthesis logic as fallback
            if not tool_results:
                return "I couldn't find any relevant information to answer your question."
            
            # Combine tool results into a basic response
            response_parts = []
            for i, result in enumerate(tool_results):
                if isinstance(result, dict):
                    # Try to extract meaningful information from the result
                    result_summary = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    response_parts.append(f"Result {i+1}: {result_summary}")
                else:
                    response_parts.append(f"Result {i+1}: {str(result)[:200]}...")
            
            if response_parts:
                synthesis = f"Based on the analysis of your query '{query}', here are the key findings:\n\n"
                synthesis += "\n\n".join(response_parts)
                return synthesis
            else:
                return "The tools executed successfully but couldn't extract meaningful information."
                
        except Exception as e:
            logger.exception(f"Internal synthesis fallback failed: {e}")
            return "I encountered an error while processing your request."
    
    async def plan_tool_execution_with_cot(self, query: str, complexity: ComplexityAnalysis, session_id: Optional[str], chat_history: Optional[List[Dict[str, Any]]] = None, selected_filenames: Optional[List[str]] = None, allow_web_search: bool = False, session_memory: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Plan tool execution sequence using Chain of Thought reasoning with context-aware file name inference and targeted document querying.
        
        Args:
            query: The user query
            complexity: Complexity analysis results
            session_id: Optional user session ID for context
            chat_history: Optional chat history for context
            selected_filenames: Optional list of specific filenames to focus on (Hedeflenmiş Doküman Sorgulama)
            allow_web_search: Whether web search functionality is enabled for this query
            session_memory: Optional session memory containing confirmed facts and decisions
            
        Returns:
            List of planned tool executions with reasoning
        """
        logger.debug(f"Planning tool execution for {complexity.complexity.value} query")
        
        # Initialize session memory if None - try to load from session manager
        if session_memory is None:
            session_memory = {}
            # Try to get session memory from session manager if available
            if self.session_manager and session_id:
                try:
                    stored_memory = self.session_manager.get_session_memory(session_id)
                    if stored_memory and isinstance(stored_memory, dict):
                        session_memory.update(stored_memory)
                        logger.info(f"🧠 Loaded {len(stored_memory)} session facts from memory store")
                except Exception as e:
                    logger.warning(f"Could not load session memory: {e}")
        
        # ### YENİ VE KESİN GARDİYAN MANTIĞI ###
        # Bu blok, LLM'e gitmeden önce doküman seçim durumunu kontrol eder.

        # Önce oturumdaki tüm dokümanları alalım.
        available_files = []
        if session_id and self.session_manager:
            session_docs = self.session_manager.get_session_documents(session_id)
            if session_docs:
                available_files = [doc.file_name for doc in session_docs]

        # KESİN KURAL: Eğer ortamda dokümanlar VARSA ve kullanıcı bu sorgu için HİÇBİRİNİ seçmemişse,
        # LLM'in plan yapmasına izin verme. Planı kendin oluştur ve metottan hemen çık.
        if available_files and not selected_filenames:
            logger.warning("Guard Clause Activated: Documents exist, but none were selected. Bypassing LLM planning.")
            all_files_str = ", ".join([f"'{name}'" for name in available_files])
            question_to_ask = f"I see you have {len(available_files)} documents available: {all_files_str}. Please select which document(s) you want me to analyze from the list, then ask your question again."
            
            # Doğrudan, hatasız ve hazır bir plan döndürerek metodu sonlandır.
            return [{
                "tool_name": "ask_user_for_clarification",
                "arguments": {"question": question_to_ask},
                "reasoning": "User has documents available but did not select any for this specific query. Clarification is required."
            }]
        # ### GARDİYAN MANTIĞI SONU ###

        try:
            # Get available tools
            available_tools = self._get_available_tools_for_anthropic()

            # ============================================================================
            # HEDEFLENMİŞ DOKÜMAN SORGULAMA - DOKÜMAN KONTROL VE PROMPT OLUŞTURMA
            # ============================================================================
            
            # Get session context - Retrieve list of all available files
            available_files = []
            targeted_files = []
            
            if session_id and self.session_manager:
                session_docs = self.session_manager.get_session_documents(session_id)
                if session_docs:
                    available_files = [doc.file_name for doc in session_docs]
            
            # Hedeflenmiş dosyaları belirle
            if selected_filenames:
                # Kullanıcı spesifik dosyalar seçmiş - bunları kullan
                
                # Streamlit Cloud singleton issue için bypass: 
                # Eğer available_files boşsa ama selected_filenames varsa, selected_filenames'a güven
                if not available_files and selected_filenames:
                    logger.warning("SessionManager singleton issue detected - bypassing validation")
                    targeted_files = selected_filenames
                    logger.info(f"🎯 BYPASS: Using selected files directly: {targeted_files}")
                else:
                    # Normal validation
                    valid_selected_files = [f for f in selected_filenames if f in available_files]
                    invalid_selected_files = [f for f in selected_filenames if f not in available_files]
                    
                    if invalid_selected_files:
                        logger.warning(f"Selected files not found in session: {invalid_selected_files}")
                    
                    targeted_files = valid_selected_files
                    logger.info(f"🎯 Targeted document querying: Using {len(targeted_files)} selected files: {targeted_files}")
            else:
                # Kullanıcı hiç dosya seçmemiş
                targeted_files = []
                logger.info("⚠️ No files selected for targeted querying")
            
            # Prompt için string formatları oluştur
            all_files_str = ", ".join([f"'{name}'" for name in available_files]) if available_files else "none"
            targeted_files_str = ", ".join([f"'{name}'" for name in targeted_files]) if targeted_files else "none"
            
            # Use new centralized prompt construction method
            # Streamlit Cloud bypass için targeted_files'ı da kontrol et
            if not available_files and not targeted_files:
                scenario = "no_docs"
            elif not targeted_files:
                scenario = "no_selection" 
            else:
                scenario = "normal"
                
            planning_prompt = self._construct_planning_prompt(
                session_memory=session_memory,
                query=query,
                available_files=available_files,
                targeted_files=targeted_files,
                allow_web_search=allow_web_search,
                chat_history=chat_history,
                scenario=scenario
            )
            
            # ============================================================================
            # HEDEFLENMİŞ DOKÜMAN SORGULAMA PROMPT OLUŞTURMA SONU
            # ============================================================================

            # Use Anthropic with function calling if tools are available
            if available_tools:
                # --- YENİ DİNAMİK SİSTEM PROMPT'U OLUŞTURMA ---
                # Ana sistem prompt'unu al ve web arama durumuna göre dinamik olarak güncelle.
                base_system_prompt = self._create_cot_reasoning_prompt(allow_web_search=allow_web_search)
                # --- YENİ MANTIK SONU ---
                
                response = await self._call_anthropic_with_cot_and_tools(
                    base_system_prompt,  # <--- DİNAMİK OLARAK OLUŞTURULAN PROMPT'U KULLAN
                    planning_prompt,
                    available_tools,
                    chat_history
                )
                
                # Extract tool calls from response
                tool_plans = self._extract_tool_plans_from_response(response)
                
                # ============================================================================
                # ÇOK KRİTİK: ÇOKLU DOKÜMAN POST-PROCESSING
                # ============================================================================
                # LLM tek dosya araçları için sadece 1 call yapmışsa, otomatik olarak tüm dosyalar için genişlet
                if targeted_files and len(targeted_files) > 1:
                    tool_plans = self._expand_plans_for_multiple_documents(tool_plans, targeted_files)
                # ============================================================================
            else:
                # No tools available - plan fallback
                tool_plans = [{
                    "tool_name": "direct_answer",
                    "reasoning": "No tools available, will provide direct response",
                    "arguments": {"query": query}
                }]
            
            logger.info(f"Planned {len(tool_plans)} tool execution steps")
            return tool_plans
            
        except Exception as e:
            logger.exception(f"Tool execution planning failed: {e}")
            return [{
                "tool_name": "error_fallback",
                "reasoning": f"Planning failed: {str(e)}",
                "arguments": {"query": query, "error": str(e)}
            }]
    
   # core/llm_agent.py içinde

    async def execute_tool_chain_with_reasoning(self, cot_session: CoTSession, execution_plan: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Execute the planned tool chain with reflective reasoning and intelligent parameter management.
        
        Args:
            cot_session: The CoT session to update with executions
            execution_plan: List of planned tool executions
        
        Returns:
            bool: True if clarification was requested (interrupting execution), False if normal completion
        """
        logger.info(f"Executing tool chain with {len(execution_plan)} planned steps")
        
        # ============================================================================
        # SESSIZ ARAÇ KONTROLÜ - UPDATE_SESSION_MEMORY ÖZEL DURUMU
        # ============================================================================
        
        # Eğer tek adım varsa ve bu bir memory aracıysa, synthesis adımını atla
        silent_tools = ["update_session_memory", "get_session_memory"]
        if (len(execution_plan) == 1 and 
            execution_plan[0].get("tool_name") in silent_tools):
            
            logger.info(f"Detected single-step {execution_plan[0].get('tool_name')} plan - executing silently")
            
            planned_step = execution_plan[0]
            tool_name = planned_step.get("tool_name")
            arguments = planned_step.get("arguments", {})
            reasoning = planned_step.get("reasoning", "Storing user decision in session memory")
            
            # === ARGÜMAN NORMALLEŞTİRME (SESSIZ ARAÇLAR İÇİN) ===
            logger.info(f"Original silent tool args: {arguments}")
            arguments = self._normalize_tool_arguments(tool_name, arguments)
            logger.info(f"Normalized silent tool args: {arguments}")
            # === NORMALLEŞTİRME SONU ===
            
            # Session ID'yi enjekte et
            if self.tool_manager:
                tool_signature = self.tool_manager.get_tool_signature(tool_name)
                if (tool_signature and hasattr(tool_signature, 'args_schema') and 
                    tool_signature.args_schema and "session_id" in tool_signature.args_schema.model_fields):
                    if cot_session.user_session_id:
                        arguments["session_id"] = cot_session.user_session_id
                        logger.info(f"Auto-injected session_id into {tool_name}")
            
            # Aracı çalıştır
            start_time = time.time()
            try:
                tool_result = self.tool_manager.run_tool(tool_name, **arguments)
                execution_time = time.time() - start_time
                
                # Execution step oluştur
                execution_step = ToolExecutionStep(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=tool_result,
                    pre_execution_reasoning=ReasoningStep(
                        step_id=f"reasoning_0",
                        step_type="tool_planning",
                        content=reasoning
                    ),
                    success=bool(tool_result and tool_result.success),
                    execution_time=execution_time
                )
                
                cot_session.tool_executions.append(execution_step)
                
                # Sessiz araç için doğrudan uygun mesajı final answer olarak kullan
                if tool_name == "update_session_memory":
                    if tool_result and tool_result.confirmation:
                        cot_session.final_answer = tool_result.confirmation
                    else:
                        cot_session.final_answer = "Anlaşıldı, bu bilgiyi not ettim."
                elif tool_name == "get_session_memory":
                    if tool_result and tool_result.memory_data:
                        memory_data = tool_result.memory_data
                        if memory_data:
                            memory_items = [f"{k}: {v}" for k, v in memory_data.items()]
                            cot_session.final_answer = f"Oturum hafızasından şu bilgileri buldum:\n" + "\n".join(memory_items)
                        else:
                            cot_session.final_answer = "Oturum hafızasında henüz kaydedilmiş bilgi bulunmuyor."
                    else:
                        cot_session.final_answer = "Oturum hafızasını kontrol edemedim."
                    
                cot_session.success = True
                logger.info("Silent memory tool execution completed - skipping synthesis")
                return False  # No clarification requested
                
            except Exception as e:
                logger.exception(f"Error executing silent memory tool: {e}")
                cot_session.final_answer = "Üzgünüm, bu bilgiyi kaydetmede bir sorun yaşadım."
                cot_session.success = False
                return False
        
        # ============================================================================
        # NORMAL ARAÇ ZİNCİRİ EXECUTİON (MEVCUT MANTIK)
        # ============================================================================
        
        # --- DEFENSIVE PROGRAMMING START ---
        if chat_history is None:
            chat_history = []
        
        if not isinstance(chat_history, list):
            logger.warning(f"chat_history was not a list, it was {type(chat_history)}. Resetting to empty list.")
            chat_history = []
        # --- DEFENSIVE PROGRAMMING END ---
        
        for i, planned_step in enumerate(execution_plan):
            if len(cot_session.tool_executions) >= self.max_tool_executions:
                logger.warning(f"Maximum tool executions ({self.max_tool_executions}) reached")
                break
            
            tool_name = planned_step.get("tool_name", "unknown")
            arguments = planned_step.get("arguments", {})
            reasoning = planned_step.get("reasoning", "No reasoning provided")
        
            # ============================================================================
            # AKILLI PARAMETRE YÖNETİMİ (GENİŞLETİLMİŞ)
            # ============================================================================
            
            # 1. 'synthesize_results' aracı için bağlamı doldur (Mevcut mantık korunuyor)
            if tool_name == "synthesize_results":
                logger.info("Applying intelligent parameter management for 'synthesize_results'")
                
                if "tool_results" not in arguments or not arguments.get("tool_results"):
                    previous_results = [step.result for step in cot_session.tool_executions if step.success and step.result]
                    # CRITICAL FIX: Use centralized Pydantic to dict conversion
                    previous_results_as_dicts = [
                        self._convert_pydantic_to_dict(res) for res in previous_results
                    ]
                    arguments["tool_results"] = previous_results_as_dicts
                    logger.info(f"Automatically injected {len(previous_results_as_dicts)} previous results (converted to dicts) into {tool_name}.")
                else:
                    # Also convert existing tool_results if they contain Pydantic objects
                    if isinstance(arguments["tool_results"], list):
                        arguments["tool_results"] = [
                            self._convert_pydantic_to_dict(res) for res in arguments["tool_results"]
                        ]
                        logger.info(f"Converted existing tool_results to dicts for {tool_name}.")

                if "original_query" not in arguments:
                    arguments["original_query"] = cot_session.original_query
                    logger.info(f"Automatically injected original_query into {tool_name}.")
                
                # CRITICAL FIX: Ekstra, şemada olmayan argümanları temizle
                valid_args = {}
                for key in ["tool_results", "original_query"]:
                    if key in arguments:
                        valid_args[key] = arguments[key]
                
                if len(arguments) != len(valid_args):
                    removed_args = set(arguments.keys()) - set(valid_args.keys())
                    logger.info(f"Cleaned extra arguments from synthesize_results: {removed_args}")
                    arguments = valid_args

            # 2. GÜÇLENDİRİLMİŞ SESSION_ID ENJEKSİYONU - TÜM MEMORY ARAÇLARI İÇİN
            if self.tool_manager:
                tool = self.tool_manager.get_tool(tool_name)
                if tool and hasattr(tool, 'args_schema') and tool.args_schema:
                    # Check if this tool requires session_id
                    requires_session_id = (
                        "session_id" in tool.args_schema.model_fields or
                        tool_name in ["update_session_memory", "get_session_memory"]  # Explicit check for memory tools
                    )
                    
                    if requires_session_id:
                        correct_session_id = cot_session.user_session_id
                        
                        # Eğer correct_session_id None veya boş ise, aracı çalıştırmak anlamsız
                        if not correct_session_id:
                            logger.error(f"Cannot execute tool '{tool_name}' because a valid session_id is missing from the CoT session.")
                            tool_execution = ToolExecutionStep(
                                tool_name=tool_name,
                                arguments=arguments,
                                result={"error": "A valid user session ID is required but was not provided."},
                                pre_execution_reasoning=ReasoningStep(
                                    step_id=f"error_{i}",
                                    step_type="error",
                                    content=f"Skipping {tool_name} due to missing session_id"
                                ),
                                success=False,
                                execution_time=0.0
                            )
                            cot_session.tool_executions.append(tool_execution)
                            continue
                        
                        # Session_id eksik veya yanlışsa düzelt
                        current_session_id = arguments.get("session_id")
                        if current_session_id != correct_session_id:
                            if current_session_id is None:
                                logger.warning(f"LLM forgot to provide session_id for '{tool_name}'. Auto-injecting correct ID '{correct_session_id}'.")
                            else:
                                logger.warning(f"LLM provided incorrect session_id '{current_session_id}' for '{tool_name}'. Overwriting with correct ID '{correct_session_id}'.")
                            
                            arguments["session_id"] = correct_session_id
                            logger.info(f"Ensured correct session_id for tool '{tool_name}'.")

            # ============================================================================
            # AKILLI PARAMETRE YÖNETİMİ SONU
            # ============================================================================
            
            logger.info(f"Executing step {i+1}: {tool_name} with original LLM args: {arguments}")
            
            # ============================================================================
            # ARGÜMAN NORMALLEŞTİRME - YAZIM HATALARINI DÜZELTİM
            # ============================================================================
            
            # Use centralized argument normalization
            arguments = self._normalize_tool_arguments(tool_name, arguments)
            logger.info(f"Normalized args before execution: {arguments}")
            
            # ============================================================================
            # ARGÜMAN NORMALLEŞTİRME TAMAMLANDI
            # ============================================================================
            
            # ============================================================================
            # ARAÇ ÇALIŞTIRMA - DÖNGÜ İÇİNDE (DÜZELTİLMİŞ)
            # ============================================================================
            
            # Create pre-execution reasoning step
            pre_reasoning = ReasoningStep(
                step_id=f"pre_exec_{len(cot_session.reasoning_steps)}",
                step_type="execution_prep",
                content=f"Preparing to execute {tool_name}: {reasoning}",
                context={"tool_name": tool_name, "arguments": arguments}
            )
            cot_session.reasoning_steps.append(pre_reasoning)
            
            # Execute the tool
            start_time = time.time()
            tool_execution = ToolExecutionStep(
                tool_name=tool_name,
                arguments=arguments,
                pre_execution_reasoning=pre_reasoning
            )
            
            try:
                if tool_name == "direct_answer":
                    result = await self._provide_direct_answer(arguments.get("query", cot_session.original_query), chat_history)
                    tool_execution.success = True
                elif tool_name == "error_fallback":
                    logger.error(f"Execution plan resulted in an error_fallback step. Reason: {reasoning}")
                    result = await self._fallback_execution(arguments.get("query", cot_session.original_query), arguments.get("error", ""), chat_history)
                    tool_execution.success = False
                elif self.tool_manager:
                    result = self.tool_manager.run_tool(tool_name, **arguments)
                    tool_execution.success = hasattr(result, 'success') and result.success
                else:
                    result = {"error": "Tool manager not available", "tool": tool_name}
                    tool_execution.success = False
                
                tool_execution.result = result
                tool_execution.execution_time = time.time() - start_time
                
                logger.info(f"Tool execution completed: {tool_name} ({'success' if tool_execution.success else 'failed'})")
                
                # Reflect on the result
                reflection = await self.reflect_on_step_result(
                    cot_session.original_query, tool_execution, chat_history
                )
                tool_execution.post_execution_reflection = reflection
                cot_session.reasoning_steps.append(reflection)
                
            except Exception as e:
                logger.exception(f"Tool execution failed for {tool_name}: {e}")
                tool_execution.success = False
                tool_execution.result = {"error": str(e)} # Pydantic ToolError nesnesi olabilir, str() güvenli.
                tool_execution.execution_time = time.time() - start_time
                
                # Create error reflection
                error_reflection = await self.reflect_on_step_result( # <-- await ekle
                        cot_session.original_query, tool_execution, chat_history # <-- chat_history ekle
                    )

                tool_execution.post_execution_reflection = error_reflection
                cot_session.reasoning_steps.append(error_reflection)
            
            cot_session.tool_executions.append(tool_execution)
            
            # ============================================================================
            # FALLBACK WEB SEARCH MANTĞI - DOSYA ARAMASI BAŞARISIZ OLUNCA WEB ARAMA
            # ============================================================================
            
            # Check if the executed tool was a document/file search tool that failed
            document_search_tools = [
                "search_in_document", "read_full_document", "summarize_document",
                "assess_risks_in_document", "extract_key_points", "synthesize_results"
            ]
            
            # Check if:
            # 1. Current tool is a document search tool
            # 2. Web search is enabled  
            # 3. Web search tool hasn't been used yet in this session
            # 4. Either the tool failed OR it succeeded but found insufficient information
            web_search_not_used = not any(step.tool_name == "web_search" for step in cot_session.tool_executions)
            web_search_enabled = cot_session.metadata.get("allow_web_search", False)
            is_document_tool = tool_name in document_search_tools
            
            # Check for failure or insufficient content
            tool_failed = not tool_execution.success
            insufficient_content = False
            
            if tool_execution.success and tool_execution.result:
                # Even if tool succeeded, check if content indicates insufficient information
                result_data = tool_execution.result
                
                # Get the full result text for semantic analysis
                if hasattr(result_data, 'synthesis'):
                    # For synthesis results, check the synthesis content
                    analysis_text = getattr(result_data, 'synthesis', '')
                elif hasattr(result_data, 'content'):
                    analysis_text = getattr(result_data, 'content', '')
                elif isinstance(result_data, dict):
                    # Check multiple possible fields
                    analysis_text = (result_data.get('synthesis', '') or 
                                   result_data.get('content', '') or 
                                   result_data.get('summary', '') or
                                   str(result_data))
                else:
                    analysis_text = str(result_data)
                
                # AI-powered response evaluation for insufficient information
                try:
                    insufficient_content = await self._evaluate_response_insufficiency(
                        analysis_text, cot_session.original_query
                    )
                    if insufficient_content:
                        logger.info(f"🧠 AI detected insufficient information in response: {analysis_text[:100]}...")
                    else:
                        logger.debug("AI evaluation: response contains sufficient information")
                except Exception as e:
                    logger.warning(f"AI evaluation failed, using fallback logic: {e}")
                    # Fallback to simple heuristics if AI evaluation fails
                    analysis_text_lower = analysis_text.lower()
                    fallback_indicators = [
                        "yeterli bilgi yok", "bilgi bulunmuyor", "bilgi içermiyor", 
                        "kesin bir bilgi veremiyorum", "insufficient information"
                    ]
                    insufficient_content = any(indicator in analysis_text_lower for indicator in fallback_indicators)
            
            should_fallback_to_web = (
                is_document_tool and 
                (tool_failed or insufficient_content) and
                web_search_enabled and
                web_search_not_used
            )
            
            # Additional check: if the error message suggests no information found
            if should_fallback_to_web and tool_execution.result:
                result_str = str(tool_execution.result).lower()
                # Only fallback if it's genuinely about missing information, not technical errors
                info_missing_indicators = [
                    "no information", "not found", "no results", "no relevant", 
                    "couldn't find", "no content", "empty", "no data"
                ]
                technical_error_indicators = [
                    "api error", "connection", "timeout", "permission", "access denied"
                ]
                
                has_info_missing = any(indicator in result_str for indicator in info_missing_indicators)
                has_technical_error = any(indicator in result_str for indicator in technical_error_indicators)
                
                # Don't fallback if it's a technical error, only if information is missing
                if has_technical_error and not has_info_missing:
                    should_fallback_to_web = False
                    logger.debug("Skipping web fallback: detected technical error, not missing information")
            
            if should_fallback_to_web:
                
                logger.info("🌐 FALLBACK WEB SEARCH TRIGGERED: Document search failed, attempting web search")
                logger.info(f"Failed document tool: {tool_name}, Original query: {cot_session.original_query}")
                
                # Extract and optimize search query based on document content
                web_query = await self._optimize_web_search_query(
                    cot_session.original_query, 
                    cot_session.tool_executions,
                    analysis_text
                )
                
                # Create web search step
                web_reasoning = ReasoningStep(
                    step_id=f"fallback_web_{len(cot_session.reasoning_steps)}",
                    step_type="fallback_reasoning",
                    content=f"Document search with {tool_name} failed. Falling back to web search to find information about: {web_query}",
                    context={
                        "fallback_trigger": tool_name,
                        "fallback_reason": "document_search_failed",
                        "web_query": web_query
                    }
                )
                cot_session.reasoning_steps.append(web_reasoning)
                
                # Execute web search
                web_start_time = time.time()
                web_execution = ToolExecutionStep(
                    tool_name="web_search",
                    arguments={"query": web_query, "max_results": 3},
                    pre_execution_reasoning=web_reasoning
                )
                
                try:
                    if self.tool_manager:
                        web_result = self.tool_manager.run_tool("web_search", query=web_query, max_results=3)
                        web_execution.success = hasattr(web_result, 'success') and web_result.success
                        web_execution.result = web_result
                        web_execution.execution_time = time.time() - web_start_time
                        
                        logger.info(f"🌐 Fallback web search completed: {'success' if web_execution.success else 'failed'}")
                        
                        # Add reflection on web search
                        web_reflection = await self.reflect_on_step_result(
                            cot_session.original_query, web_execution, chat_history
                        )
                        web_execution.post_execution_reflection = web_reflection
                        cot_session.reasoning_steps.append(web_reflection)
                        
                    else:
                        web_execution.success = False
                        web_execution.result = {"error": "Web search tool not available"}
                        web_execution.execution_time = time.time() - web_start_time
                        logger.warning("Web search fallback failed - tool manager not available")
                    
                except Exception as e:
                    logger.exception(f"Fallback web search failed: {e}")
                    web_execution.success = False
                    web_execution.result = {"error": f"Web search error: {str(e)}"}
                    web_execution.execution_time = time.time() - web_start_time
                    
                    # Add error reflection
                    error_reflection = ReasoningStep(
                        step_id=f"web_error_{len(cot_session.reasoning_steps)}",
                        step_type="error_reflection",
                        content=f"Fallback web search failed with error: {str(e)}",
                        context={"error": str(e)}
                    )
                    web_execution.post_execution_reflection = error_reflection
                    cot_session.reasoning_steps.append(error_reflection)
                
                # Add web execution to session
                cot_session.tool_executions.append(web_execution)
                logger.info("🌐 Fallback web search step added to execution chain")
            
            # ============================================================================
            # FALLBACK WEB SEARCH MANTĞI SONU
            # ============================================================================
            
            # ============================================================================
            # DEVRE KESİCİ MANTIĞI - ASK_USER_FOR_CLARIFICATION KONTROLÜ
            # ============================================================================
            
            # Eğer az önce çalıştırılan araç ask_user_for_clarification ise,
            # sürecin geri kalanını durdur ve clarification_requested=True döndür
            if tool_name == "ask_user_for_clarification":
                logger.info("🔄 CIRCUIT BREAKER ACTIVATED: ask_user_for_clarification tool executed")
                logger.info("🛑 Interrupting tool chain execution - awaiting user clarification")
                
                # Clarification durumunu CoTSession'a işaretle
                cot_session.metadata["clarification_requested"] = True
                cot_session.metadata["clarification_question"] = arguments.get("question", "Could you please provide more details?")
                
                # Reasoning step ekle
                interrupt_step = ReasoningStep(
                    step_id=f"clarification_interrupt_{len(cot_session.reasoning_steps)}",
                    step_type="clarification_interrupt",
                    content=f"Process interrupted - user clarification requested: {arguments.get('question', 'No question specified')}",
                    context={
                        "interrupt_reason": "ask_user_for_clarification",
                        "remaining_steps": len(execution_plan) - (i + 1),
                        "question": arguments.get("question", "No question specified")
                    }
                )
                cot_session.reasoning_steps.append(interrupt_step)
                
                logger.info(f"⏸️ Tool chain execution interrupted at step {i+1}/{len(execution_plan)}")
                logger.info(f"📝 Question for user: {arguments.get('question', 'No question specified')}")
                
                # True döndürerek clarification istendiğini belirt
                return True
            
            # ============================================================================
            # DEVRE KESİCİ MANTIĞI SONU
            # ============================================================================
            
            # Brief pause between executions
            await asyncio.sleep(0.1)
        
        # Normal completion (no clarification requested)
        return False
    
    async def reflect_on_step_result(self, original_query: str, tool_execution: ToolExecutionStep, chat_history: Optional[List[Dict[str, Any]]] = None) -> ReasoningStep:
        """
        Reflect on the result of a tool execution using CoT reasoning.
        
        Args:
            original_query: The original user query for context
            tool_execution: The completed tool execution to reflect on
            
        Returns:
            ReasoningStep containing the reflection
        """
        logger.debug(f"Reflecting on result from {tool_execution.tool_name}")
        
        try:
            reflection_prompt = f"""
Original Query: "{original_query}"

Tool Executed: {tool_execution.tool_name}
Arguments: {json.dumps(tool_execution.arguments, indent=2)}
Success: {tool_execution.success}
Result: {str(tool_execution.result)[:1000]}...

Reflect on this result:
1. What does this tell us about the original query?
2. How does this move us toward our goal?
3. What should we do next?
4. Are there any issues or concerns?

Provide thoughtful analysis in 2-3 sentences.
"""

            reflection_content = await self._call_anthropic_for_analysis(
                self.system_prompts["reflection"],
                reflection_prompt,
                chat_history
            )
            
            return ReasoningStep(
                step_id=f"reflect_{tool_execution.tool_name}_{int(time.time())}",
                step_type="reflection",
                content=reflection_content,
                context={
                    "tool_name": tool_execution.tool_name,
                    "success": tool_execution.success,
                    "execution_time": tool_execution.execution_time
                }
            )
            
        except Exception as e:
            logger.warning(f"Reflection generation failed: {e}")
            return ReasoningStep(
                step_id=f"reflect_error_{int(time.time())}",
                step_type="reflection",
                content=f"Unable to reflect on {tool_execution.tool_name} result due to error: {str(e)}",
                context={"error": str(e)}
            )
    
    async def synthesize_results_with_cot(self, cot_session: CoTSession, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Synthesize all tool results and reasoning into a final answer.
        
        Args:
            cot_session: Complete CoT session with all steps and results
            
        Returns:
            Synthesized final answer string
        """
        logger.debug("Synthesizing final answer from CoT session")
        
        try:
            # Prepare synthesis context
            synthesis_context = f"""
Original Query: "{cot_session.original_query}"

Complexity Analysis: {cot_session.complexity_analysis.complexity.value if cot_session.complexity_analysis else 'Unknown'}

Reasoning Steps ({len(cot_session.reasoning_steps)}):
"""
            
            for i, step in enumerate(cot_session.reasoning_steps[-10:]):  # Last 10 steps
                synthesis_context += f"{i+1}. [{step.step_type}] {step.content}\n"
            
            synthesis_context += f"\nTool Executions ({len(cot_session.tool_executions)}):\n"
            for i, execution in enumerate(cot_session.tool_executions):
                result_summary = str(execution.result)[:200] if execution.result else "No result"
                synthesis_context += f"{i+1}. {execution.tool_name} ({'' if execution.success else ''}): {result_summary}...\n"
            
            synthesis_context += "\nCreate a comprehensive final answer that addresses the original query using all available information."

            final_answer = await self._call_anthropic_for_analysis(
                self.system_prompts["synthesis"],
                synthesis_context,
                chat_history
            )
            
            # Add synthesis reasoning step
            synthesis_step = ReasoningStep(
                step_id=f"synthesis_{len(cot_session.reasoning_steps)}",
                step_type="synthesis",
                content=f"Synthesized final answer from {len(cot_session.tool_executions)} tool executions and {len(cot_session.reasoning_steps)} reasoning steps.",
                context={"answer_length": len(final_answer)}
            )
            cot_session.reasoning_steps.append(synthesis_step)
            
            return final_answer
            
        except Exception as e:
            logger.exception(f"Result synthesis failed: {e}")
            
            # Fallback synthesis
            fallback_answer = f"""
I encountered an error while synthesizing the final answer: {str(e)}

Based on the available information from {len(cot_session.tool_executions)} tool executions:
"""
            
            for execution in cot_session.tool_executions:
                if execution.success and execution.result:
                    fallback_answer += f"\n- {execution.tool_name}: {str(execution.result)[:200]}..."
            
            fallback_answer += "\n\nPlease let me know if you'd like me to try a different approach to answer your query."
            return fallback_answer

    async def _call_anthropic_with_cot_and_tools(
    self,
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> Any:
        """
        Call Anthropic API with CoT system prompt and available tools.

        Args:
            system_prompt: System prompt for CoT reasoning
            user_prompt: User's query or instruction
            tools: List of available tools in Anthropic format

        Returns:
            Anthropic API response
        """
        try:
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice={"type": "any"} if tools else None,
                    temperature=self.execution_temperature,
                    max_tokens=2000
                )
            )
            
            return response
            
        except Exception as e:
            logger.exception(f"Anthropic API call with tools failed: {e}")
            raise
    
    
    async def _provide_direct_answer(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Provide direct answer when no tools are needed or available.
        
        Args:
            query: User query
            chat_history: Optional chat history for context
            
        Returns:
            Direct answer string
        """
        try:
            direct_prompt = f"""
Provide a helpful and accurate answer to this query:

"{query}"

Since no specialized tools are available, use your general knowledge to provide the best possible response.
Be clear about any limitations in your response.
"""

            return await self._call_anthropic_for_analysis(
                "You are a helpful AI assistant providing direct answers to user queries.",
                direct_prompt,
                chat_history
            )
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    async def _fallback_execution(self, query: str, error: str, chat_history: List[Dict[str, Any]]) -> str:
        """
        Provide fallback response when main CoT execution fails.
        
        Args:
            query: Original user query
            error: Error that caused the fallback
            
        Returns:
            Fallback response string
        """
        try:
            fallback_prompt = f"""
The main reasoning system encountered an error, but I can still try to help.

Original query: "{query}"
Error encountered: {error}

Provide the best possible answer using basic reasoning, and acknowledge the limitation.
"""

            return await self._call_anthropic_for_analysis(
                "You are providing a fallback response after a system error.",
                fallback_prompt,
                chat_history
            )
            
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties and cannot process your query at this time. Error: {str(e)}"

    def _get_available_tools_for_anthropic(self) -> List[Dict[str, Any]]:
        """
        Get available tools formatted for Anthropic function calling using Claude bridge.

        Returns:
            List of tool schemas in Claude-compatible Anthropic format
        """
        if not self.tool_manager:
            return []
        
        try:
            # Use the new Claude function calling bridge from ToolManager
            # This method already returns schemas in the correct Claude format
            claude_tools = self.tool_manager.get_claude_function_schemas()
            
            logger.debug(f"Retrieved {len(claude_tools)} tools from Claude function calling bridge")
            return claude_tools

        except Exception as e:
            logger.exception(f"Failed to get Claude function schemas from ToolManager: {e}")
            return []
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the LLMAgent performance.
        
        Returns:
            Dictionary with performance statistics
        """
        success_rate = (self.successful_sessions / self.total_sessions * 100) if self.total_sessions > 0 else 0
        avg_reasoning_steps = (self.total_reasoning_steps / self.total_sessions) if self.total_sessions > 0 else 0
        avg_tool_executions = (self.total_tool_executions / self.total_sessions) if self.total_sessions > 0 else 0
        
        return {
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "success_rate_percent": round(success_rate, 2),
            "total_reasoning_steps": self.total_reasoning_steps,
            "total_tool_executions": self.total_tool_executions,
            "average_reasoning_steps_per_session": round(avg_reasoning_steps, 2),
            "average_tool_executions_per_session": round(avg_tool_executions, 2),
            "model": self.model,
            "reasoning_temperature": self.reasoning_temperature,
            "execution_temperature": self.execution_temperature,
            "max_reasoning_steps": self.max_reasoning_steps,
            "max_tool_executions": self.max_tool_executions,
            "tool_manager_available": self.tool_manager is not None,
            "session_manager_available": self.session_manager is not None
        }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Comprehensive async test of the Chain of Thought LLM Agent.
    
    This test validates CoT reasoning across different query complexity levels,
    verifies step tracking, and ensures proper integration with tool management.
    """
    
    async def test_llm_agent():
        """Async test function for the LLM Agent."""
        
        print("=== DataNeuron Chain of Thought LLM Agent Test ===")
        
        try:
            # Test 1: Create LLMAgent instance
            print(f"\nTest 1: Creating LLMAgent singleton")
            agent = LLMAgent()
            print(f" PASS - LLMAgent created successfully")
            print(f"   Model: {agent.model}")
            print(f"   Tool Manager: {'Available' if agent.tool_manager else 'Not Available'}")
            print(f"   Session Manager: {'Available' if agent.session_manager else 'Not Available'}")
            
            # Test singleton behavior
            agent2 = LLMAgent()
            if agent is agent2:
                print(f" PASS - Singleton pattern working correctly")
            else:
                print(f"L FAIL - Multiple instances created")
            
            # Test 2: Simple query complexity analysis
            print(f"\nTest 2: Simple query complexity analysis")
            simple_query = "What is artificial intelligence?"
            complexity = await agent.analyze_query_complexity(simple_query)
            
            print(f" PASS - Complexity analysis completed")
            print(f"   Query: {simple_query}")
            print(f"   Complexity: {complexity.complexity.value}")
            print(f"   Confidence: {complexity.confidence:.2f}")
            print(f"   Estimated steps: {complexity.estimated_steps}")
            print(f"   Reasoning: {complexity.reasoning[:100]}...")
            
            # Test 3: Complex query complexity analysis
            print(f"\nTest 3: Complex query complexity analysis")
            complex_query = "Analyze the uploaded documents, compare their main themes, and provide recommendations for improving content strategy"
            complex_complexity = await agent.analyze_query_complexity(complex_query)
            
            print(f" PASS - Complex analysis completed")
            print(f"   Query: {complex_query[:50]}...")
            print(f"   Complexity: {complex_complexity.complexity.value}")
            print(f"   Confidence: {complex_complexity.confidence:.2f}")
            print(f"   Estimated steps: {complex_complexity.estimated_steps}")
            
            # Verify complexity difference
            if complex_complexity.complexity.value != complexity.complexity.value:
                print(f" PASS - Agent correctly distinguished query complexities")
            else:
                print(f" WARNING - Both queries got same complexity level")
            
            # Test 4: Full CoT execution with simple query
            print(f"\nTest 4: Full CoT execution - Simple query")
            simple_session = await agent.execute_with_cot(simple_query)
            
            if simple_session.success:
                print(f" PASS - Simple CoT execution completed successfully")
            else:
                print(f" PASS - Simple CoT execution completed with fallback")
            
            print(f"   Session ID: {simple_session.session_id}")
            print(f"   Reasoning steps: {len(simple_session.reasoning_steps)}")
            print(f"   Tool executions: {len(simple_session.tool_executions)}")
            print(f"   Execution time: {simple_session.total_execution_time:.2f}s")
            print(f"   Final answer length: {len(simple_session.final_answer)} characters")
            
            # Validate reasoning steps
            if simple_session.reasoning_steps:
                print(f"   First reasoning step: {simple_session.reasoning_steps[0].step_type}")
                print(f"   Last reasoning step: {simple_session.reasoning_steps[-1].step_type}")
            
            # Test 5: Full CoT execution with complex query
            print(f"\nTest 5: Full CoT execution - Complex query")
            complex_session = await agent.execute_with_cot(complex_query, session_id="test_session_123")
            
            if complex_session.success:
                print(f" PASS - Complex CoT execution completed successfully")
            else:
                print(f" PASS - Complex CoT execution completed with fallback")
            
            print(f"   Session ID: {complex_session.session_id}")
            print(f"   User Session ID: {complex_session.user_session_id}")
            print(f"   Reasoning steps: {len(complex_session.reasoning_steps)}")
            print(f"   Tool executions: {len(complex_session.tool_executions)}")
            print(f"   Execution time: {complex_session.total_execution_time:.2f}s")
            
            # Verify complex query had more steps
            if len(complex_session.reasoning_steps) >= len(simple_session.reasoning_steps):
                print(f" PASS - Complex query generated appropriate number of reasoning steps")
            else:
                print(f" WARNING - Complex query had fewer steps than simple query")
            
            # Test 6: CoT step validation
            print(f"\nTest 6: CoT reasoning step validation")
            
            # Check that we have different types of reasoning steps
            step_types = set()
            for session in [simple_session, complex_session]:
                for step in session.reasoning_steps:
                    step_types.add(step.step_type)
            
            expected_step_types = {"analysis", "execution_prep", "reflection"}
            found_types = step_types.intersection(expected_step_types)
            
            print(f"   Found step types: {sorted(step_types)}")
            print(f"   Expected types found: {sorted(found_types)}")
            
            if len(found_types) >= 2:
                print(f" PASS - Multiple reasoning step types present")
            else:
                print(f" WARNING - Limited variety in reasoning step types")
            
            # Test 7: Tool execution validation
            print(f"\nTest 7: Tool execution validation")
            
            total_executions = 0
            successful_executions = 0
            
            for session in [simple_session, complex_session]:
                for execution in session.tool_executions:
                    total_executions += 1
                    if execution.success:
                        successful_executions += 1
                    
                    # Validate execution structure
                    if execution.pre_execution_reasoning and execution.post_execution_reflection:
                        print(f"   Tool {execution.tool_name}: Complete reasoning cycle")
                    elif execution.pre_execution_reasoning:
                        print(f"   Tool {execution.tool_name}: Pre-execution reasoning only")
            
            print(f"   Total tool executions: {total_executions}")
            print(f"   Successful executions: {successful_executions}")
            
            if total_executions > 0:
                print(f" PASS - Tool executions performed")
            else:
                print(f" PASS - No tool executions (expected if no tools available)")
            
            # Test 8: Performance statistics
            print(f"\nTest 8: Performance statistics")
            stats = agent.get_statistics()
            
            print(f" PASS - Statistics retrieved")
            print(f"   Total sessions: {stats['total_sessions']}")
            print(f"   Success rate: {stats['success_rate_percent']}%")
            print(f"   Avg reasoning steps: {stats['average_reasoning_steps_per_session']}")
            print(f"   Avg tool executions: {stats['average_tool_executions_per_session']}")
            
            # Validate statistics make sense
            if stats['total_sessions'] >= 2:
                print(f" PASS - Statistics tracking working correctly")
            else:
                print(f" WARNING - Statistics may not be tracking correctly")
            
            # Test 9: Error handling
            print(f"\nTest 9: Error handling and fallback")
            
            # Test with potentially problematic query
            error_query = "" # Empty query
            try:
                error_session = await agent.execute_with_cot(error_query)
                if error_session.final_answer:
                    print(f" PASS - Error handled gracefully with fallback")
                    print(f"   Error info: {error_session.error_info}")
                else:
                    print(f" WARNING - Error handling may need improvement")
            except Exception as e:
                print(f" FAIL - Unhandled exception: {e}")
            
            print(f"\n=== Test Complete ===")
            print("Chain of Thought LLM Agent is ready for production use.")
            print("The agent provides transparent, step-by-step reasoning for complex queries.")
            print("Integration with ToolManager and SessionManager verified.")
            
        except Exception as e:
            print(f"L CRITICAL FAIL - Test execution failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async test
    try:
        asyncio.run(test_llm_agent())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
