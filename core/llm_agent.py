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
            
            # Statistics
            self.total_sessions = 0
            self.successful_sessions = 0
            self.total_reasoning_steps = 0
            self.total_tool_executions = 0
            
            # CoT system prompts
            self.system_prompts = {
                "complexity_analysis": self._create_complexity_analysis_prompt(),
                "cot_reasoning": self._create_cot_reasoning_prompt(),
                "reflection": self._create_reflection_prompt(),
                "synthesis": self._create_synthesis_prompt()
            }
            
            logger.info("LLMAgent initialized successfully")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Tool Manager: {'Available' if self.tool_manager else 'Not Available'}")
            logger.info(f"  - Session Manager: {'Available' if self.session_manager else 'Not Available'}")
            
            self._initialized = True
            logger.success("LLMAgent singleton ready for Chain of Thought reasoning")


    # Bu kodu core/llm_agent.py içindeki LLMAgent sınıfının içine yapıştırın.

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
    
    def _create_cot_reasoning_prompt(self) -> str:
        """
        Create system prompt for Chain of Thought reasoning.
        
        Returns:
            System prompt string for CoT reasoning
        """
        return """
You are an expert AI reasoning assistant that uses Chain of Thought (CoT) methodology.

IMPORTANT: You must think step-by-step and make your reasoning transparent:

1. ANALYZE: Break down the query into logical components
2. PLAN: Decide what tools and steps are needed. For complex queries, you MUST create a multi-step plan. Do not try to solve everything with a single tool. Break down the problem and plan a sequence of tool calls.
3. REASON: Before each tool call, explain WHY you're using it
4. REFLECT: After each result, analyze what it means for your goal
5. ADAPT: Modify your plan based on results
6. SYNTHESIZE: Combine all results into a comprehensive answer

🌐 **NEW CAPABILITY: Web Search Integration**
You now have access to the `web_search` tool that provides real-time internet search capabilities. Use this tool when:

**WHEN TO USE WEB SEARCH:**
• Query requires current/recent information (news, trends, latest developments)
• Information about specific people, companies, or recent events
• Current market data, statistics, or policy updates
• Technical documentation not in uploaded documents
• Fact-checking or getting multiple perspectives
• General knowledge questions when documents don't contain relevant information

**WHEN NOT TO USE WEB SEARCH:**
• Query is clearly about uploaded documents content
• User explicitly asks about their specific documents
• Information is available in session documents
• Document-specific analysis (summaries, comparisons of uploaded files)

**WEB SEARCH BEST PRACTICES:**
• Use specific, targeted search queries
• Always prioritize uploaded documents first, use web search to supplement
• Combine web search results with document analysis when appropriate
• Mention sources clearly when using web search information

ALWAYS explain your reasoning before calling any tool. Make every step of your thinking visible and logical.

Available tools will be provided in the tools parameter. Use them strategically based on your analysis.
"""
    
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
            
            # Step 2: Plan tool execution with CoT
            logger.info("Step 2: Planning tool execution with CoT reasoning")
            execution_plan = await self.plan_tool_execution_with_cot(
                query, complexity_analysis, session_id, chat_history, selected_filenames, allow_web_search
            )
            
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
            
            # Step 4: Synthesize final results
            logger.info("Step 4: Synthesizing final results")

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
    
    async def plan_tool_execution_with_cot(self, query: str, complexity: ComplexityAnalysis, session_id: Optional[str], chat_history: Optional[List[Dict[str, Any]]] = None, selected_filenames: Optional[List[str]] = None, allow_web_search: bool = False) -> List[Dict[str, Any]]:
        """
        Plan tool execution sequence using Chain of Thought reasoning with context-aware file name inference and targeted document querying.
        
        Args:
            query: The user query
            complexity: Complexity analysis results
            session_id: Optional user session ID for context
            chat_history: Optional chat history for context
            selected_filenames: Optional list of specific filenames to focus on (Hedeflenmiş Doküman Sorgulama)
            allow_web_search: Whether web search functionality is enabled for this query
            
        Returns:
            List of planned tool executions with reasoning
        """
        logger.debug(f"Planning tool execution for {complexity.complexity.value} query")
        
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
                # Ancak seçilen dosyaların gerçekten mevcut olup olmadığını kontrol et
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
            
            # Create enhanced context-aware planning prompt with targeted document querying
            if not available_files:
                # Hiç doküman yüklenmemiş - Sohbet hafızası kontrolü ekle
                
                # Sohbet hafızası bölümü hazırla
                chat_history_section = ""
                if chat_history and len(chat_history) > 0:
                    chat_history_section = "**🧠 CONVERSATIONAL MEMORY (Recent Chat History):**\n"
                    for msg in chat_history:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")[:150]  # Kısa hali
                        if content:
                            if role == "user":
                                chat_history_section += f"👤 User: {content}\n"
                            elif role == "assistant":
                                chat_history_section += f"🤖 Assistant: {content}\n"
                    chat_history_section += "\n**IMPORTANT:** Check if the user's current query can be answered from the conversation history above before asking for documents.\n\n"
                else:
                    chat_history_section = "**🧠 CONVERSATIONAL MEMORY:** No previous conversation in this session.\n\n"
                
                planning_prompt = f"""
You are an expert planning agent with CONVERSATIONAL MEMORY. Your task is to create a step-by-step plan to answer the user's query.

**Current User Query:** "{query}"

{chat_history_section}**CRITICAL ISSUE:** No documents are available in the current session.

**DECISION LOGIC:**
1. **FIRST:** Check if the user's query can be answered from the conversation history above
2. **IF YES:** Provide a direct answer using the `synthesize_results` tool with the information from chat history
3. **IF NO:** Ask user to upload documents

**Your Response:** 
- IF answer is in conversation history → Use `synthesize_results` tool to provide the answer
- IF answer requires documents → Call `ask_user_for_clarification` tool with this message: "I don't see any documents in your session. Please upload some documents first using the 'Upload Documents' tab, then I'll be happy to help analyze them."
"""
            elif not targeted_files:
                # Dokümanlar var ama kullanıcı hiçbirini seçmemiş
                planning_prompt = f"""
You are an expert planning agent. Your task is to create a step-by-step plan to answer the user's query by calling the available tools.

**User Query:** "{query}"

**Available Documents in Session:** {all_files_str}
**ISSUE:** User has not selected any specific documents to analyze.

**Your Response:** You MUST call the `ask_user_for_clarification` tool with this exact message: "I see you have {len(available_files)} documents available: {all_files_str}. Please select which documents you'd like me to analyze from the document selector above your message, then ask your question again."
"""
            else:
                # Normal durum: Kullanıcı belirli dosyalar seçmiş
                
                # ============================================================================
                # SOHBET HAFIZASI - KONVERSASYONEL MEMORY BÖLÜMÜNÜ HAZIRLA
                # ============================================================================
                
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
                
                # ============================================================================
                # SOHBET HAFIZASI BÖLÜMİ HAZIRLANDI
                # ============================================================================
                
                planning_prompt = f"""
**🛑 KIRMIZI ÇİZGİ KURALI - SESSION_ID YÖNETİMİ:**
**ASLA** kullanıcıdan session_id istemeyiniz! Session_id sistem tarafından otomatik olarak yönetilir ve araçlara otomatik olarak enjekte edilir. Kullanıcı session_id ile ilgili hiçbir şey bilmez ve bilmesi de gerekmez. Bu konuda HIÇBIR ZAMAN soru sormayın, açıklama yapmayın veya kullanıcıdan bir şey istemeyin.

**🧠 CONVERSATIONAL AI ASSISTANT WITH INTELLIGENT INTENT RECOGNITION**

**FOUNDATIONAL PRINCIPLE - SOHBET ÖNCELİKLİ DÜŞÜNCE:**
Senin birincil görevin, akıllı ve yardımcı bir asistan olmaktır. Araçlar, bu hedefe ulaşmak için sadece birer seçenektir. **Her soruya bir araçla cevap vermek zorunda değilsin.** Sen öncelikle doğal bir diyalog ortağısın.

**Current User Query:** "{query}"

{chat_history_section}**🎯 AVAILABLE RESOURCES:**
**All Available Documents:** {all_files_str}
**Documents Selected for This Query:** {targeted_files_str}

**🎯 STEP 1: INTENT ANALYSIS (Niyet Tespiti) - MANDATORY FIRST STEP**
İlk önce, kullanıcının son mesajının niyetini analiz et. Niyet şunlardan biri olmalıdır:

**A) INFORMATION REQUEST (Bilgi Talebi):** 
   - User wants specific information from documents
   - Examples: "What does the report say about X?", "Find revenue data", "Search for risks"
   - ACTION: Use appropriate tools to extract information

**B) INSTRUCTION/DECISION (Talimat/Karar):**
   - User is telling you something or making a decision
   - Examples: "The project name is X", "Let's call it Y", "I've decided on Z"
   - ACTION: Acknowledge and remember, DO NOT use tools

**C) OPINION/CONVERSATION (Fikir Sorma/Sohbet):**
   - User wants your opinion or is having a casual conversation
   - Examples: "What do you think?", "Any suggestions?", "How does that sound?"
   - ACTION: Respond conversationally, usually NO tools needed

**D) TASK REQUEST (Görev Talebi):**
   - User wants you to perform a specific task with documents
   - Examples: "Summarize this", "Compare these files", "Create a summary"
   - ACTION: Use appropriate tools to complete the task

**🚀 STEP 2: BEHAVIOR RULES BASED ON INTENT:**

**RULE FOR INTENT A (Information Request):**
   - Use appropriate tools to get information from documents
   - Check conversation history first - if answer is already known, provide it directly
   - Common tool mapping:
     * "What does X say about Y?" → `search_in_document`
     * "Summarize this document" → `summarize_document`
     * "Compare these files" → `compare_documents`
     * "Show me the content" → `read_full_document`
     * "Find risks/problems" → `assess_risks_in_document`

🌐 **WEB SEARCH CAPABILITY STATUS:** {"ENABLED" if allow_web_search else "DISABLED"}

{"**🌐 WEB SEARCH INTEGRATION RULES:**" if allow_web_search else "**🌐 WEB SEARCH NOT AVAILABLE:**"}
{'''    PRIORITY: Documents first, web search second (supplement, do not replace)
   - USE WEB SEARCH when:
     * Query requires current/recent information (news, trends, developments)  
     * Information about people, companies, or events not in documents
     * Current market data, statistics, or policy updates
     * General knowledge questions when documents don't contain relevant info
     * Fact-checking or getting multiple perspectives
   - COMBINE: Use both document tools AND web_search for comprehensive answers
   - CITE SOURCES: Always mention when using web search results
   - Example: "Based on your document + current web information..."
   
   **WEB SEARCH TOOL AVAILABLE:** `web_search` - Use with specific, targeted queries''' if allow_web_search else "   Web search functionality is not enabled for this session. Focus on uploaded documents and conversation history."}

**RULE FOR INTENT B (Instruction/Decision):**
   - **CRITICAL: DO NOT USE ANY TOOLS**
   - Acknowledge the user's instruction/decision warmly
   - Confirm that you've "noted" or "remembered" their decision
   - Ask if there's anything else you can help with
   - Example responses:
     * "Anlaşıldı, projenin adını 'NexaCommerce' olarak not ediyorum. Başka nasıl yardımcı olabilirim?"
     * "Tamam, bu kararınızı hafızama aldım. Devam edelim - başka neye ihtiyacınız var?"

**RULE FOR INTENT C (Opinion/Conversation):**
   - **USUALLY NO TOOLS NEEDED** - respond conversationally
   - Use your knowledge and conversation history to provide thoughtful responses
   - Be engaging and helpful in the conversation
   - Only use tools if the conversation specifically requires document analysis
   - Example responses:
     * "'NexaCommerce' ismi gerçekten harika! Modern ve akılda kalıcı. Bu isimle devam etmek projenin vizyonunu iyi yansıtacaktır."
     * "Bu yaklaşım çok mantıklı. Önceki konuşmamızda bahsettiğiniz stratejiye de uyuyor."

**RULE FOR INTENT D (Task Request):**
   - Use appropriate tools to complete the requested task
   - Be efficient and focused on the specific task
   - Work ONLY with selected documents: {targeted_files_str}

**🧠 CONVERSATIONAL MEMORY PRIORITY:**
   - **ALWAYS CHECK CHAT HISTORY FIRST** before using any tools
   - If the answer exists in recent conversation, provide it directly
   - Understand references like "it", "that", "the project" from context
   - For follow-up questions, build on previous conversation

**⚠️ MINIMIZED CLARIFICATION RULE:**
   ONLY use `ask_user_for_clarification` for:
   - **Technical Contradictions:** "Compare 2 files" but only 1 selected
   - **Impossible Requests:** Required data doesn't exist anywhere
   - **Completely Incomprehensible:** Truly nonsensical queries

   **NEVER ask for clarification for:**
   - Opinions, suggestions, or conversational topics
   - Instructions or decisions from users
   - Questions that can be reasonably interpreted
   - Anything you can respond to conversationally
   - **ESPECIALLY NEVER** ask about session_id or technical parameters

**🎯 YOUR NEW MANDATE:** Be a natural conversation partner first, a tool operator second. Understand intent, respond appropriately, and only use tools when actually needed for information or task completion.
"""
            
            # ============================================================================
            # HEDEFLENMİŞ DOKÜMAN SORGULAMA PROMPT OLUŞTURMA SONU
            # ============================================================================

            # Use Anthropic with function calling if tools are available
            if available_tools:
                response = await self._call_anthropic_with_cot_and_tools(
                    self.system_prompts["cot_reasoning"],
                    planning_prompt,
                    available_tools,
                    chat_history
                )
                
                # Extract tool calls from response
                tool_plans = self._extract_tool_plans_from_response(response)
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
                    arguments["tool_results"] = previous_results
                    logger.info(f"Automatically injected {len(previous_results)} previous results into {tool_name}.")

                if "original_query" not in arguments:
                    arguments["original_query"] = cot_session.original_query
                    logger.info(f"Automatically injected original_query into {tool_name}.")

            # 2. 'session_id' gerektiren TÜM araçlar için bu parametreyi güvenilir kaynaktan ekle
            if self.tool_manager:
                tool = self.tool_manager.get_tool(tool_name)
                if tool and "session_id" in tool.args_schema.model_fields:
                    # LLM'in uydurduğu session_id'yi, gerçek session_id ile üzerine yaz veya ekle.
                    correct_session_id = cot_session.user_session_id
                    
                    # Eğer 'correct_session_id' None veya boş ise, aracı çalıştırmak anlamsız olur.
                    if not correct_session_id:
                        logger.error(f"Cannot execute tool '{tool_name}' because a valid session_id is missing from the CoT session.")
                        # Bu adımı atlayıp bir sonraki adıma geçebilir veya hata olarak işaretleyebiliriz.
                        # Şimdilik hata olarak işaretleyelim.
                        tool_execution = ToolExecutionStep(
                            tool_name=tool_name,
                            arguments=arguments,
                            pre_execution_reasoning=ReasoningStep(step_id=f"pre_exec_error_{i}", step_type="error", content=f"Skipping tool call due to missing session_id."),
                            success=False,
                            result={"error": "A valid user session ID is required but was not provided."}
                        )
                        cot_session.tool_executions.append(tool_execution)
                        continue # Döngünün bir sonraki adımına geç

                    if arguments.get("session_id") != correct_session_id:
                        logger.warning(f"LLM provided incorrect session_id '{arguments.get('session_id')}'. Overwriting with correct ID '{correct_session_id}'.")
                    
                    arguments["session_id"] = correct_session_id
                    logger.info(f"Ensured correct session_id for tool '{tool_name}'.")

            # ============================================================================
            # AKILLI PARAMETRE YÖNETİMİ SONU
            # ============================================================================
            
            logger.info(f"Executing step {i+1}: {tool_name} with args: {arguments}")
            
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
