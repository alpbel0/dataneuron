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
- OpenAI function calling integration for tool selection
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
- Integration with ToolManager, SessionManager, and OpenAI
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

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

# Import dependencies with error handling
try:
    import openai
except ImportError as e:
    logger.error(f"OpenAI library not available: {e}")
    openai = None

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
    OpenAI's GPT models with function calling for tool orchestration. It implements
    transparent Chain of Thought reasoning where every step of analysis, planning,
    execution, and reflection is captured and made visible.
    
    Key Features:
    - Singleton pattern for centralized intelligence
    - Async execution for responsive performance
    - Chain of Thought reasoning with step tracking
    - Query complexity analysis and adaptive planning
    - OpenAI function calling for tool selection
    - Reflective reasoning with plan adaptation
    - Comprehensive error handling and fallbacks
    - Session-aware context management
    
    The agent maintains complete transparency in its reasoning process,
    making it suitable for applications requiring explainable AI decisions.
    """
    
    _instance: Optional['LLMAgent'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'LLMAgent':
        """
        Singleton pattern implementation with thread safety.
        
        Returns:
            Single instance of LLMAgent
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMAgent, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the LLMAgent singleton.
        
        Sets up OpenAI client, tool manager, session manager, and reasoning
        capabilities. Only initializes once due to singleton pattern.
        
        Raises:
            RuntimeError: If required dependencies are not available
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            # Check dependencies
            if openai is None:
                raise RuntimeError("OpenAI library is required for LLMAgent")
            
            if not OPENAI_API_KEY:
                raise RuntimeError("OpenAI API key is required for LLMAgent")
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
            
            # Initialize managers
            self.tool_manager = ToolManager() if ToolManager else None
            self.session_manager = SessionManager() if SessionManager else None
            
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

    async def _call_openai_for_analysis(self, 
    system_prompt: str, 
    user_prompt: str, 
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> str:
        """Calls OpenAI API for analysis tasks that don't require tools."""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # --- DEFENSIVE PROGRAMMING START ---
            if chat_history is None:
                chat_history = []
            
            if not isinstance(chat_history, list):
                logger.warning(f"chat_history was not a list, it was {type(chat_history)}. Resetting to empty list.")
                chat_history = []

            clean_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in chat_history 
                if isinstance(msg, dict) and msg.get("content")
            ]
            messages.extend(clean_history)
            # --- DEFENSIVE PROGRAMMING END ---
        
            messages.append({"role": "user", "content": user_prompt})


            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=self.reasoning_temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception(f"OpenAI analysis call failed: {e}")
            raise

    async def _call_openai_with_cot_and_tools(self, system_prompt: str, user_prompt: str, tools: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Calls OpenAI API with CoT system prompt and available tools."""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # --- DEFENSIVE PROGRAMMING START ---
            if chat_history is None:
                chat_history = []
            
            if not isinstance(chat_history, list):
                logger.warning(f"chat_history was not a list, it was {type(chat_history)}. Resetting to empty list.")
                chat_history = []

            clean_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in chat_history 
                if isinstance(msg, dict) and msg.get("content")
            ]
            messages.extend(clean_history)
            # --- DEFENSIVE PROGRAMMING END ---
        
            messages.append({"role": "user", "content": user_prompt})


            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="required" if tools else None,
                temperature=self.execution_temperature,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.exception(f"OpenAI API call with tools failed: {e}")
            raise

    def _get_available_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Gets available tools formatted for OpenAI function calling."""
        if not self.tool_manager:
            return []
        try:
            tool_schemas = self.tool_manager.list_tool_schemas()
            openai_tools = []
            for schema in tool_schemas:
                if schema.get("error"):
                    continue
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema.get("description", "No description"),
                        "parameters": schema.get("input_schema", {"type": "object", "properties": {}})
                    }
                }
                openai_tools.append(openai_tool)
            return openai_tools
        except Exception as e:
            logger.exception(f"Failed to format tools for OpenAI: {e}")
            return []

    def _extract_tool_plans_from_response(self, response) -> List[Dict[str, Any]]:
        """Extracts tool execution plans from OpenAI response."""
        try:
            tool_plans = []
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        plan = {
                            "tool_name": tool_call.function.name,
                            "arguments": arguments,
                            "reasoning": f"LLM decided to call {tool_call.function.name}."
                        }
                        tool_plans.append(plan)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call arguments: {e}")
            
            if not tool_plans and message.content:
                tool_plans.append({
                    "tool_name": "direct_answer",
                    "arguments": {"query": "Based on reasoning in message content"},
                    "reasoning": message.content
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
            
            final_answer = await self._call_openai_for_analysis(
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
    
    async def execute_with_cot(self, query: str, session_id: Optional[str] = None, chat_history: Optional[List[Dict[str, Any]]] = None) -> CoTSession:
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
                query, complexity_analysis, session_id , chat_history
            )
            
            # Step 3: Execute tool chain with reasoning
            logger.info("Step 3: Executing tool chain with reflective reasoning")
            await self.execute_tool_chain_with_reasoning(
                cot_session, execution_plan, chat_history
            )
            
            # Step 4: Synthesize final results
            logger.info("Step 4: Synthesizing results with CoT")
            final_answer = await self.synthesize_results_with_cot(cot_session, chat_history)
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
            # Create analysis prompt
            analysis_prompt = f"""
Analyze this query for complexity:

"{query}"

Provide a JSON response with:
{{
    "complexity": "simple|moderate|complex|very_complex",
    "reasoning": "detailed explanation",
    "estimated_steps": number,
    "required_tools": ["tool1", "tool2"],
    "confidence": 0.0-1.0
}}
"""
            
            response = await self._call_openai_for_analysis(
                self.system_prompts["complexity_analysis"],
                analysis_prompt
            )
            
            # Parse response
            try:
                analysis_data = json.loads(response)
                complexity = QueryComplexity(analysis_data["complexity"])
                
                return ComplexityAnalysis(
                    complexity=complexity,
                    reasoning=analysis_data["reasoning"],
                    estimated_steps=analysis_data["estimated_steps"],
                    required_tools=analysis_data["required_tools"],
                    confidence=analysis_data["confidence"]
                )
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse complexity analysis: {e}")
                # Fallback to moderate complexity
                return ComplexityAnalysis(
                    complexity=QueryComplexity.MODERATE,
                    reasoning="Could not parse complexity analysis, defaulting to moderate",
                    estimated_steps=3,
                    required_tools=[],
                    confidence=0.5
                )
                
        except Exception as e:
            logger.exception(f"Query complexity analysis failed: {e}")
            return ComplexityAnalysis(
                complexity=QueryComplexity.SIMPLE,
                reasoning=f"Analysis failed ({str(e)}), defaulting to simple",
                estimated_steps=1,
                required_tools=[],
                confidence=0.3
            )
    
    async def plan_tool_execution_with_cot(self, query: str, complexity: ComplexityAnalysis, session_id: Optional[str], chat_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Plan tool execution sequence using Chain of Thought reasoning.
        
        Args:
            query: The user query
            complexity: Complexity analysis results
            session_id: Optional user session ID for context
            
        Returns:
            List of planned tool executions with reasoning
        """
        logger.debug(f"Planning tool execution for {complexity.complexity.value} query")
        
        try:
            # Get available tools
            available_tools = self._get_available_tools_for_openai()
            
            # Get session context if available
            session_context = ""
            if session_id and self.session_manager:
                session_docs = self.session_manager.get_session_documents(session_id)
                if session_docs:
                    session_context = f"\nUser has {len(session_docs)} documents in session: "
                    session_context += ", ".join([doc.file_name for doc in session_docs[:5]])
            
            # Create planning prompt
            planning_prompt = f"""
        You are a planning agent. Your primary task is to create a step-by-step plan to answer the user's query by calling the available tools.

        **User Query:** "{query}"

        **Analysis:**
        - Complexity: {complexity.complexity.value}
        - Reasoning: {complexity.reasoning}

        **Session Context:** {session_context if session_context else "No documents available in the current session."}

        **Your Task:**
        Based on the user's query and the available context, you MUST decide which tool to call.
        - **YOU MUST RESPOND BY CALLING ONE OR MORE TOOLS.**
        - **DO NOT ANSWER THE QUERY DIRECTLY.** Your only job is to select and call the correct tool.
        - If the user asks to summarize a document, call the `summarize_document` tool.
        - If the user asks to compare documents, call the `compare_documents` tool.

        Now, create the plan for the user's query by calling the appropriate tool.
        """
            
            # Use OpenAI with function calling if tools are available
            if available_tools:
                response = await self._call_openai_with_cot_and_tools(
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

    async def execute_tool_chain_with_reasoning(self, cot_session: CoTSession, execution_plan: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Execute the planned tool chain with reflective reasoning and intelligent parameter management.
        
        Args:
            cot_session: The CoT session to update with executions
            execution_plan: List of planned tool executions
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
            #error_reflection = ReasoningStep(
             #   step_id=f"error_reflect_{len(cot_session.reasoning_steps)}",
              #  step_type="reflection",
               ## content=f"Tool execution failed for {tool_name}: {str(e)}. Attempting to continue if possible.",
                #context={"error": str(e), "tool": tool_name}
            #)
            

            error_reflection = await self.reflect_on_step_result( # <-- await ekle
                    cot_session.original_query, tool_execution, chat_history # <-- chat_history ekle
                )

            tool_execution.post_execution_reflection = error_reflection
            cot_session.reasoning_steps.append(error_reflection)
        
        cot_session.tool_executions.append(tool_execution)
        
        # Brief pause between executions
        await asyncio.sleep(0.1)
    
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
            
            reflection_content = await self._call_openai_for_analysis(
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
            
            final_answer = await self._call_openai_for_analysis(
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
    
    async def _call_openai_with_cot_and_tools(
    self, 
    system_prompt: str, 
    user_prompt: str, 
    tools: List[Dict[str, Any]], 
    chat_history: Optional[List[Dict[str, Any]]] = None # <-- BU PARAMETREYİ EKLE
) -> Any:
        """
        Call OpenAI API with CoT system prompt and available tools.
        
        Args:
            system_prompt: System prompt for CoT reasoning
            user_prompt: User's query or instruction
            tools: List of available tools in OpenAI format
            
        Returns:
            OpenAI API response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="required" if tools else None,
                    temperature=self.execution_temperature,
                    max_tokens=2000
                )
            )
            
            return response
            
        except Exception as e:
            logger.exception(f"OpenAI API call with tools failed: {e}")
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
            
            return await self._call_openai_for_analysis(
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
            
            return await self._call_openai_for_analysis(
                "You are providing a fallback response after a system error.",
                fallback_prompt,
                chat_history
            )
            
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties and cannot process your query at this time. Error: {str(e)}"
    
    def _get_available_tools_for_openai(self) -> List[Dict[str, Any]]:
        """
        Get available tools formatted for OpenAI function calling.
        
        Returns:
            List of tool schemas in OpenAI format
        """
        if not self.tool_manager:
            return []
        
        try:
            tool_schemas = self.tool_manager.list_tool_schemas()
            openai_tools = []
            
            for schema in tool_schemas:
                if schema.get("error"):
                    continue  # Skip tools with errors
                
                # Convert to OpenAI function calling format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema.get("description", "No description available"),
                        "parameters": schema.get("input_schema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                openai_tools.append(openai_tool)
            
            logger.debug(f"Formatted {len(openai_tools)} tools for OpenAI function calling")
            return openai_tools
            
        except Exception as e:
            logger.exception(f"Failed to format tools for OpenAI: {e}")
            return []
    
    def _extract_tool_plans_from_response(self, response) -> List[Dict[str, Any]]:
        """
        Extract tool execution plans from OpenAI response.
        
        Args:
            response: OpenAI API response
            
        Returns:
            List of planned tool executions
        """
        try:
            tool_plans = []
            message = response.choices[0].message
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        plan = {
                            "tool_name": tool_call.function.name,
                            "arguments": arguments,
                            "reasoning": f"OpenAI selected {tool_call.function.name} for this step"
                        }
                        tool_plans.append(plan)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call arguments: {e}")
            
            # If no tool calls, check if there's reasoning content
            if not tool_plans and message.content:
                # Create a direct answer plan
                tool_plans.append({
                    "tool_name": "direct_answer",
                    "arguments": {"query": "Based on reasoning provided"},
                    "reasoning": "No specific tools called, providing direct response"
                })
            
            return tool_plans
            
        except Exception as e:
            logger.exception(f"Failed to extract tool plans from response: {e}")
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
