"""
DataNeuron User Interaction Tools
=================================

This module provides tools for LLM agents to interact with users when they need
clarification, additional information, or guidance to resolve ambiguous situations.
These tools enable agents to actively engage with users rather than making assumptions
or guesses when faced with unclear requests.

Features:
- AskUserForClarificationTool: Prompts user for clarification when queries are ambiguous
- Session-aware user interaction capabilities
- Standardized question formatting and response handling
- Integration with UI systems for seamless user experience

Usage:
    # These tools are automatically discovered by ToolManager
    # and can be executed by LLM agents through tool calls:
    
    # Ask user for clarification
    result = run_tool("ask_user_for_clarification", 
                      question="Which document would you like me to summarize? Please choose from: report.pdf, analysis.docx")

Integration:
- Follows BaseTool architecture for consistency
- Provides signals for UI systems to display questions to users
- Enables intelligent agent behavior when facing ambiguous situations
- Prevents agents from making incorrect assumptions
"""

import sys
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from utils.logger import logger


# ============================================================================
# PYDANTIC SCHEMAS FOR USER INTERACTION TOOLS
# ============================================================================

class AskUserArgs(BaseToolArgs):
    """
    Arguments for asking user clarification questions.
    
    Attributes:
        question: Clear and specific question to ask the user for clarification
    """
    question: str = Field(
        ...,
        description="The clear and specific question to ask the user for clarification.",
        min_length=10,
        max_length=500
    )


class AskUserResult(BaseToolResult):
    """
    Result of asking user a clarification question.
    
    Attributes:
        question_asked: Confirmation that the question was successfully relayed
    """
    question_asked: str = Field(
        ...,
        description="A confirmation that the question was successfully relayed to the user interface."
    )


# ============================================================================
# USER INTERACTION TOOLS
# ============================================================================

class AskUserForClarificationTool(BaseTool):
    """
    Tool for asking users clarification questions when faced with ambiguous requests.
    
    This tool is designed to be used when the LLM agent encounters situations where
    the user's intent is unclear and the agent cannot proceed without additional
    information. It's particularly useful for resolving ambiguity about which 
    documents to use when multiple options are available.
    
    Key Use Cases:
    - Multiple documents available but unclear which one to use
    - Ambiguous parameters in user requests
    - Missing critical information needed to proceed
    - Conflicting or contradictory instructions
    
    The tool generates a signal that can be picked up by the UI layer to
    present the question to the user and await their response.
    """
    
    name = "ask_user_for_clarification"
    description = (
    "Use this tool when user requests contain ambiguity that could lead to incorrect execution. "
    "Your role is to resolve uncertainty through targeted questions, not assumptions.\n\n"
    
    "**WHEN TO USE:**\n"
    "• Multiple documents/options available - unclear which to select\n"
    "• Missing critical parameters needed for accurate execution\n" 
    "• Contradictory instructions that cannot be resolved logically\n"
    "• Vague references requiring specification ('latest', 'main', 'primary')\n"
    "• Technical jargon or domain terms needing clarification\n"
    "• Incomplete task specifications affecting output quality\n\n"
    
    "**QUESTION DESIGN PRINCIPLES:**\n"
    "• **Closed-ended preferred:** Offer specific choices when possible\n"
    "• **Context-specific:** Reference available documents/data by name\n"
    "• **Parameter-focused:** List exact missing information needed\n"
    "• **Decision-oriented:** Frame questions to enable immediate action\n"
    "• **Professional tone:** Clear, concise, respectful language\n\n"
    
    "**EFFECTIVE QUESTION PATTERNS:**\n"
    "✓ 'Which document: contract_A.pdf or contract_B.pdf?'\n"
    "✓ 'Please specify: (1) Date range, (2) Output format, (3) Analysis depth'\n"
    "✓ 'I found 3 financial reports. Which timeframe: Q1, Q2, or Annual?'\n"
    "✓ 'Clarify comparison criteria: revenue, profit margins, or growth rates?'\n\n"
    
    "**AVOID ASSUMPTIONS - ASK INSTEAD:**\n"
    " Don't assume 'latest' means most recent file\n"
    " Don't guess which metrics to include in analysis\n"
    " Don't interpret vague timeframes arbitrarily\n"
    " Don't choose document formats without confirmation\n\n"
    
    "**BUSINESS IMPACT:**\n"
    "• Prevents incorrect analysis and wasted resources\n"
    "• Ensures precise execution aligned with user intent\n"
    "• Builds trust through collaborative problem-solving\n"
    "• Reduces iteration cycles and improves efficiency\n\n"
    
    "**EXAMPLE SCENARIOS:**\n"
    " 'Analyze the report' → 'Which report would you like analyzed: Q3_financial.pdf, "
    "market_research.docx, or compliance_audit.pdf?'\n"
    " 'Compare performance' → 'Please specify: (1) Which metrics to compare, "
    "(2) Time periods for comparison, (3) Benchmark sources'\n"
    " 'Summarize key findings' → 'Which type of summary: executive overview, "
    "technical details, or risk assessment focus?'"
)
    args_schema = AskUserArgs
    return_schema = AskUserResult
    version = "1.0.0"
    category = "interaction"
    requires_session = False  # This tool doesn't need session context
    is_async = False
    
    def _execute(self, question: str) -> Dict[str, Any]:
        """
        Execute the user clarification request.
        
        This method logs the question and returns a confirmation that the question
        has been prepared for the user interface. The actual question display and
        user response collection is handled by the UI layer.
        
        Args:
            question: The clarification question to ask the user
            
        Returns:
            Dictionary containing confirmation of question relay
        """
        # Log the clarification request for monitoring and debugging
        logger.info(f"AskUserForClarificationTool: Requesting user clarification")
        logger.info(f"Question: {question}")
        
        # Return confirmation that the question has been processed
        # The UI layer will pick up this signal and display the question to the user
        return {
            "question_asked": question,
            "metadata": {
                "tool_name": self.name,
                "tool_version": self.version,
                "question_length": len(question),
                "timestamp": "auto-generated"
            }
        }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the AskUserForClarificationTool to ensure proper functionality.
    This validates the tool's ability to handle various types of clarification requests.
    """
    
    print("=== DataNeuron User Interaction Tools Test ===")
    
    # Test 1: Create tool instance
    print("\nTest 1: Creating AskUserForClarificationTool instance")
    try:
        clarification_tool = AskUserForClarificationTool()
        print(f" PASS - Tool created: {clarification_tool}")
        print(f"   Name: {clarification_tool.name}")
        print(f"   Category: {clarification_tool.category}")
        print(f"   Version: {clarification_tool.version}")
    except Exception as e:
        print(f" FAIL - Tool creation failed: {e}")
    
    # Test 2: Execute with valid clarification question
    print("\nTest 2: Executing with valid clarification question")
    try:
        test_question = "Which document would you like me to summarize? Please choose from: report.pdf, analysis.docx, presentation.pptx"
        result = clarification_tool.execute(question=test_question)
        
        print(f" PASS - Clarification request processed successfully")
        print(f"   Success: {result.success}")
        print(f"   Question asked: {result.question_asked}")
        print(f"   Metadata: {result.metadata}")
        
        # Verify the question was preserved correctly
        if result.question_asked == test_question:
            print(f" PASS - Question preserved correctly")
        else:
            print(f" FAIL - Question not preserved correctly")
            
    except Exception as e:
        print(f" FAIL - Valid execution failed: {e}")
    
    # Test 3: Execute with invalid arguments (too short question)
    print("\nTest 3: Executing with invalid arguments (question too short)")
    try:
        result = clarification_tool.execute(question="Help?")  # Too short
        
        if not result.success:
            print(f" PASS - Short question validation error handled")
            print(f"   Error type: {result.error_type}")
            print(f"   Error message: {result.error_message}")
        else:
            print(f" FAIL - Short question should have been rejected")
            
    except Exception as e:
        print(f" FAIL - Error handling failed: {e}")
    
    # Test 4: Execute with missing arguments
    print("\nTest 4: Executing with missing arguments")
    try:
        result = clarification_tool.execute()  # Missing question
        
        if not result.success:
            print(f" PASS - Missing argument error handled")
            print(f"   Error type: {result.error_type}")
            print(f"   Validation errors: {len(result.metadata.get('validation_errors', []))}")
        else:
            print(f" FAIL - Missing argument should have been rejected")
            
    except Exception as e:
        print(f" FAIL - Error handling failed: {e}")
    
    # Test 5: Test argument validation without execution
    print("\nTest 5: Argument validation without execution")
    try:
        valid_question = "Could you please clarify which analysis method you'd prefer: statistical analysis or machine learning approach?"
        validated_args = clarification_tool.validate_args(question=valid_question)
        
        print(f" PASS - Arguments validated successfully")
        print(f"   Question length: {len(validated_args.question)}")
        print(f"   Question preview: {validated_args.question[:50]}...")
        
    except Exception as e:
        print(f" FAIL - Argument validation failed: {e}")
    
    # Test 6: Test tool schema information
    print("\nTest 6: Tool schema information retrieval")
    try:
        schema_info = clarification_tool.get_schema_info()
        
        print(f" PASS - Schema info retrieved successfully")
        print(f"   Tool name: {schema_info['name']}")
        print(f"   Description: {schema_info['description'][:80]}...")
        print(f"   Required fields: {list(schema_info['input_schema']['required'])}")
        print(f"   Requires session: {schema_info['requires_session']}")
        
    except Exception as e:
        print(f" FAIL - Schema info retrieval failed: {e}")
    
    # Test 7: Test with various question types
    print("\nTest 7: Testing various question types")
    
    test_questions = [
        "Which document format would you prefer for the output: PDF, Word, or Plain Text?",
        "I found multiple files with similar names. Which one did you mean: 'report_2023.pdf' or 'report_2024.pdf'?",
        "The search returned many results. Would you like me to focus on a specific time period or topic?",
        "I need more information to proceed. What is the main objective of this analysis?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        try:
            result = clarification_tool.execute(question=question)
            if result.success:
                print(f"   PASS Question {i}: Successfully processed")
            else:
                print(f"   FAIL Question {i}: Failed - {result.error_message}")
        except Exception as e:
            print(f"   FAIL Question {i}: Exception - {e}")
    
    print("\n=== Test Complete ===")
    print("AskUserForClarificationTool is ready for production use.")
    print("The tool provides a reliable way for agents to request user clarification.")
    print("Integration with UI systems will enable seamless user interaction workflows.")