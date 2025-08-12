"""
DataNeuron Analysis Tools
========================

This module contains tools for analyzing documents and data.
"""

from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from typing import Dict, Any


# ============================================================================
# ANALYSIS TOOL IMPLEMENTATIONS
# ============================================================================

class TextAnalysisArgs(BaseToolArgs):
    """Arguments for text analysis tool."""
    text: str
    analysis_type: str = "summary"


class TextAnalysisResult(BaseToolResult):
    """Result for text analysis tool."""
    analysis: str
    word_count: int
    character_count: int


class TextAnalysisTool(BaseTool):
    """
    Tool for analyzing text content.
    
    Provides basic text analysis including word count, character count,
    and simple analysis based on the specified type.
    """
    
    name = "analyze_text"
    description = (
        "Analyzes text content and provides statistics and insights. "
        "Supports different analysis types including 'summary', 'stats', and 'structure'. "
        "Returns word count, character count, and analysis based on the specified type."
    )
    args_schema = TextAnalysisArgs
    return_schema = TextAnalysisResult
    version = "1.0.0"
    category = "analysis"
    
    def _execute(self, text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Analyze the provided text.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        word_count = len(text.split())
        character_count = len(text)
        
        if analysis_type == "summary":
            analysis = f"Text contains {word_count} words and {character_count} characters."
        elif analysis_type == "stats":
            sentences = text.count('.') + text.count('!') + text.count('?')
            paragraphs = text.count('\n\n') + 1
            analysis = f"Statistics: {word_count} words, {character_count} characters, ~{sentences} sentences, ~{paragraphs} paragraphs."
        elif analysis_type == "structure":
            lines = len(text.split('\n'))
            analysis = f"Structure: {lines} lines, {word_count} words, average {character_count/word_count:.1f} chars per word."
        else:
            analysis = f"Unknown analysis type '{analysis_type}'. Available types: summary, stats, structure."
        
        return {
            "analysis": analysis,
            "word_count": word_count,
            "character_count": character_count,
            "metadata": {
                "analysis_type": analysis_type,
                "tool_version": self.version
            }
        }