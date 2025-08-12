"""
DataNeuron Advanced LLM-Powered Analysis Tools
==============================================

This module provides sophisticated analysis tools that leverage LLM capabilities
for deep document insights, multi-document comparisons, risk assessments, and
result synthesis. These tools are designed to work with the LLMAgent for
complex analytical workflows.

Features:
- Multi-document comparison with structured analysis
- Risk assessment and categorization for documents  
- Multi-tool result synthesis for comprehensive insights
- LLM-powered deep analysis and understanding
- Markdown formatted outputs for readability
- Session-aware document processing

Usage:
    from tools.analysis_tools import CompareDocumentsTool, RiskAssessmentTool
    
    # Compare multiple documents
    compare_tool = CompareDocumentsTool()
    result = compare_tool.execute(
        file_names=["doc1.pdf", "doc2.pdf"],
        comparison_criteria="financial performance",
        session_id="user_session_123"
    )
    
    # Assess risks in a document
    risk_tool = RiskAssessmentTool()
    result = risk_tool.execute(
        file_name="contract.pdf",
        session_id="user_session_123"
    )

Architecture:
- Built on BaseTool architecture for consistency
- LLM integration for intelligent analysis
- Session manager integration for document access
- Comprehensive error handling and validation
- Structured outputs for downstream processing
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, BaseToolArgs, BaseToolResult
from utils.logger import logger
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

# Import dependencies with error handling
try:
    import openai
except ImportError as e:
    logger.error(f"OpenAI library not available: {e}")
    openai = None

try:
    from core.session_manager import SessionManager
except ImportError as e:
    logger.warning(f"SessionManager not available: {e}")
    SessionManager = None


# ============================================================================
# DOCUMENT COMPARISON TOOL
# ============================================================================

class CompareDocumentsArgs(BaseToolArgs):
    """Arguments for document comparison tool."""
    file_names: List[str]
    comparison_criteria: str
    session_id: str
    output_format: str = "markdown"  # "markdown" or "text"


class CompareDocumentsResult(BaseToolResult):
    """Result for document comparison tool."""
    comparison_analysis: str
    documents_compared: List[str]
    criteria_used: str
    similarities: List[str]
    differences: List[str]
    key_insights: List[str]


class CompareDocumentsTool(BaseTool):
    """
    Advanced multi-document comparison tool using LLM analysis.
    
    This tool compares multiple documents based on specified criteria and
    provides structured analysis highlighting similarities, differences,
    and key insights. It supports various comparison criteria such as
    financial performance, legal obligations, main themes, etc.
    """
    
    name = "compare_documents"
    description = (
        "Compares two or more documents based on specified criteria such as "
        "'financial performance', 'legal obligations', or 'main themes'. "
        "Provides structured analysis highlighting similarities, differences, "
        "and important discrepancies in a table or text format."
    )
    args_schema = CompareDocumentsArgs
    return_schema = CompareDocumentsResult
    version = "1.0.0"
    category = "analysis"
    requires_session = True
    
    def __init__(self):
        super().__init__()
        # Initialize OpenAI client
        if openai and OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("OpenAI client not available for CompareDocumentsTool")
        
        # Initialize session manager
        self.session_manager = SessionManager() if SessionManager else None
    
    def _execute(self, file_names: List[str], comparison_criteria: str, session_id: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Compare multiple documents based on specified criteria.
        
        Args:
            file_names: List of document names to compare
            comparison_criteria: Criteria for comparison (e.g., "financial performance")
            session_id: Session ID to retrieve documents
            output_format: Output format ("markdown" or "text")
            
        Returns:
            Dictionary with comparison analysis results
        """
        logger.info(f"Comparing {len(file_names)} documents: {file_names}")
        logger.info(f"Criteria: {comparison_criteria}")
        
        try:
            # Step 1: Retrieve document contents from session
            document_contents = self._get_document_contents(file_names, session_id)
            
            if not document_contents:
                return {
                    "comparison_analysis": "No documents found in session or documents could not be accessed.",
                    "documents_compared": file_names,
                    "criteria_used": comparison_criteria,
                    "similarities": [],
                    "differences": [],
                    "key_insights": ["Error: Documents not accessible"],
                    "metadata": {"error": "Documents not found or accessible"}
                }
            
            # Step 2: Perform LLM-powered comparison
            if self.client:
                comparison_result = self._perform_llm_comparison(
                    document_contents, comparison_criteria, output_format
                )
            else:
                # Fallback analysis without LLM
                comparison_result = self._perform_fallback_comparison(
                    document_contents, comparison_criteria
                )
            
            # Step 3: Structure the results
            return {
                "comparison_analysis": comparison_result["analysis"],
                "documents_compared": list(document_contents.keys()),
                "criteria_used": comparison_criteria,
                "similarities": comparison_result["similarities"],
                "differences": comparison_result["differences"],
                "key_insights": comparison_result["insights"],
                "metadata": {
                    "documents_count": len(document_contents),
                    "total_content_length": sum(len(content) for content in document_contents.values()),
                    "output_format": output_format,
                    "llm_powered": self.client is not None
                }
            }
            
        except Exception as e:
            logger.exception(f"Document comparison failed: {e}")
            return {
                "comparison_analysis": f"Document comparison failed: {str(e)}",
                "documents_compared": file_names,
                "criteria_used": comparison_criteria,
                "similarities": [],
                "differences": [],
                "key_insights": [f"Error occurred: {str(e)}"],
                "metadata": {"error": str(e)}
            }
    
    def _get_document_contents(self, file_names: List[str], session_id: str) -> Dict[str, str]:
        """
        Retrieve document contents from session manager.
        
        Args:
            file_names: List of document names
            session_id: Session ID
            
        Returns:
            Dictionary mapping file names to their contents
        """
        document_contents = {}
        
        if not self.session_manager:
            logger.error("SessionManager not available")
            return document_contents
        
        try:
            # Get all documents from session
            session_documents = self.session_manager.get_session_documents(session_id)
            
            # Create a mapping of file names to documents
            doc_map = {doc.file_name: doc for doc in session_documents}
            
            # Retrieve content for requested files
            for file_name in file_names:
                if file_name in doc_map:
                    # For now, we'll use the content preview since full content retrieval
                    # would require additional integration with document processor
                    document_contents[file_name] = doc_map[file_name].content_preview or "Content not available"
                    logger.debug(f"Retrieved content for {file_name}: {len(document_contents[file_name])} characters")
                else:
                    logger.warning(f"Document {file_name} not found in session {session_id}")
                    document_contents[file_name] = f"Document '{file_name}' not found in session"
        
        except Exception as e:
            logger.exception(f"Failed to retrieve document contents: {e}")
        
        return document_contents
    
    def _perform_llm_comparison(self, document_contents: Dict[str, str], criteria: str, output_format: str) -> Dict[str, Any]:
        """
        Perform LLM-powered document comparison.
        
        Args:
            document_contents: Mapping of file names to contents
            criteria: Comparison criteria
            output_format: Output format preference
            
        Returns:
            Dictionary with analysis, similarities, differences, and insights
        """
        try:
            # Prepare documents for comparison
            doc_texts = []
            for file_name, content in document_contents.items():
                doc_texts.append(f"**Document: {file_name}**\n{content[:2000]}...")  # Limit content length
            
            combined_docs = "\n\n---\n\n".join(doc_texts)
            
            # Create comparison prompt
            comparison_prompt = f"""
Compare the following documents based on the criteria: "{criteria}"

Documents:
{combined_docs}

Provide a comprehensive comparison analysis that includes:

1. **Overall Comparison Summary**: A brief overview of how the documents relate to the specified criteria

2. **Key Similarities**: List the main similarities between the documents regarding the criteria

3. **Key Differences**: List the main differences between the documents regarding the criteria

4. **Important Insights**: Provide analytical insights and observations

5. **Detailed Analysis**: {'Create a markdown table comparing key aspects' if output_format == 'markdown' else 'Provide detailed comparative analysis'}

Format your response clearly with structured sections.
"""
            
            # Call OpenAI for comparison
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document analyst specializing in comparative analysis. Provide thorough, structured comparisons based on specified criteria."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract structured information from the analysis
            similarities = self._extract_list_from_analysis(analysis_text, "similarities")
            differences = self._extract_list_from_analysis(analysis_text, "differences")
            insights = self._extract_list_from_analysis(analysis_text, "insights")
            
            return {
                "analysis": analysis_text,
                "similarities": similarities,
                "differences": differences,
                "insights": insights
            }
            
        except Exception as e:
            logger.exception(f"LLM comparison failed: {e}")
            # Fallback to basic comparison
            return self._perform_fallback_comparison(document_contents, criteria)
    
    def _perform_fallback_comparison(self, document_contents: Dict[str, str], criteria: str) -> Dict[str, Any]:
        """
        Perform basic comparison without LLM when OpenAI is not available.
        
        Args:
            document_contents: Mapping of file names to contents
            criteria: Comparison criteria
            
        Returns:
            Dictionary with basic analysis
        """
        file_names = list(document_contents.keys())
        
        # Basic analysis
        analysis = f"# Document Comparison: {criteria}\n\n"
        analysis += f"Comparing {len(file_names)} documents: {', '.join(file_names)}\n\n"
        
        # Basic statistics
        stats = {}
        for name, content in document_contents.items():
            word_count = len(content.split())
            char_count = len(content)
            stats[name] = {"words": word_count, "chars": char_count}
        
        analysis += "## Document Statistics\n\n"
        for name, stat in stats.items():
            analysis += f"- **{name}**: {stat['words']} words, {stat['chars']} characters\n"
        
        # Basic insights
        similarities = ["All documents contain text content"]
        differences = [f"Document lengths vary: {', '.join([f'{name} ({stats[name]['words']} words)' for name in file_names])}"]
        insights = ["Detailed comparison requires LLM analysis", f"Comparison criteria: {criteria}"]
        
        return {
            "analysis": analysis,
            "similarities": similarities,
            "differences": differences,
            "insights": insights
        }
    
    def _extract_list_from_analysis(self, text: str, section_type: str) -> List[str]:
        """
        Extract lists from analysis text based on section type.
        
        Args:
            text: Analysis text
            section_type: Type of section to extract ("similarities", "differences", "insights")
            
        Returns:
            List of extracted items
        """
        items = []
        try:
            # Look for section headers and extract bullet points
            lines = text.split('\n')
            in_section = False
            
            section_keywords = {
                "similarities": ["similarities", "similar", "common"],
                "differences": ["differences", "different", "differ"],
                "insights": ["insights", "observations", "analysis"]
            }
            
            keywords = section_keywords.get(section_type, [])
            
            for line in lines:
                line = line.strip()
                
                # Check if we're entering the section
                if any(keyword in line.lower() for keyword in keywords) and ('**' in line or '#' in line):
                    in_section = True
                    continue
                
                # Check if we're leaving the section
                if in_section and line.startswith('#') or (line.startswith('**') and line.endswith('**')):
                    if not any(keyword in line.lower() for keyword in keywords):
                        in_section = False
                        continue
                
                # Extract items from the section
                if in_section and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    item = line[1:].strip()
                    if item and len(item) > 5:  # Filter out very short items
                        items.append(item)
            
            # If no items found, provide a default
            if not items:
                items = [f"No specific {section_type} identified"]
                
        except Exception as e:
            logger.warning(f"Failed to extract {section_type}: {e}")
            items = [f"Could not extract {section_type} from analysis"]
        
        return items[:5]  # Limit to 5 items


# ============================================================================
# RISK ASSESSMENT TOOL
# ============================================================================

class RiskAssessmentArgs(BaseToolArgs):
    """Arguments for risk assessment tool."""
    file_name: str
    session_id: str
    risk_categories: List[str] = ["financial", "legal", "operational", "reputational", "compliance"]


class RiskAssessmentResult(BaseToolResult):
    """Result for risk assessment tool."""
    risk_analysis: str
    document_analyzed: str
    risks_identified: List[Dict[str, Any]]
    risk_categories: Dict[str, int]  # Category -> count mapping
    overall_risk_level: str
    recommendations: List[str]


class RiskAssessmentTool(BaseTool):
    """
    Advanced risk assessment tool for document analysis.
    
    This tool analyzes documents (especially legal contracts, reports, etc.)
    to identify potential risks, uncertainties, or negative statements.
    It categorizes risks and assigns importance levels.
    """
    
    name = "assess_risks_in_document"
    description = (
        "Analyzes a document to identify potential risks, uncertainties, or negative statements. "
        "Particularly useful for legal contracts, reports, and business documents. "
        "Categorizes risks (financial, legal, operational) and assigns importance levels."
    )
    args_schema = RiskAssessmentArgs
    return_schema = RiskAssessmentResult
    version = "1.0.0"
    category = "analysis"
    requires_session = True
    
    def __init__(self):
        super().__init__()
        # Initialize OpenAI client
        if openai and OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("OpenAI client not available for RiskAssessmentTool")
        
        # Initialize session manager
        self.session_manager = SessionManager() if SessionManager else None
    
    def _execute(self, file_name: str, session_id: str, risk_categories: List[str] = None) -> Dict[str, Any]:
        """
        Assess risks in a document.
        
        Args:
            file_name: Name of the document to analyze
            session_id: Session ID to retrieve the document
            risk_categories: Categories to focus on for risk assessment
            
        Returns:
            Dictionary with risk assessment results
        """
        if risk_categories is None:
            risk_categories = ["financial", "legal", "operational", "reputational", "compliance"]
        
        logger.info(f"Assessing risks in document: {file_name}")
        logger.info(f"Risk categories: {risk_categories}")
        
        try:
            # Step 1: Retrieve document content
            document_content = self._get_document_content(file_name, session_id)
            
            if not document_content:
                return {
                    "risk_analysis": f"Document '{file_name}' not found or not accessible.",
                    "document_analyzed": file_name,
                    "risks_identified": [],
                    "risk_categories": {cat: 0 for cat in risk_categories},
                    "overall_risk_level": "unknown",
                    "recommendations": ["Unable to analyze - document not accessible"],
                    "metadata": {"error": "Document not found"}
                }
            
            # Step 2: Perform risk assessment
            if self.client:
                risk_result = self._perform_llm_risk_assessment(document_content, risk_categories)
            else:
                risk_result = self._perform_fallback_risk_assessment(document_content, risk_categories)
            
            # Step 3: Structure the results
            return {
                "risk_analysis": risk_result["analysis"],
                "document_analyzed": file_name,
                "risks_identified": risk_result["risks"],
                "risk_categories": risk_result["categories"],
                "overall_risk_level": risk_result["overall_level"],
                "recommendations": risk_result["recommendations"],
                "metadata": {
                    "content_length": len(document_content),
                    "categories_analyzed": len(risk_categories),
                    "risks_found": len(risk_result["risks"]),
                    "llm_powered": self.client is not None
                }
            }
            
        except Exception as e:
            logger.exception(f"Risk assessment failed: {e}")
            return {
                "risk_analysis": f"Risk assessment failed: {str(e)}",
                "document_analyzed": file_name,
                "risks_identified": [],
                "risk_categories": {cat: 0 for cat in risk_categories},
                "overall_risk_level": "error",
                "recommendations": [f"Error occurred: {str(e)}"],
                "metadata": {"error": str(e)}
            }
    
    def _get_document_content(self, file_name: str, session_id: str) -> Optional[str]:
        """
        Retrieve document content from session manager.
        
        Args:
            file_name: Name of the document
            session_id: Session ID
            
        Returns:
            Document content or None if not found
        """
        if not self.session_manager:
            logger.error("SessionManager not available")
            return None
        
        try:
            # Get all documents from session
            session_documents = self.session_manager.get_session_documents(session_id)
            
            # Find the requested document
            for doc in session_documents:
                if doc.file_name == file_name:
                    logger.debug(f"Retrieved document content for {file_name}")
                    # Use content preview for now - in production, full content would be retrieved
                    return doc.content_preview or "Content not available"
            
            logger.warning(f"Document {file_name} not found in session {session_id}")
            return None
        
        except Exception as e:
            logger.exception(f"Failed to retrieve document content: {e}")
            return None
    
    def _perform_llm_risk_assessment(self, content: str, risk_categories: List[str]) -> Dict[str, Any]:
        """
        Perform LLM-powered risk assessment.
        
        Args:
            content: Document content to analyze
            risk_categories: Categories to focus on
            
        Returns:
            Dictionary with risk assessment results
        """
        try:
            # Limit content length for API call
            analysis_content = content[:3000]  # First 3000 characters
            
            risk_prompt = f"""
Analyze the following document for potential risks, uncertainties, and negative statements.

Document Content:
{analysis_content}

Focus on these risk categories: {', '.join(risk_categories)}

Provide a comprehensive risk assessment that includes:

1. **Overall Risk Level**: Rate as "Low", "Medium", "High", or "Critical"

2. **Identified Risks**: For each risk found, specify:
   - Risk description
   - Category (from: {', '.join(risk_categories)})
   - Severity level (1-5, where 5 is most severe)
   - Specific text that indicates this risk

3. **Risk Category Summary**: Count of risks in each category

4. **Key Risk Indicators**: Specific phrases or clauses that signal risks

5. **Recommendations**: Actions to mitigate identified risks

Format your response with clear sections and be specific about risk locations in the text.
"""
            
            # Call OpenAI for risk assessment
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert risk analyst specializing in document risk assessment. Identify potential risks, uncertainties, and negative indicators with high accuracy."},
                    {"role": "user", "content": risk_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract structured risk information
            risks = self._extract_risks_from_analysis(analysis_text)
            overall_level = self._extract_overall_risk_level(analysis_text)
            recommendations = self._extract_recommendations_from_analysis(analysis_text)
            
            # Count risks by category
            category_counts = {cat: 0 for cat in risk_categories}
            for risk in risks:
                category = risk.get("category", "other")
                if category in category_counts:
                    category_counts[category] += 1
            
            return {
                "analysis": analysis_text,
                "risks": risks,
                "categories": category_counts,
                "overall_level": overall_level,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.exception(f"LLM risk assessment failed: {e}")
            return self._perform_fallback_risk_assessment(content, risk_categories)
    
    def _perform_fallback_risk_assessment(self, content: str, risk_categories: List[str]) -> Dict[str, Any]:
        """
        Perform basic risk assessment without LLM.
        
        Args:
            content: Document content to analyze
            risk_categories: Risk categories to check
            
        Returns:
            Dictionary with basic risk assessment
        """
        # Basic keyword-based risk detection
        risk_keywords = {
            "financial": ["loss", "cost", "expense", "liability", "debt", "penalty", "fine"],
            "legal": ["sue", "lawsuit", "breach", "violation", "illegal", "comply", "regulation"],
            "operational": ["failure", "delay", "disruption", "unable", "cannot", "risk"],
            "reputational": ["reputation", "image", "brand", "public", "media", "scandal"],
            "compliance": ["regulation", "standard", "requirement", "audit", "inspection"]
        }
        
        content_lower = content.lower()
        risks = []
        category_counts = {cat: 0 for cat in risk_categories}
        
        # Simple keyword matching
        for category, keywords in risk_keywords.items():
            if category in risk_categories:
                for keyword in keywords:
                    if keyword in content_lower:
                        risks.append({
                            "description": f"Potential {category} risk indicated by keyword: '{keyword}'",
                            "category": category,
                            "severity": 2,  # Medium severity for keyword matches
                            "indicator": keyword
                        })
                        category_counts[category] += 1
        
        # Determine overall risk level
        total_risks = len(risks)
        if total_risks == 0:
            overall_level = "Low"
        elif total_risks <= 3:
            overall_level = "Medium"
        else:
            overall_level = "High"
        
        analysis = f"# Risk Assessment (Fallback Analysis)\n\n"
        analysis += f"Found {total_risks} potential risk indicators using keyword analysis.\n\n"
        analysis += "**Note**: This is a basic analysis. Full LLM-powered analysis would provide more detailed insights.\n"
        
        recommendations = [
            "Review document manually for comprehensive risk assessment",
            "Consider legal review for contracts and agreements",
            "Implement risk monitoring procedures"
        ]
        
        return {
            "analysis": analysis,
            "risks": risks,
            "categories": category_counts,
            "overall_level": overall_level,
            "recommendations": recommendations
        }
    
    def _extract_risks_from_analysis(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract individual risks from LLM analysis.
        
        Args:
            text: Analysis text from LLM
            
        Returns:
            List of risk dictionaries
        """
        risks = []
        try:
            lines = text.split('\n')
            current_risk = None
            
            for line in lines:
                line = line.strip()
                
                # Look for risk entries
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    risk_text = line[1:].strip()
                    if len(risk_text) > 10:  # Filter out short entries
                        # Try to extract category and severity
                        category = "other"
                        severity = 3  # Default medium severity
                        
                        # Simple pattern matching for category
                        if any(word in risk_text.lower() for word in ["financial", "money", "cost"]):
                            category = "financial"
                        elif any(word in risk_text.lower() for word in ["legal", "law", "contract"]):
                            category = "legal"
                        elif any(word in risk_text.lower() for word in ["operational", "operation", "process"]):
                            category = "operational"
                        elif any(word in risk_text.lower() for word in ["reputation", "brand", "image"]):
                            category = "reputational"
                        elif any(word in risk_text.lower() for word in ["compliance", "regulation", "standard"]):
                            category = "compliance"
                        
                        risks.append({
                            "description": risk_text,
                            "category": category,
                            "severity": severity,
                            "indicator": "LLM analysis"
                        })
        
        except Exception as e:
            logger.warning(f"Failed to extract risks: {e}")
        
        return risks[:10]  # Limit to 10 risks
    
    def _extract_overall_risk_level(self, text: str) -> str:
        """
        Extract overall risk level from analysis.
        
        Args:
            text: Analysis text
            
        Returns:
            Risk level string
        """
        text_lower = text.lower()
        
        if "critical" in text_lower:
            return "Critical"
        elif "high" in text_lower:
            return "High"
        elif "medium" in text_lower:
            return "Medium"
        elif "low" in text_lower:
            return "Low"
        else:
            return "Medium"  # Default
    
    def _extract_recommendations_from_analysis(self, text: str) -> List[str]:
        """
        Extract recommendations from analysis text.
        
        Args:
            text: Analysis text
            
        Returns:
            List of recommendations
        """
        recommendations = []
        try:
            lines = text.split('\n')
            in_recommendations = False
            
            for line in lines:
                line = line.strip()
                
                # Look for recommendations section
                if "recommendation" in line.lower() and ('**' in line or '#' in line):
                    in_recommendations = True
                    continue
                
                # Extract recommendation items
                if in_recommendations and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    rec = line[1:].strip()
                    if rec and len(rec) > 5:
                        recommendations.append(rec)
                
                # Stop if we hit another section
                if in_recommendations and line.startswith('#') and "recommendation" not in line.lower():
                    break
        
        except Exception as e:
            logger.warning(f"Failed to extract recommendations: {e}")
        
        # Default recommendations if none found
        if not recommendations:
            recommendations = [
                "Review identified risks with relevant stakeholders",
                "Develop mitigation strategies for high-severity risks",
                "Monitor risk indicators regularly"
            ]
        
        return recommendations[:5]  # Limit to 5 recommendations


# ============================================================================
# MULTI-TOOL RESULT SYNTHESIS TOOL
# ============================================================================

class SynthesizeMultiToolResultsArgs(BaseToolArgs):
    """Arguments for multi-tool result synthesis."""
    tool_results: List[Dict[str, Any]]
    original_query: str
    synthesis_focus: str = "comprehensive"  # "comprehensive", "executive_summary", "actionable"


class SynthesizeMultiToolResultsResult(BaseToolResult):
    """Result for multi-tool result synthesis."""
    synthesized_analysis: str
    original_query: str
    tools_synthesized: List[str]
    key_findings: List[str]
    recommendations: List[str]
    executive_summary: str


class SynthesizeMultiToolResultsTool(BaseTool):
    """
    Advanced tool for synthesizing results from multiple tool executions.
    
    This tool takes results from various tools and creates a coherent,
    comprehensive analysis that addresses the original user query.
    It's designed to work as the final step in complex analytical workflows.
    """
    
    name = "synthesize_multi_tool_results"
    description = (
        "Synthesizes results from multiple previously executed tools into a coherent, "
        "comprehensive analysis or executive summary. Combines different analyses "
        "to provide a holistic view and actionable insights that address the original user query."
    )
    args_schema = SynthesizeMultiToolResultsArgs
    return_schema = SynthesizeMultiToolResultsResult
    version = "1.0.0"
    category = "analysis"
    
    def __init__(self):
        super().__init__()
        # Initialize OpenAI client
        if openai and OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("OpenAI client not available for SynthesizeMultiToolResultsTool")
    
    def _execute(self, tool_results: List[Dict[str, Any]], original_query: str, synthesis_focus: str = "comprehensive") -> Dict[str, Any]:
        """
        Synthesize results from multiple tools.
        
        Args:
            tool_results: List of results from previous tool executions
            original_query: The original user query
            synthesis_focus: Focus of the synthesis ("comprehensive", "executive_summary", "actionable")
            
        Returns:
            Dictionary with synthesized analysis
        """
        logger.info(f"Synthesizing results from {len(tool_results)} tools")
        logger.info(f"Original query: {original_query[:100]}...")
        logger.info(f"Synthesis focus: {synthesis_focus}")
        
        try:
            # Step 1: Validate and process tool results
            processed_results = self._process_tool_results(tool_results)
            
            if not processed_results:
                return {
                    "synthesized_analysis": "No valid tool results to synthesize.",
                    "original_query": original_query,
                    "tools_synthesized": [],
                    "key_findings": ["No results available"],
                    "recommendations": ["Re-run analysis with valid tools"],
                    "executive_summary": "Unable to synthesize - no valid results",
                    "metadata": {"error": "No valid results"}
                }
            
            # Step 2: Perform synthesis
            if self.client:
                synthesis_result = self._perform_llm_synthesis(
                    processed_results, original_query, synthesis_focus
                )
            else:
                synthesis_result = self._perform_fallback_synthesis(
                    processed_results, original_query, synthesis_focus
                )
            
            # Step 3: Structure the final results
            return {
                "synthesized_analysis": synthesis_result["analysis"],
                "original_query": original_query,
                "tools_synthesized": list(processed_results.keys()),
                "key_findings": synthesis_result["findings"],
                "recommendations": synthesis_result["recommendations"],
                "executive_summary": synthesis_result["executive_summary"],
                "metadata": {
                    "tools_count": len(processed_results),
                    "synthesis_focus": synthesis_focus,
                    "analysis_length": len(synthesis_result["analysis"]),
                    "llm_powered": self.client is not None
                }
            }
            
        except Exception as e:
            logger.exception(f"Multi-tool synthesis failed: {e}")
            return {
                "synthesized_analysis": f"Synthesis failed: {str(e)}",
                "original_query": original_query,
                "tools_synthesized": [],
                "key_findings": [f"Error: {str(e)}"],
                "recommendations": ["Check tool results and try again"],
                "executive_summary": f"Synthesis error: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def _process_tool_results(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and validate tool results for synthesis.
        
        Args:
            tool_results: Raw tool results
            
        Returns:
            Processed results dictionary
        """
        processed = {}
        
        try:
            for i, result in enumerate(tool_results):
                if not isinstance(result, dict):
                    continue
                
                # Extract tool name if available
                tool_name = result.get('tool_name', result.get('name', f'tool_{i}'))
                
                # Extract key information
                processed_result = {
                    'raw_result': result,
                    'success': result.get('success', True),
                    'summary': self._extract_result_summary(result),
                    'key_data': self._extract_key_data(result)
                }
                
                processed[tool_name] = processed_result
                logger.debug(f"Processed result from {tool_name}")
        
        except Exception as e:
            logger.warning(f"Error processing tool results: {e}")
        
        return processed
    
    def _extract_result_summary(self, result: Dict[str, Any]) -> str:
        """
        Extract a summary from a tool result.
        
        Args:
            result: Tool result dictionary
            
        Returns:
            Summary string
        """
        # Try different fields that might contain summary information
        summary_fields = [
            'analysis', 'summary', 'result', 'comparison_analysis',
            'risk_analysis', 'synthesized_analysis', 'description'
        ]
        
        for field in summary_fields:
            if field in result and result[field]:
                summary = str(result[field])
                # Return first 500 characters
                return summary[:500] + ('...' if len(summary) > 500 else '')
        
        return "No summary available"
    
    def _extract_key_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key data points from a tool result.
        
        Args:
            result: Tool result dictionary
            
        Returns:
            Dictionary with key data points
        """
        key_data = {}
        
        # Common fields to extract
        key_fields = [
            'key_findings', 'findings', 'recommendations', 'insights',
            'similarities', 'differences', 'risks_identified',
            'overall_risk_level', 'documents_compared'
        ]
        
        for field in key_fields:
            if field in result:
                key_data[field] = result[field]
        
        return key_data
    
    def _perform_llm_synthesis(self, processed_results: Dict[str, Any], original_query: str, focus: str) -> Dict[str, Any]:
        """
        Perform LLM-powered synthesis of tool results.
        
        Args:
            processed_results: Processed tool results
            original_query: Original user query
            focus: Synthesis focus
            
        Returns:
            Dictionary with synthesis results
        """
        try:
            # Prepare synthesis context
            context = f"Original Query: {original_query}\n\n"
            context += "Tool Results:\n\n"
            
            for tool_name, result in processed_results.items():
                context += f"**{tool_name}**:\n"
                context += f"- Success: {result['success']}\n"
                context += f"- Summary: {result['summary']}\n"
                if result['key_data']:
                    context += f"- Key Data: {str(result['key_data'])[:300]}\n"
                context += "\n"
            
            # Create focus-specific prompt
            focus_prompts = {
                "comprehensive": "Provide a comprehensive analysis that integrates all tool results",
                "executive_summary": "Create a concise executive summary highlighting the most important findings",
                "actionable": "Focus on actionable insights and specific recommendations"
            }
            
            focus_instruction = focus_prompts.get(focus, focus_prompts["comprehensive"])
            
            synthesis_prompt = f"""
{context}

Based on the above tool results and the original query, {focus_instruction}.

Provide a structured response that includes:

1. **Executive Summary**: A brief overview of the key findings across all tools

2. **Key Findings**: The most important insights from the combined analysis

3. **Cross-Tool Insights**: Connections and patterns identified across different tool results

4. **Recommendations**: Specific, actionable recommendations based on the comprehensive analysis

5. **Detailed Analysis**: A thorough synthesis that addresses the original query

Ensure your response is coherent, well-structured, and directly addresses the original query.
"""
            
            # Call OpenAI for synthesis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst specializing in synthesizing complex multi-tool analysis results. Create coherent, insightful syntheses that provide clear value to decision-makers."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,  # Slightly higher temperature for creative synthesis
                max_tokens=2500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract structured components
            findings = self._extract_findings_from_synthesis(analysis_text)
            recommendations = self._extract_recommendations_from_analysis(analysis_text)
            executive_summary = self._extract_executive_summary(analysis_text)
            
            return {
                "analysis": analysis_text,
                "findings": findings,
                "recommendations": recommendations,
                "executive_summary": executive_summary
            }
            
        except Exception as e:
            logger.exception(f"LLM synthesis failed: {e}")
            return self._perform_fallback_synthesis(processed_results, original_query, focus)
    
    def _perform_fallback_synthesis(self, processed_results: Dict[str, Any], original_query: str, focus: str) -> Dict[str, Any]:
        """
        Perform basic synthesis without LLM.
        
        Args:
            processed_results: Processed tool results
            original_query: Original user query
            focus: Synthesis focus
            
        Returns:
            Dictionary with basic synthesis
        """
        tool_names = list(processed_results.keys())
        
        analysis = f"# Multi-Tool Analysis Synthesis\n\n"
        analysis += f"**Original Query**: {original_query}\n\n"
        analysis += f"**Tools Used**: {', '.join(tool_names)}\n\n"
        
        analysis += "## Tool Results Summary\n\n"
        
        findings = []
        recommendations = []
        
        for tool_name, result in processed_results.items():
            analysis += f"### {tool_name}\n"
            analysis += f"- **Status**: {'Success' if result['success'] else 'Failed'}\n"
            analysis += f"- **Summary**: {result['summary'][:200]}...\n\n"
            
            # Extract findings and recommendations
            if result['key_data']:
                key_data = result['key_data']
                if 'findings' in key_data:
                    findings.extend(key_data['findings'][:2])  # Take first 2 findings per tool
                if 'recommendations' in key_data:
                    recommendations.extend(key_data['recommendations'][:2])  # Take first 2 recs per tool
        
        # Create executive summary
        executive_summary = f"Analysis of '{original_query}' using {len(tool_names)} tools. "
        successful_tools = sum(1 for result in processed_results.values() if result['success'])
        executive_summary += f"{successful_tools} tools executed successfully. "
        executive_summary += "Detailed synthesis requires LLM analysis for optimal insights."
        
        # Default findings and recommendations if none extracted
        if not findings:
            findings = ["Multiple tools executed on the query", "Basic synthesis provided without LLM"]
        if not recommendations:
            recommendations = ["Review individual tool results for specific insights", "Consider re-running with LLM synthesis for comprehensive analysis"]
        
        return {
            "analysis": analysis,
            "findings": findings[:5],  # Limit findings
            "recommendations": recommendations[:5],  # Limit recommendations
            "executive_summary": executive_summary
        }
    
    def _extract_findings_from_synthesis(self, text: str) -> List[str]:
        """
        Extract key findings from synthesis text.
        
        Args:
            text: Synthesis text
            
        Returns:
            List of key findings
        """
        return self._extract_list_from_analysis(text, "findings")
    
    def _extract_executive_summary(self, text: str) -> str:
        """
        Extract executive summary from synthesis text.
        
        Args:
            text: Synthesis text
            
        Returns:
            Executive summary string
        """
        try:
            lines = text.split('\n')
            in_summary = False
            summary_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Look for executive summary section
                if "executive summary" in line.lower() and ('**' in line or '#' in line):
                    in_summary = True
                    continue
                
                # Stop at next section
                if in_summary and (line.startswith('#') or (line.startswith('**') and line.endswith('**'))):
                    if "executive summary" not in line.lower():
                        break
                
                # Collect summary lines
                if in_summary and line and not line.startswith('#'):
                    summary_lines.append(line)
            
            if summary_lines:
                return ' '.join(summary_lines)[:500]  # Limit length
            else:
                return "Executive summary: Analysis completed using multiple tools with comprehensive results."
        
        except Exception as e:
            logger.warning(f"Failed to extract executive summary: {e}")
            return "Executive summary extraction failed. Please refer to the full analysis."
    
    def _extract_list_from_analysis(self, text: str, section_type: str) -> List[str]:
        """
        Extract lists from analysis text based on section type.
        
        Args:
            text: Analysis text
            section_type: Type of section to extract
            
        Returns:
            List of extracted items
        """
        items = []
        try:
            lines = text.split('\n')
            in_section = False
            
            section_keywords = {
                "findings": ["findings", "insights", "key findings"],
                "recommendations": ["recommendations", "recommend", "suggest"]
            }
            
            keywords = section_keywords.get(section_type, [section_type])
            
            for line in lines:
                line = line.strip()
                
                # Check if we're entering the section
                if any(keyword in line.lower() for keyword in keywords) and ('**' in line or '#' in line):
                    in_section = True
                    continue
                
                # Check if we're leaving the section
                if in_section and (line.startswith('#') or (line.startswith('**') and line.endswith('**'))):
                    if not any(keyword in line.lower() for keyword in keywords):
                        in_section = False
                        continue
                
                # Extract items from the section
                if in_section and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    item = line[1:].strip()
                    if item and len(item) > 5:
                        items.append(item)
            
            # If no items found, provide a default
            if not items:
                items = [f"No specific {section_type} identified"]
                
        except Exception as e:
            logger.warning(f"Failed to extract {section_type}: {e}")
            items = [f"Could not extract {section_type} from analysis"]
        
        return items[:5]  # Limit to 5 items


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the analysis tools with comprehensive scenarios.
    """
    
    print("=== DataNeuron Advanced Analysis Tools Test ===")
    
    # Test 1: Create tool instances
    print(f"\nTest 1: Creating analysis tool instances")
    try:
        compare_tool = CompareDocumentsTool()
        risk_tool = RiskAssessmentTool()
        synthesis_tool = SynthesizeMultiToolResultsTool()
        
        print(f" PASS - All analysis tools created successfully")
        print(f"   CompareDocumentsTool: {compare_tool.name}")
        print(f"   RiskAssessmentTool: {risk_tool.name}")
        print(f"   SynthesizeMultiToolResultsTool: {synthesis_tool.name}")
        
    except Exception as e:
        print(f"L FAIL - Tool creation failed: {e}")
        exit(1)
    
    # Test 2: Schema validation
    print(f"\nTest 2: Schema validation")
    try:
        # Test CompareDocumentsTool schema
        compare_args = CompareDocumentsArgs(
            file_names=["doc1.pdf", "doc2.pdf"],
            comparison_criteria="financial performance",
            session_id="test_session"
        )
        print(f" PASS - CompareDocuments schema validation passed")
        
        # Test RiskAssessmentTool schema  
        risk_args = RiskAssessmentArgs(
            file_name="contract.pdf",
            session_id="test_session"
        )
        print(f" PASS - RiskAssessment schema validation passed")
        
        # Test SynthesizeMultiToolResultsTool schema
        synthesis_args = SynthesizeMultiToolResultsArgs(
            tool_results=[{"tool_name": "test", "result": "test result"}],
            original_query="test query"
        )
        print(f" PASS - SynthesizeMultiToolResults schema validation passed")
        
    except Exception as e:
        print(f"L FAIL - Schema validation failed: {e}")
    
    # Test 3: Mock execution without real session data
    print(f"\nTest 3: Mock tool execution (without real session data)")
    
    # Test CompareDocumentsTool with non-existent session
    try:
        result = compare_tool.execute(
            file_names=["mock_doc1.pdf", "mock_doc2.pdf"],
            comparison_criteria="content themes",
            session_id="non_existent_session"
        )
        
        if not result.success:
            print(f" PASS - CompareDocuments handled missing session gracefully")
            print(f"   Error handled: {result.error_message is not None}")
        else:
            print(f" WARNING - Expected failure due to missing session data")
            
    except Exception as e:
        print(f"L FAIL - CompareDocuments execution failed: {e}")
    
    # Test RiskAssessmentTool with non-existent session
    try:
        result = risk_tool.execute(
            file_name="mock_contract.pdf",
            session_id="non_existent_session"
        )
        
        if not result.success:
            print(f" PASS - RiskAssessment handled missing session gracefully")
            print(f"   Error handled: {result.error_message is not None}")
        else:
            print(f" WARNING - Expected failure due to missing session data")
            
    except Exception as e:
        print(f"L FAIL - RiskAssessment execution failed: {e}")
    
    # Test SynthesizeMultiToolResultsTool with mock data
    try:
        mock_tool_results = [
            {
                "tool_name": "compare_documents",
                "success": True,
                "comparison_analysis": "Documents show similar financial trends but different risk profiles.",
                "similarities": ["Both mention revenue growth", "Similar market focus"],
                "differences": ["Different risk tolerance", "Varied investment strategies"],
                "key_insights": ["Strong correlation in performance metrics"]
            },
            {
                "tool_name": "assess_risks",
                "success": True,
                "risk_analysis": "Identified several financial and operational risks.",
                "overall_risk_level": "Medium",
                "risks_identified": [{"category": "financial", "severity": 3}],
                "recommendations": ["Monitor cash flow closely", "Diversify revenue streams"]
            }
        ]
        
        result = synthesis_tool.execute(
            tool_results=mock_tool_results,
            original_query="Compare these documents and assess their risks",
            synthesis_focus="comprehensive"
        )
        
        if result.success:
            print(f" PASS - SynthesizeMultiToolResults executed successfully")
            print(f"   Tools synthesized: {len(result.tools_synthesized)}")
            print(f"   Key findings: {len(result.key_findings)}")
            print(f"   Executive summary length: {len(result.executive_summary)}")
        else:
            print(f" WARNING - Synthesis completed but with issues: {result.error_message}")
            
    except Exception as e:
        print(f"L FAIL - SynthesizeMultiToolResults execution failed: {e}")
    
    # Test 4: Schema information retrieval
    print(f"\nTest 4: Schema information retrieval")
    try:
        for tool_name, tool in [("CompareDocuments", compare_tool), 
                               ("RiskAssessment", risk_tool),
                               ("SynthesizeMultiToolResults", synthesis_tool)]:
            schema_info = tool.get_schema_info()
            print(f" PASS - {tool_name} schema info retrieved")
            print(f"   Name: {schema_info['name']}")
            print(f"   Category: {schema_info['category']}")
            print(f"   Requires session: {schema_info['requires_session']}")
            print(f"   Input properties: {len(schema_info.get('input_schema', {}).get('properties', {}))}")
            
    except Exception as e:
        print(f"L FAIL - Schema info retrieval failed: {e}")
    
    # Test 5: Error handling with invalid inputs
    print(f"\nTest 5: Error handling with invalid inputs")
    
    # Test with empty file names
    try:
        result = compare_tool.execute(
            file_names=[],  # Empty list
            comparison_criteria="test",
            session_id="test"
        )
        
        if not result.success and "ValidationError" in result.error_type:
            print(f" PASS - Empty file names correctly rejected")
        else:
            print(f" WARNING - Empty file names should have been rejected")
            
    except Exception as e:
        print(f" Expected validation error: {e}")
    
    # Test with invalid synthesis input
    try:
        result = synthesis_tool.execute(
            tool_results="not_a_list",  # Should be list
            original_query="test"
        )
        
        if not result.success and "ValidationError" in result.error_type:
            print(f" PASS - Invalid tool_results correctly rejected")
        else:
            print(f" WARNING - Invalid tool_results should have been rejected")
            
    except Exception as e:
        print(f" Expected validation error: {e}")
    
    print(f"\n=== Test Complete ===")
    print("Advanced Analysis Tools are ready for integration with LLMAgent.")
    print("These tools provide sophisticated document comparison, risk assessment,")
    print("and multi-tool result synthesis capabilities using LLM-powered analysis.")
    print("\nNote: Full functionality requires:")
    print("- Valid OpenAI API key for LLM-powered analysis")
    print("- Active SessionManager with document data")
    print("- Integration with document processing pipeline")