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
from config.settings import ANTHROPIC_MODEL, OPENAI_API_KEY,ANTHROPIC_API_KEY
from pydantic import Field
import openai

# Import dependencies with error handling
try:
    import anthropic
except ImportError as e:
    logger.error(f"Anthropic library not available: {e}")
    anthropic = None

try:
    import tiktoken
except ImportError as e:
    logger.error(f"tiktoken library not available: {e}")
    tiktoken = None

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
        "Compares two or more documents side-by-side based on your specified criteria. "
    "Use this tool when you need to analyze differences and similarities between "
    "documents such as contracts, reports, proposals, or policies. "
    "\n\n"
    "Comparison criteria examples:"
    "\n• 'financial performance and KPIs' - for business reports"
    "\n• 'legal terms and obligations' - for contracts and agreements" 
    "\n• 'technical specifications' - for product documentation"
    "\n• 'risk factors and mitigation' - for risk assessments"
    "\n• 'methodology and approach' - for research papers"
    "\n\n"
    "Output includes structured analysis with similarities table, "
    "differences breakdown, and actionable insights. Perfect for due diligence, "
    "vendor selection, policy reviews, and competitive analysis."
    )
    args_schema = CompareDocumentsArgs
    return_schema = CompareDocumentsResult
    version = "1.0.0"
    category = "analysis"
    requires_session = True
    
    def __init__(self, session_manager=None):
        super().__init__(session_manager=session_manager)
        # Initialize Anthropic client
        if anthropic and ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("Anthropic client not available for CompareDocumentsTool")
    
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
        Perform LLM-powered document comparison using Map-Reduce technique.
        
        Args:
            document_contents: Mapping of file names to contents
            criteria: Comparison criteria
            output_format: Output format preference
            
        Returns:
            Dictionary with analysis, similarities, differences, and insights
        """
        try:
            # Step 1: Map phase - Extract key points from each document
            mapped_results = {}
            for file_name, content in document_contents.items():
                logger.info(f"Mapping document: {file_name}")
                mapped_result = self._map_document(content, criteria)
                mapped_results[file_name] = mapped_result
            
            # Step 2: Reduce phase - Compare the mapped results
            logger.info("Reducing mapped results for comparison")
            comparison_result = self._reduce_comparison(mapped_results, criteria, output_format)
            
            return comparison_result
            
        except Exception as e:
            logger.exception(f"LLM comparison failed: {e}")
            logger.warning("Falling back to basic comparison")
            # Fallback to basic comparison
            return self._perform_fallback_comparison(document_contents, criteria)
    
    def _map_document(self, content: str, criteria: str) -> str:
        """
        Map phase: Extract key points from a single document based on criteria.
        Includes token limit control to prevent API errors.
        
        Args:
            content: Document content to analyze
            criteria: Analysis criteria
            
        Returns:
            Extracted key points as string
        """
        try:
            # Token limit control
            if tiktoken:
                try:
                    # Get encoding for the model
                    encoding = tiktoken.encoding_for_model(self.model)
                    token_count = len(encoding.encode(content))
                    
                    # Check if content exceeds 80% of model's context window
                    # Assuming gpt-4o has ~120k token limit, use 96k as safe threshold
                    max_tokens = 96000
                    if token_count > max_tokens:
                        # Calculate safe character limit (rough approximation: 1 token ≈ 4 chars)
                        safe_char_limit = max_tokens * 4
                        content = content[:safe_char_limit]
                        logger.warning(f"Document content was too long for mapping, truncating to fit context window. Original tokens: {token_count}, Safe limit: {max_tokens}")
                except Exception as e:
                    logger.warning(f"Token counting failed: {e}, using character limit fallback")
                    # Fallback to character limit if tiktoken fails
                    if len(content) > 200000:  # ~50k tokens
                        content = content[:200000]
                        logger.warning("Document content was too long for mapping, truncating to fit context window.")
            else:
                # Fallback when tiktoken is not available
                if len(content) > 200000:
                    content = content[:200000]
                    logger.warning("Document content was too long for mapping, truncating to fit context window.")
            
            # Document Analyst prompt
            map_prompt = f"""
**ROLE:** Senior Document Intelligence Analyst
**OBJECTIVE:** Extract and structure key information from the document specifically relevant to "{criteria}" for comparative analysis.

**DOCUMENT TO ANALYZE:**
{content}

**ANALYSIS FRAMEWORK:**

**Phase 1 - Content Identification:**
Scan the document and identify all sections, data points, statements, and evidence directly related to "{criteria}".

**Phase 2 - Structured Extraction:**
Organize your findings into the following categories:

**🎯 PRIMARY FINDINGS (related to "{criteria}"):**
- [Extract 5-8 most critical points, facts, or statements]
- [Include specific data, numbers, percentages, dates where relevant]
- [Focus on decision-making information]

**📊 QUANTITATIVE DATA:**
- [Any metrics, KPIs, financial figures, percentages, quantities]
- [Comparative benchmarks or performance indicators]
- [Timeline information or deadlines]

**🔍 QUALITATIVE INSIGHTS:**
- [Strategic approaches, methodologies, philosophies]
- [Risk factors, opportunities, or concerns mentioned]
- [Unique characteristics or competitive advantages]

**⚖️ COMPLIANCE & CONSTRAINTS:**
- [Rules, regulations, requirements, or limitations]
- [Approval processes, governance structures]
- [Legal or contractual obligations]

**🚀 FORWARD-LOOKING ELEMENTS:**
- [Goals, targets, projections, or future plans]
- [Recommendations or proposed actions]
- [Success criteria or milestones]

**EXTRACTION RULES:**
- Be precise and factual - no interpretation or opinion
- Include specific quotes when they provide key insights
- Preserve context for numbers and data points
- Note any missing information relevant to "{criteria}"
- Maximum 15 bullet points total across all categories
- Use exact terminology from the document

**STRUCTURED KEY POINTS EXTRACTION:**
"""

            # Call Anthropic for document mapping
            response = self.client.messages.create(
                model=self.model,
                system="""You are a confident document analysis expert with direct access to document content. 

KEY DIRECTIVES:
- Make DEFINITIVE statements about document contents
- Use CONFIDENT language: "contains", "includes", "focuses on", "outlines"
- NEVER use uncertain language: "may", "might", "likely", "appears", "seems"
- Extract SPECIFIC details: names, dates, technologies, objectives, requirements
- Respond in the SAME language as the user's question
- Base analysis ONLY on the provided document content

You have successfully extracted the document content. Provide a comprehensive, confident analysis of what the document actually contains.""",
                messages=[
                    {"role": "user", "content": map_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.exception(f"Document mapping failed: {e}")
            # Return truncated content as fallback
            return f"Key points extraction failed. Raw content (truncated): {content[:500]}..."
    
    def _reduce_comparison(self, mapped_results: Dict[str, str], criteria: str, output_format: str) -> Dict[str, Any]:
        """
        Reduce phase: Compare the mapped results from multiple documents.
        Includes token limit control for the combined mapped results.
        
        Args:
            mapped_results: Dictionary mapping file names to their extracted key points
            criteria: Comparison criteria
            output_format: Output format preference
            
        Returns:
            Dictionary with analysis, similarities, differences, and insights
        """
        try:
            # Combine all mapped results
            combined_mapped = ""
            for file_name, key_points in mapped_results.items():
                combined_mapped += f"**Document: {file_name}**\n{key_points}\n\n---\n\n"
            
            # Token limit control for combined results
            if tiktoken:
                try:
                    encoding = tiktoken.encoding_for_model(self.model)
                    token_count = len(encoding.encode(combined_mapped))
                    
                    # Check if combined results exceed safe threshold
                    max_tokens = 80000  # Conservative limit for reduce phase
                    if token_count > max_tokens:
                        # Truncate each document's key points proportionally
                        char_limit_per_doc = max_tokens * 4 // len(mapped_results)
                        truncated_mapped = ""
                        for file_name, key_points in mapped_results.items():
                            truncated_points = key_points[:char_limit_per_doc]
                            truncated_mapped += f"**Document: {file_name}**\n{truncated_points}\n\n---\n\n"
                        combined_mapped = truncated_mapped
                        logger.warning(f"Combined mapped results were too long for reduction, truncating each document. Original tokens: {token_count}, Safe limit: {max_tokens}")
                except Exception as e:
                    logger.warning(f"Token counting failed during reduce: {e}, using character limit fallback")
                    # Fallback truncation
                    if len(combined_mapped) > 150000:
                        char_limit_per_doc = 150000 // len(mapped_results)
                        truncated_mapped = ""
                        for file_name, key_points in mapped_results.items():
                            truncated_points = key_points[:char_limit_per_doc]
                            truncated_mapped += f"**Document: {file_name}**\n{truncated_points}\n\n---\n\n"
                        combined_mapped = truncated_mapped
                        logger.warning("Combined mapped results were too long for reduction, truncating each document.")
            
            # Lead Analyst prompt for comparison
            reduce_prompt = reduce_prompt = f"""
**ROLE:** Lead Strategic Analyst & Synthesis Expert
**MISSION:** Synthesize extracted intelligence to deliver comprehensive comparative analysis for "{criteria}".

**INTELLIGENCE BRIEFINGS FROM FIELD ANALYSTS:**
{combined_mapped}

**SYNTHESIS PROTOCOL:**

**📋 EXECUTIVE SUMMARY**
Provide a concise strategic overview of how these documents align, compete, or complement each other regarding "{criteria}". Include overall assessment and key strategic implications.

**🔗 CONVERGENCE ANALYSIS (Key Similarities)**
Identify strategic alignments and common elements:
- Shared approaches, methodologies, or frameworks
- Common data patterns, metrics, or performance indicators  
- Aligned objectives, priorities, or strategic directions
- Consistent regulatory/compliance requirements
- Similar risk profiles or mitigation strategies

**⚡ DIVERGENCE ANALYSIS (Key Differences)**
Highlight critical distinctions and competitive differentiators:
- Contrasting strategies, approaches, or methodologies
- Significant data variations, performance gaps, or metric differences
- Different priorities, objectives, or strategic focus areas
- Varying compliance requirements or regulatory approaches
- Distinct risk tolerances, opportunities, or challenge areas

**💡 STRATEGIC INTELLIGENCE (Critical Insights)**
Provide high-value analytical observations:
- Competitive advantages or disadvantages identified
- Best practices or optimization opportunities
- Risk-reward assessments and strategic implications
- Market positioning or differentiation insights
- Recommended actions or strategic considerations

**📊 DETAILED COMPARATIVE MATRIX**
{f'''Create a structured markdown comparison table highlighting:
- Key criteria dimensions as rows
- Document names as columns  
- Specific findings, data points, or approaches in cells
- Clear visual distinction of similarities vs differences''' if output_format == 'markdown' else '''Provide detailed section-by-section comparative analysis:
- Break down comparison by major criteria components
- Include specific evidence and data points
- Highlight decision-making implications
- Structure with clear subheadings and bullet points'''}

**ANALYTICAL STANDARDS:**
✓ Evidence-based conclusions only - cite specific findings
✓ Strategic context - focus on decision-making implications
✓ Balanced perspective - highlight both strengths and gaps
✓ Actionable insights - provide practical strategic value
✓ Precision in language - use exact terminology from source material

**SYNTHESIS REPORT:**
"""

            # Call Anthropic for comparison reduction
            response = self.client.messages.create(
                model=self.model,
                system="You are a Lead Analyst specializing in comparative analysis. Synthesize extracted information to provide comprehensive, structured comparisons highlighting key similarities and differences.",
                messages=[
                    {"role": "user", "content": reduce_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            analysis_text = response.content[0].text
            
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
            logger.exception(f"Comparison reduction failed: {e}")
            # Create basic comparison from mapped results
            return self._create_basic_comparison_from_mapped(mapped_results, criteria)
    
    def _create_basic_comparison_from_mapped(self, mapped_results: Dict[str, str], criteria: str) -> Dict[str, Any]:
        """
        Create a basic comparison when LLM reduction fails.
        
        Args:
            mapped_results: Mapped document results
            criteria: Comparison criteria
            
        Returns:
            Basic comparison dictionary
        """
        file_names = list(mapped_results.keys())
        
        analysis = f"# Document Comparison: {criteria}\n\n"
        analysis += f"Comparing {len(file_names)} documents: {', '.join(file_names)}\n\n"
        
        analysis += "## Extracted Key Points\n\n"
        for file_name, key_points in mapped_results.items():
            analysis += f"### {file_name}\n{key_points[:300]}...\n\n"
        
        similarities = ["All documents contain relevant content for the specified criteria"]
        differences = [f"Each document has unique characteristics and focus areas"]
        insights = ["Detailed comparison requires successful LLM analysis", f"Comparison criteria: {criteria}"]
        
        return {
            "analysis": analysis,
            "similarities": similarities,
            "differences": differences,
            "insights": insights
        }
    
    def _perform_fallback_comparison(self, document_contents: Dict[str, str], criteria: str) -> Dict[str, Any]:
        """
        Perform basic comparison without LLM when Anthropic is not available.

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
        #differences = [f"Document lengths vary: {', '.join([f'{name} ({stats[name]['words']} words)' for name in file_names])}"]
        details_list = [f"{name} ({stats[name]['words']} words)" for name in file_names]
        differences = [f"Document lengths vary: {', '.join(details_list)}"]
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
                "similarities": ["similarities", "similar", "common","CONVERGENCE ANALYSIS", "Key Similarities"],
                "differences": ["differences", "different", "differ","DIVERGENCE ANALYSIS", "Key Differences"],
                "insights": ["insights", "observations", "analysis","STRATEGIC INTELLINTELLIGENCE", "Critical Insights", "Important Insights"]
            }
            
            keywords = section_keywords.get(section_type, [])


            target_headers = [h.lower() for h in section_keywords.get(section_type, [])]
            if not target_headers:
                return [f"No headers defined for section type: {section_type}"]

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
    risk_categories: List[str] = [
    "financial", "operational", "strategic", "technological",
    "legal", "compliance", "regulatory", "ethical", 
    "reputational", "market", "environmental", "geopolitical",
    "human_resources", "organizational", "safety", "quality",
    "project", "vendor", "process", "performance"
]


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
    
    def __init__(self, session_manager=None):
        super().__init__(session_manager=session_manager)
        # Initialize Anthropic client
        if anthropic and ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("Anthropic client not available for RiskAssessmentTool")
    
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
**ROLE:** Senior Risk Intelligence Analyst
**MISSION:** Conduct comprehensive risk assessment of document content across {len(risk_categories)} risk categories.

**DOCUMENT FOR ANALYSIS:**
{analysis_content}

**RISK ASSESSMENT FRAMEWORK:**
Target Categories: {', '.join(risk_categories)}

**ANALYSIS PROTOCOL:**

**🚨 RISK IDENTIFICATION MATRIX**
For each identified risk, provide:
- **Risk Description:** Clear, specific risk statement
- **Category:** Primary category from [{', '.join(risk_categories)}]
- **Severity Score:** 1 (Minor) → 5 (Critical)
- **Evidence Quote:** Exact text passage indicating this risk
- **Probability Assessment:** High/Medium/Low likelihood
- **Impact Scope:** Localized/Department/Organization/Enterprise

**🎯 CATEGORY-SPECIFIC RISK ANALYSIS**
Systematically scan for:

**Financial Risks:** Cost overruns, revenue shortfalls, cash flow issues, budget constraints, financial obligations
**Operational Risks:** Process failures, capacity limitations, supply chain disruptions, efficiency gaps
**Legal Risks:** Contract violations, compliance breaches, litigation exposure, regulatory violations
**Reputational Risks:** Brand damage, stakeholder trust erosion, public perception threats
**Strategic Risks:** Market positioning threats, competitive disadvantages, business model vulnerabilities
**Technological Risks:** System failures, cybersecurity threats, data breaches, technology obsolescence
**Compliance Risks:** Policy violations, audit findings, standard deviations, regulatory non-compliance
**Human Resources Risks:** Talent loss, skills gaps, workforce instability, safety concerns

**📊 RISK ASSESSMENT OUTPUTS**

**1. OVERALL RISK RATING**
Assign comprehensive risk level: "Low" | "Medium" | "High" | "Critical"
Justification: Based on highest severity risks and cumulative exposure

**2. RISK INVENTORY**
List 8-12 most significant risks with:
- Precise risk description
- Primary category assignment
- Severity rating (1-5 scale)
- Supporting evidence quote
- Potential impact assessment

**3. CATEGORY RISK PROFILE**
Risk distribution across categories:
- High-risk categories (3+ severity risks)
- Medium-risk categories (1-2 severity risks)
- Low-risk categories (minor or no risks)
- Category-specific risk counts

**4. RED FLAG INDICATORS**
Critical phrases/clauses that signal immediate attention:
- Contract termination clauses
- Financial penalty provisions
- Compliance violation indicators
- Performance failure triggers
- Liability exposure statements

**5. RISK MITIGATION ROADMAP**
Strategic recommendations for:
- Immediate action items (Critical/High severity)
- Medium-term mitigation strategies
- Monitoring and control mechanisms
- Stakeholder communication requirements
- Risk transfer or insurance considerations

**ASSESSMENT STANDARDS:**
✓ Evidence-based risk identification with direct text citations
✓ Quantified severity assessment using 1-5 scale
✓ Category-specific risk pattern recognition
✓ Forward-looking impact assessment
✓ Actionable mitigation recommendations

**RISK INTELLIGENCE REPORT:**
"""

            # Call Anthropic for risk assessment
            response = self.client.messages.create(
                model=self.model,
                system="You are an expert risk analyst specializing in document risk assessment. Identify potential risks, uncertainties, and negative indicators with high accuracy.",
                messages=[
                    {"role": "user", "content": risk_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            analysis_text = response.content[0].text
            
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
# RESULT SYNTHESIS TOOL
# ============================================================================

class SynthesizeResultsArgs(BaseToolArgs):
    """Arguments for result synthesis tool."""
    tool_results: List[Dict[str, Any]] = Field(
        description="A list of dictionary-like results from previously executed tools. Each result should ideally contain keys like 'tool_name', 'success', and the tool's specific output."
    )
    original_query: str = Field(
        description="The original user query that initiated the entire reasoning process. This provides essential context for the final synthesis."
    )


class SynthesizeResultsResult(BaseToolResult):
    """Result for result synthesis tool."""
    synthesis: str = Field(
        description="The final, synthesized answer that coherently combines all insights to address the original query."
    )
    key_findings: List[str] = Field(
        description="A bulleted list of the most important, high-level findings derived from the synthesis."
    )
    confidence_score: float = Field(
        description="The agent's confidence in the quality and completeness of the synthesized answer, on a scale of 0.0 to 1.0.",
        ge=0.0,
        le=1.0
    )


class SynthesizeResultsTool(BaseTool):
    """
    Tool for synthesizing results from multiple previously executed tools.
    
    This tool combines insights from various tools to create a final, coherent
    answer that addresses the original user query. It's designed to be used as
    the final step in multi-step reasoning workflows.
    """
    
    name = "synthesize_results"
    description = (
        "Synthesizes results from a list of previously executed tools into a final, "
        "comprehensive answer to the user's original query. Use this as the final step "
        "in a multi-step plan to combine all gathered information into a coherent response."
    )
    args_schema = SynthesizeResultsArgs
    return_schema = SynthesizeResultsResult
    version = "1.0.0"
    category = "analysis"
    
    def __init__(self, session_manager=None):
        super().__init__(session_manager=session_manager)
        # Initialize Anthropic client
        if anthropic and ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("Anthropic client not available for SynthesizeResultsTool")
    
    def _execute(self, tool_results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """
        Synthesize results from multiple tools into a final coherent answer.
        
        Args:
            tool_results: List of results from previous tool executions
            original_query: The original user query that initiated the reasoning process
            
        Returns:
            Dictionary with synthesis, key findings, and confidence score
        """
        logger.info(f"Synthesizing results from {len(tool_results)} tools")
        logger.info(f"Original query: {original_query[:100]}...")




        
        try:
            # Step 1: Validate and process tool results
            processed_results = self._process_tool_results(tool_results)
            
            if not processed_results:
                return {
                    "synthesis": "No valid tool results to synthesize. Unable to provide a meaningful analysis.",
                    "key_findings": ["No results available for synthesis"],
                    "confidence_score": 0.0,
                    "metadata": {"error": "No valid results", "tools_processed": 0}
                }
            
            # Step 2: Perform synthesis using LLM or fallback
            if self.client:
                synthesis_result = self._perform_llm_synthesis(processed_results, original_query)
            else:
                synthesis_result = self._perform_fallback_synthesis(processed_results, original_query)
            
            # Step 3: Structure the final results according to return schema
            final_result = {
                "synthesis": synthesis_result["synthesis"],
                "key_findings": synthesis_result["key_findings"],
                "confidence_score": synthesis_result["confidence_score"],
                "metadata": {
                    "tools_processed": len(processed_results),
                    "llm_powered": self.client is not None,
                    "synthesis_length": len(synthesis_result["synthesis"])
                }
            }
            
            logger.info(f"SynthesizeResultsTool execution completed successfully")
            logger.info(f"Final result contains synthesis of {len(final_result['synthesis'])} characters")
            logger.debug(f"Returning final result with keys: {list(final_result.keys())}")
            
            return final_result
            
        except Exception as e:
            logger.exception(f"Result synthesis failed: {e}")
            return {
                "synthesis": f"Synthesis failed due to an error: {str(e)}",
                "key_findings": [f"Error occurred during synthesis: {str(e)}"],
                "confidence_score": 0.0,
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
        
        # Debug: Log the result structure
        logger.info(f"Extracting summary from result with keys: {list(result.keys())}")
        
        for field in summary_fields:
            if field in result and result[field]:
                summary = str(result[field])
                logger.info(f"Found summary in field '{field}': {summary[:100]}...")
                # Return first 500 characters
                return summary[:500] + ('...' if len(summary) > 500 else '')
        
        # If no field found, try to return any meaningful content
        if result:
            logger.warning(f"No standard summary field found. Available fields: {list(result.keys())}")
            # Try to find any text content
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 20:
                    logger.info(f"Using content from field '{key}' as fallback")
                    return str(value)[:500] + ('...' if len(str(value)) > 500 else '')
        
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
    
    def _perform_llm_synthesis(self, processed_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Performs a focused LLM synthesis using only the raw content from tool results,
        avoiding meta-conversation about the tools themselves.
        """
        try:
            # === SESSION MEMORY ENTEGRASYONU ===
            session_facts_str = ""
            
            # 1. Hafıza verisini bul ve formatla
            for tool_name, result in processed_results.items():
                if result.get('success') and result.get('raw_result'):
                    raw_result = result.get('raw_result')
                    
                    # get_session_memory'den gelen sonucu tespit et
                    if isinstance(raw_result, dict) and 'memory_data' in raw_result:
                        memory_data = raw_result.get('memory_data')
                        if memory_data and isinstance(memory_data, dict) and memory_data:
                            facts = []
                            for key, value in memory_data.items():
                                # XML güvenli formatla
                                safe_key = str(key).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                                safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                                facts.append(f"  <fact key=\"{safe_key}\">{safe_value}</fact>")
                            
                            if facts:
                                session_facts_str = "<session_facts>\n" + "\n".join(facts) + "\n</session_facts>"
                                logger.info(f"Found session facts to inject into prompt: {memory_data}")
                            break
            
            # === WEB SEARCH KAYNAKLARINI TESPIT ET ===
            web_search_sources = []
            
            # İkinci döngü: Web search sonuçlarını tespit et ve kaynakları çıkar
            for tool_name, result in processed_results.items():
                if result.get('success'):
                    raw_result = result.get('raw_result')
                    
                    # Dict formatında sonuç kontrolü
                    if isinstance(raw_result, dict):
                        # WebSearchTool'dan gelen sonucu tespit et
                        if ('sources' in raw_result and 'search_results' in raw_result and 
                            'query' in raw_result and isinstance(raw_result.get('search_results'), list)):
                            logger.info(f"Detected WebSearchTool result from {tool_name}")
                            
                            # search_results'tan title ve url bilgilerini çıkar
                            search_results = raw_result.get('search_results', [])
                            for search_result in search_results:
                                if isinstance(search_result, dict):
                                    title = search_result.get('title', 'Başlıksız')
                                    url = search_result.get('url', '')
                                    if url and title:
                                        web_search_sources.append({
                                            'title': title,
                                            'url': url
                                        })
                                        logger.debug(f"Extracted web source: {title} -> {url}")
                    
                    # Pydantic nesnesi kontrolü
                    elif hasattr(raw_result, 'sources') and hasattr(raw_result, 'search_results'):
                        logger.info(f"Detected WebSearchTool Pydantic result from {tool_name}")
                        
                        # Pydantic nesnesinden search_results'ı al
                        search_results = getattr(raw_result, 'search_results', [])
                        if isinstance(search_results, list):
                            for search_result in search_results:
                                if isinstance(search_result, dict):
                                    title = search_result.get('title', 'Başlıksız')
                                    url = search_result.get('url', '')
                                    if url and title:
                                        web_search_sources.append({
                                            'title': title,
                                            'url': url
                                        })
                                        logger.debug(f"Extracted web source from Pydantic: {title} -> {url}")

            # === BAĞLAM OLUŞTURMA ===
            final_context = ""
            relevant_documents_info = []

            for tool_name, result in processed_results.items():
                if result.get('success'):
                    raw_result = result.get('raw_result')
                    content_to_add = ""
                    source_name = "Unknown Source"

                    logger.debug(f"Processing result from {tool_name}: {type(raw_result)}")
                    
                    if isinstance(raw_result, dict):
                        # ReadFullDocumentTool'dan gelen içeriği ara
                        if 'content' in raw_result:
                            content_to_add = raw_result['content']
                            logger.debug(f"Found 'content' field in {tool_name}")
                        # SummarizeDocumentTool'dan gelen summary içeriğini ara
                        elif 'summary' in raw_result:
                            content_to_add = raw_result['summary']
                            logger.debug(f"Found 'summary' field in {tool_name}")
                        # WebSearchTool'dan gelen summary içeriğini ara
                        elif 'query' in raw_result and 'search_results' in raw_result:
                            # Web search summary'sini kullan
                            content_to_add = raw_result.get('summary', '')
                            logger.debug(f"Found WebSearchTool summary in {tool_name}")
                        # SearchInDocumentTool'dan gelen içeriği ara - KAYNAK BİLGİSİYLE ZENGİNLEŞTİR
                        elif 'search_results' in raw_result and isinstance(raw_result['search_results'], list):
                            search_results = raw_result['search_results']
                            enriched_content_parts = []
                            
                            for res in search_results:
                                content_text = res.get('content', '')
                                page_number = res.get('page_number', 'Bilinmiyor')
                                file_name = raw_result.get('metadata', {}).get('file_name', 'Bilinmeyen Dosya')
                                
                                # Zenginleştirilmiş format oluştur
                                enriched_part = f"[ALINTI BAŞLANGICI - Kaynak: {file_name}, Sayfa: {page_number}]\n{content_text}\n[ALINTI SONU]"
                                enriched_content_parts.append(enriched_part)
                            
                            content_to_add = "\n\n".join(enriched_content_parts)
                            logger.debug(f"Found 'search_results' field in {tool_name} - enriched with source info")
                        # CompareDocumentsTool'dan gelen analiz içeriğini ara
                        elif 'comparison_analysis' in raw_result:
                            content_to_add = raw_result['comparison_analysis']
                            logger.debug(f"Found 'comparison_analysis' field in {tool_name}")
                        # RiskAssessmentTool'dan gelen risk analizi içeriğini ara
                        elif 'risk_analysis' in raw_result:
                            content_to_add = raw_result['risk_analysis']
                            logger.debug(f"Found 'risk_analysis' field in {tool_name}")
                        # Generic content extraction - check common field names
                        else:
                            content_fields = ['analysis', 'result', 'synthesized_analysis', 'text', 'description']
                            for field in content_fields:
                                if field in raw_result and raw_result[field]:
                                    content_to_add = str(raw_result[field])
                                    logger.debug(f"Found '{field}' field in {tool_name}")
                                    break
                        
                        # Try to extract source name from various fields
                        source_name = (
                            raw_result.get('file_info', {}).get('file_name') or
                            raw_result.get('file_name') or 
                            raw_result.get('source') or
                            tool_name
                        )
                    
                    elif hasattr(raw_result, 'search_results'): # SearchInDocumentTool Pydantic result
                        search_results = getattr(raw_result, 'search_results', [])
                        if isinstance(search_results, list):
                            enriched_content_parts = []
                            
                            for res in search_results:
                                if isinstance(res, dict):
                                    content_text = res.get('content', '')
                                    page_number = res.get('page_number', 'Bilinmiyor')
                                else:
                                    # Pydantic SearchResult nesnesi
                                    content_text = getattr(res, 'content', '')
                                    page_number = getattr(res, 'page_number', 'Bilinmiyor')
                                
                                file_name = getattr(raw_result, 'file_name', 'Bilinmeyen Dosya')
                                if hasattr(raw_result, 'metadata') and raw_result.metadata:
                                    file_name = raw_result.metadata.get('file_name', file_name)
                                
                                # Zenginleştirilmiş format oluştur
                                enriched_part = f"[ALINTI BAŞLANGICI - Kaynak: {file_name}, Sayfa: {page_number}]\n{content_text}\n[ALINTI SONU]"
                                enriched_content_parts.append(enriched_part)
                            
                            content_to_add = "\n\n".join(enriched_content_parts)
                            logger.debug(f"Found Pydantic search_results in {tool_name} - enriched with source info")
                        else:
                            content_to_add = ""
                        source_name = getattr(raw_result, 'file_name', tool_name)
                    elif hasattr(raw_result, 'content'): # Pydantic nesnesi ise
                        content_to_add = raw_result.content
                        logger.debug(f"Found Pydantic .content attribute in {tool_name}")
                        if hasattr(raw_result, 'file_info') and raw_result.file_info:
                             source_name = raw_result.file_info.get('file_name', tool_name)
                    elif hasattr(raw_result, 'summary'): # SummarizeDocumentTool veya WebSearchTool Pydantic result
                        content_to_add = raw_result.summary
                        logger.debug(f"Found Pydantic .summary attribute in {tool_name}")
                        source_name = getattr(raw_result, 'file_name', tool_name)
                    elif hasattr(raw_result, 'comparison_analysis'): # CompareDocumentsTool Pydantic result
                        content_to_add = raw_result.comparison_analysis
                        logger.debug(f"Found Pydantic .comparison_analysis attribute in {tool_name}")
                        source_name = tool_name
                    elif hasattr(raw_result, 'risk_analysis'): # RiskAssessmentTool Pydantic result
                        content_to_add = raw_result.risk_analysis
                        logger.debug(f"Found Pydantic .risk_analysis attribute in {tool_name}")
                        source_name = tool_name

                    if content_to_add and content_to_add.strip():
                        logger.info(f"Successfully extracted content from {tool_name}: {len(content_to_add)} characters")
                        final_context += f"--- START OF CONTEXT FROM: {source_name} ---\n"
                        final_context += content_to_add.strip() + "\n"
                        final_context += f"--- END OF CONTEXT FROM: {source_name} ---\n\n"
                        relevant_documents_info.append(source_name)
                    else:
                        logger.warning(f"No extractable content found in {tool_name} result. Available keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")

            if not final_context.strip():
                logger.warning("No actual content found in successful tool results for synthesis.")
                logger.debug(f"Processed {len(processed_results)} results, none contained extractable content")
                return self._perform_fallback_synthesis(processed_results, original_query)
            
            logger.info(f"Successfully assembled final context with {len(relevant_documents_info)} sources: {relevant_documents_info}")
            logger.info(f"Final context length: {len(final_context)} characters")
            logger.info(f"Web search sources found: {len(web_search_sources)}")

            # === KAYNAK METNİNİ HAZIRLA ===
            sources_markdown = ""
            if web_search_sources:
                sources_markdown = "\n\n---\n**Kaynaklar:**\n"
                for source in web_search_sources:
                    sources_markdown += f"* [{source['title']}]({source['url']})\n"
                logger.info(f"Formatted {len(web_search_sources)} web sources for inclusion")

            # === YENİ GÜÇLÜ EMREDİCİ PROMPT FORMATI ===
            synthesis_prompt = ""
            
            # 1. KESİN GERÇEKLER (Session Facts) - En başa ve emredici formatta
            if session_facts_str:
                synthesis_prompt += "**GÖREV TALİMATLARI - BUNLARA HARFİYEN UYULACAK:**\n\n"
                synthesis_prompt += "**1. KESİN GERÇEKLER (SESSION FACTS):**\n"
                
                # Session facts'i processed_results'dan al ve daha okunabilir formatta listele
                session_facts = {}
                for tool_name, result in processed_results.items():
                    if result.get('success') and result.get('raw_result'):
                        raw_result = result.get('raw_result')
                        if isinstance(raw_result, dict) and 'memory_data' in raw_result:
                            session_facts = raw_result.get('memory_data', {})
                            break
                
                for key, value in session_facts.items():
                    synthesis_prompt += f"   - {key}: {value}\n"
                
                synthesis_prompt += f"\n**2. ANA GÖREV:**\n   Kullanıcının şu sorusuna cevap ver: \"{original_query}\"\n\n"
                
                synthesis_prompt += f"**3. KANIT MATERYALLERİ (DOKÜMAN İÇERİĞİ):**\n"
                synthesis_prompt += f"```text\n{final_context}\n```\n\n"
                
                synthesis_prompt += "**4. CEVAP ÜRETME KURALLARI:**\n"
                synthesis_prompt += "   - Cevabını oluştururken, **KESİNLİKLE VE SADECE** \"KESİN GERÇEKLER\" bölümündeki bilgileri kullan.\n"
                synthesis_prompt += "   - Eğer kullanıcının sorusu bu kesin gerçeklerle ilgiliyse, \"KANIT MATERYALLERİ\" bölümünü **göz ardı et.**\n"
                synthesis_prompt += "   - Cevabında ASLA \"hafızaya göre\", \"bana verilen bilgiye göre\" gibi ifadeler kullanma. Doğrudan bilgiyi ver.\n"
                synthesis_prompt += "   - Kaynak referansı için [dosya_adı, Sayfa X] formatını kullan.\n"
                synthesis_prompt += "   - Kullanıcının diliyle (Türkçe/İngilizce) cevap ver.\n"
            else:
                # Hafıza yoksa normal format
                synthesis_prompt += f"""**ROLE & MISSION:** You are an expert analyst. Your mission is to provide a direct, detailed, and factual answer to the user's query based ONLY on the provided context.

**USER'S QUERY:**
"{original_query}"

**RELEVANT CONTEXT FROM DOCUMENT(S):**
```text
{final_context}
```

**INSTRUCTIONS:**
1.  **Directly answer the user's query.**
2.  Base your entire answer **exclusively** on the "RELEVANT CONTEXT" provided above. Do not use any outside knowledge.
3.  **Be specific.** Quote or reference key information from the context if it helps to answer the question.
4.  Respond in the same language as the user's query.
5.  **CRITICAL RULE: Do not talk about the tools, the process, or the act of analysis.** Do not say "Based on the analysis and tool execution...". Just provide the answer as if you are an expert who has read the document(s) directly.
6.  **MUTLAK KURAL - KAYNAK REFERANSI:** Cevabını oluştururken, kullandığın her bilgi için, o bilginin alındığı [ALINTI BAŞLANGICI - Kaynak: ...] etiketindeki dosya adını ve sayfa numarasını, cümlenin veya paragrafın sonuna **[dosya_adı, Sayfa X]** formatında eklemek ZORUNDASIN. Bu, cevabının doğrulanabilirliği için hayati önem taşır. ASLA kaynak bilgisi olmadan bir iddiada bulunma. Örnek: "Projenin ana hedefi gelişmiş RAG sistemi oluşturmaktır **[Project Proposal.pdf, Sayfa 2]**."
"""

            # Eğer web search kaynakları varsa, prompt'a kaynak ekleme talimatı ekle
            if sources_markdown:
                synthesis_prompt += f"""7.  **ÖNEMLİ: Cevabını bitirdikten sonra, aşağıda `[KAYNAKLAR]` bölümünde sana verdiğim metni HİÇBİR DEĞİŞİKLİK YAPMADAN cevabının sonuna ekle.**

[KAYNAKLAR]
{sources_markdown.strip()}
"""

            synthesis_prompt += "\n\n**FINAL ANSWER:**"

            # API çağrısı - Güçlü emredici system prompt
            system_prompt = "You are an expert analyst who provides direct answers. CRITICAL PRIORITY RULES: 1) If 'KESİN GERÇEKLER (SESSION FACTS)' section exists, treat it as ABSOLUTE TRUTH and prioritize it over ALL document content. 2) Never mention the process of analysis, only provide results. 3) Include source references in [file_name, Page X] format. 4) Follow the GÖREV TALİMATLARI instructions exactly."
            
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,  # Yaratıcı ve akıcı cevaplar için artırıldı
                max_tokens=2000
            )

            analysis_text = response.content[0].text.strip()

            key_findings = self._extract_key_findings_from_analysis(analysis_text)
            confidence_score = self._calculate_confidence_score(processed_results)
            
            logger.info(f"LLM synthesis completed successfully. Response length: {len(analysis_text)} characters")
            logger.info(f"Key findings extracted: {len(key_findings)}, Confidence score: {confidence_score}")
            logger.debug(f"Synthesis preview: {analysis_text[:200]}...")
            
            return {
                "synthesis": analysis_text,
                "key_findings": key_findings,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.exception(f"Enhanced LLM synthesis failed: {e}")
            return self._perform_fallback_synthesis(processed_results, original_query)

    def _extract_key_findings_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract key findings from analysis text."""
        try:
            # Look for bullet points, numbered lists, or key statements
            findings = []
            
            # Split by common separators
            lines = analysis_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for bullet points or numbered items
                if (line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')) or
                    ('önemli' in line.lower() and len(line) > 20) or
                    ('important' in line.lower() and len(line) > 20) or
                    ('key' in line.lower() and len(line) > 20)):
                    
                    # Clean up the line
                    clean_line = line.lstrip('•-*123456789. ').strip()
                    if len(clean_line) > 10 and len(clean_line) < 200:
                        findings.append(clean_line)
            
            # If no structured findings found, extract first few sentences
            if not findings:
                sentences = analysis_text.split('. ')
                for sentence in sentences[:3]:
                    if len(sentence.strip()) > 20:
                        findings.append(sentence.strip() + '.')
            
            return findings[:5]  # Limit to 5 key findings
            
        except Exception as e:
            logger.warning(f"Key findings extraction failed: {e}")
            return ["Analysis completed successfully"]

    def _calculate_confidence_score(self, processed_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality."""
        try:
            total_tools = len(processed_results)
            successful_tools = sum(1 for result in processed_results.values() if result['success'])
            
            # Base confidence from success rate
            base_confidence = successful_tools / total_tools if total_tools > 0 else 0.5
            
            # Boost confidence if we have substantial content
            content_bonus = 0.0
            for result in processed_results.values():
                if result['key_data'] and result['success']:
                    key_data_str = str(result['key_data'])
                    if len(key_data_str) > 1000:  # Substantial content
                        content_bonus += 0.2
                    elif len(key_data_str) > 500:  # Moderate content  
                        content_bonus += 0.1
            
            # Final confidence score
            confidence = min(0.95, base_confidence + content_bonus)
            return max(0.1, confidence)  # Minimum 0.1, maximum 0.95
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.7  # Default moderate confidence
    
    def _extract_synthesis_manually(self, analysis_text: str, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract synthesis components manually when JSON parsing fails.
        
        Args:
            analysis_text: Raw LLM response text
            processed_results: Processed tool results for fallback
            
        Returns:
            Dictionary with synthesis components
        """
        try:
            # Use the full text as synthesis
            synthesis = analysis_text
            
            # Extract key findings from text if possible
            key_findings = []
            lines = analysis_text.split('\n')
            in_findings = False
            
            for line in lines:
                line = line.strip()
                if 'finding' in line.lower() and ('**' in line or '#' in line):
                    in_findings = True
                    continue
                if in_findings and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    finding = line[1:].strip()
                    if finding and len(finding) > 5:
                        key_findings.append(finding)
                if in_findings and line.startswith('#') and 'finding' not in line.lower():
                    break
            
            # If no findings found, create basic ones
            if not key_findings:
                key_findings = [
                    f"Analysis completed using {len(processed_results)} tools",
                    "Results combined for comprehensive insights",
                    "Manual extraction used due to parsing issues"
                ]
            
            return {
                "synthesis": synthesis,
                "key_findings": key_findings[:5],
                "confidence_score": 0.6  # Medium confidence for manual extraction
            }
            
        except Exception as e:
            logger.warning(f"Manual extraction failed: {e}")
            return self._perform_fallback_synthesis(processed_results, "Unknown query")
    
    def _perform_fallback_synthesis(self, processed_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Perform basic synthesis without LLM.
        
        Args:
            processed_results: Processed tool results
            original_query: Original user query
            
        Returns:
            Dictionary with basic synthesis components
        """
        tool_names = list(processed_results.keys())
        
        # Create basic synthesis text
        synthesis = f"Analysis Summary for: {original_query}\n\n"
        synthesis += f"Tools Used: {', '.join(tool_names)}\n\n"
        
        key_findings = []
        
        for tool_name, result in processed_results.items():
            synthesis += f"{tool_name}: {'Success' if result['success'] else 'Failed'}\n"
            synthesis += f"Summary: {result['summary'][:150]}...\n\n"
            
            # Extract findings from tool results
            if result['key_data']:
                key_data = result['key_data']
                if 'findings' in key_data:
                    key_findings.extend(key_data['findings'][:2])
                elif 'key_findings' in key_data:
                    key_findings.extend(key_data['key_findings'][:2])
        
        # Default findings if none extracted
        if not key_findings:
            successful_tools = sum(1 for result in processed_results.values() if result['success'])
            key_findings = [
                f"Processed query using {len(tool_names)} tools",
                f"{successful_tools} tools executed successfully",
                "Basic synthesis without LLM analysis",
                "Review individual tool results for detailed insights"
            ]
        
        return {
            "synthesis": synthesis,
            "key_findings": key_findings[:5],
            "confidence_score": 0.4  # Lower confidence for basic fallback
        }


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
        synthesis_tool = SynthesizeResultsTool()
        
        print(f" PASS - All analysis tools created successfully")
        print(f"   CompareDocumentsTool: {compare_tool.name}")
        print(f"   RiskAssessmentTool: {risk_tool.name}")
        print(f"   SynthesizeResultsTool: {synthesis_tool.name}")
        
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
        
        # Test SynthesizeResultsTool schema
        synthesis_args = SynthesizeResultsArgs(
            tool_results=[{"tool_name": "test", "result": "test result"}],
            original_query="test query"
        )
        print(f" PASS - SynthesizeResults schema validation passed")
        
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
    
    # Test SynthesizeResultsTool with mock data
    try:
        mock_tool_results = [
            {
                "tool_name": "compare_documents",
                "success": True,
                "comparison_analysis": "Documents show similar financial trends but different risk profiles.",
                "similarities": ["Both mention revenue growth", "Similar market focus"],
                "differences": ["Different risk tolerance", "Varied investment strategies"],
                "key_findings": ["Strong correlation in performance metrics"]
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
            original_query="Compare these documents and assess their risks"
        )
        
        if result.success:
            print(f" PASS - SynthesizeResults executed successfully")
            print(f"   Synthesis length: {len(result.synthesis)}")
            print(f"   Key findings: {len(result.key_findings)}")
            print(f"   Confidence score: {result.confidence_score}")
        else:
            print(f" WARNING - Synthesis completed but with issues: {result.error_message}")
            
    except Exception as e:
        print(f"L FAIL - SynthesizeResults execution failed: {e}")
    
    # Test 4: Schema information retrieval
    print(f"\nTest 4: Schema information retrieval")
    try:
        for tool_name, tool in [("CompareDocuments", compare_tool), 
                               ("RiskAssessment", risk_tool),
                               ("SynthesizeResults", synthesis_tool)]:
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