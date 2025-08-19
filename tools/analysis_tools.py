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
from pydantic import Field

# Import dependencies with error handling
try:
    import openai
except ImportError as e:
    logger.error(f"OpenAI library not available: {e}")
    openai = None

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
            
            # Call OpenAI for document mapping
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Document Analyst specializing in extracting key information from documents. Focus on identifying specific, relevant details that support comparative analysis."},
                    {"role": "user", "content": map_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
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
            
            # Call OpenAI for comparison reduction
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Lead Analyst specializing in comparative analysis. Synthesize extracted information to provide comprehensive, structured comparisons highlighting key similarities and differences."},
                    {"role": "user", "content": reduce_prompt}
                ],
                temperature=0.1,
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
    
    def __init__(self):
        super().__init__()
        # Initialize OpenAI client
        if openai and OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        else:
            self.client = None
            self.model = None
            logger.warning("OpenAI client not available for SynthesizeResultsTool")
    
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
            return {
                "synthesis": synthesis_result["synthesis"],
                "key_findings": synthesis_result["key_findings"],
                "confidence_score": synthesis_result["confidence_score"],
                "metadata": {
                    "tools_processed": len(processed_results),
                    "llm_powered": self.client is not None,
                    "synthesis_length": len(synthesis_result["synthesis"])
                }
            }
            
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
    
    def _perform_llm_synthesis(self, processed_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Perform LLM-powered synthesis of tool results.
        
        Args:
            processed_results: Processed tool results
            original_query: Original user query
            
        Returns:
            Dictionary with synthesis, key findings, and confidence score
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
            
            # Create synthesis prompt for expert analyst behavior
            synthesis_prompt = f"""
**ROLE:** Chief Intelligence Analyst
**MISSION:** Synthesize multi-tool analysis results to deliver definitive, executive-level response to the original query.

**ORIGINAL QUERY:** "{original_query}"

**INTELLIGENCE GATHERED:**
{context}

**SYNTHESIS PROTOCOL:**

You are conducting final intelligence synthesis for executive decision-making. Your response must be:
- **DEFINITIVE:** No uncertainty language ("maybe", "possibly", "might")
- **EVIDENCE-BASED:** Every conclusion supported by tool results
- **ACTIONABLE:** Clear next steps and recommendations
- **EXECUTIVE-READY:** Strategic context and business implications

**SYNTHESIS FRAMEWORK:**

**1. EXECUTIVE SUMMARY**
Provide a concise, definitive answer to "{original_query}" based on analyzed data. Lead with your key conclusion and strategic implications.

**2. EVIDENCE ANALYSIS**
Synthesize findings from multiple tools:
- Cross-reference consistent patterns across tool results
- Identify contradictions and resolve with strongest evidence
- Highlight unique insights that inform decision-making
- Quantify findings where possible

**3. STRATEGIC ASSESSMENT**
Strategic context and business implications:
- Risk-opportunity matrix based on findings
- Competitive positioning insights
- Resource allocation recommendations
- Timeline considerations for implementation

**4. DEFINITIVE CONCLUSIONS**
Clear, evidence-backed conclusions:
- Primary recommendation with supporting rationale
- Secondary options with trade-off analysis
- Success metrics and measurement criteria
- Critical success factors for implementation

**5. ACTION FRAMEWORK**
Concrete next steps organized by priority:
- **IMMEDIATE (0-2 weeks):** Critical actions requiring urgent attention
- **SHORT-TERM (1-3 months):** Important initiatives for systematic execution
- **STRATEGIC (3-12 months):** Long-term positioning and capability building

**OUTPUT REQUIREMENT:**
Respond with valid JSON format containing exactly these fields:
```json
{{
    "synthesis": "Comprehensive executive response to the original query with definitive conclusions and strategic context",
    "key_findings": [
        "Evidence-based finding 1 with specific supporting data",
        "Evidence-based finding 2 with quantified insights", 
        "Evidence-based finding 3 with strategic implications",
        "Evidence-based finding 4 with actionable recommendations",
        "Evidence-based finding 5 with competitive context"
    ],
    "confidence_score": 0.85
}}"""

            # Call OpenAI for synthesis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Sen deneyimli bir uzman analistsin. Görevin, çoklu araç analiz sonuçlarını sentezleyerek kesin, güvenli ve detaylı değerlendirmeler yapmak. Belirsizlik ifadeleri kullanma. Elimdeki verilere dayanarak net sonuçlar çıkar ve öneriler sun. Her zaman geçerli JSON formatında synthesis, key_findings ve confidence_score alanlarını içeren yanıt ver."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent structured output
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                synthesis_data = json.loads(analysis_text)
                
                # Validate required fields
                if all(key in synthesis_data for key in ["synthesis", "key_findings", "confidence_score"]):
                    # Ensure confidence score is within bounds
                    confidence = float(synthesis_data["confidence_score"])
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return {
                        "synthesis": str(synthesis_data["synthesis"]),
                        "key_findings": list(synthesis_data["key_findings"])[:5],  # Limit to 5
                        "confidence_score": confidence
                    }
                else:
                    logger.warning("LLM response missing required fields, using fallback")
                    raise ValueError("Missing required fields in LLM response")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                # Extract manually if JSON parsing fails
                return self._extract_synthesis_manually(analysis_text, processed_results)
            
        except Exception as e:
            logger.exception(f"LLM synthesis failed: {e}")
            return self._perform_fallback_synthesis(processed_results, original_query)
    
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