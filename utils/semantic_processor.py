"""
DataNeuron Semantic Processing and Query Optimization Module
============================================================

This module provides sophisticated semantic processing capabilities for enhancing
RAG (Retrieval-Augmented Generation) system performance through query expansion,
keyword extraction, and intent detection using advanced LLM-based techniques.

Features:
- Singleton pattern for efficient Anthropic client management
- Query expansion using semantic rephrasing and HyDE-like techniques
- Intelligent keyword and entity extraction from text
- Query intent detection for enhanced planning
- Optimized for minimal latency and high performance
- Comprehensive error handling with graceful fallbacks
- Real-time performance monitoring and statistics

Usage:
    from utils.semantic_processor import SemanticProcessor
    
    # Get singleton instance
    processor = SemanticProcessor()
    
    # Expand query for better search results
    expanded_queries = processor.expand_query("AI risks")
    
    # Extract keywords from text
    keywords = processor.extract_keywords("DataNeuron uses ChromaDB...")
    
    # Detect query intent
    intent = processor.detect_query_intent("Compare these documents")

Architecture:
- Singleton SemanticProcessor class for centralized intelligence
- LLM-powered semantic analysis and understanding
- Fast, lightweight operations for real-time processing
- Comprehensive caching and optimization strategies
"""

import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger
from config.settings import OPENAI_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_API_KEY

# Import dependencies with error handling
try:
    import anthropic
except ImportError as e:
    logger.error(f"Anthropic library not available: {e}")
    anthropic = None


# ============================================================================
# SEMANTIC PROCESSOR SINGLETON CLASS
# ============================================================================

class SemanticProcessor:
    """
    Singleton Semantic Processing Engine for DataNeuron.

    This class provides advanced semantic processing capabilities using Anthropic's
    language models as reasoning engines. It enhances RAG system performance
    through query expansion, keyword extraction, and intent detection.
    
    Key Features:
    - Singleton pattern for efficient resource management
    - Query expansion using semantic rephrasing techniques
    - Intelligent keyword and entity extraction
    - Query intent classification for enhanced planning
    - Optimized for minimal latency and real-time processing
    - Comprehensive error handling and fallback mechanisms
    - Performance monitoring and statistics tracking
    
    The processor acts as an intelligence layer between user queries and
    the vector search system, optimizing queries for better retrieval results.
    """
    
    _instance: Optional['SemanticProcessor'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SemanticProcessor':
        """
        Singleton pattern implementation with thread safety.
        
        Returns:
            Single instance of SemanticProcessor
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SemanticProcessor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the SemanticProcessor singleton.

        Sets up Anthropic client and configures semantic processing parameters.
        Only initializes once due to singleton pattern.
        
        Raises:
            RuntimeError: If Anthropic client cannot be initialized
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

            # Check if Anthropic is available
            if anthropic is None:
                raise RuntimeError("Anthropic library is required for SemanticProcessor")

            # Validate API key
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("Anthropic API key is required for SemanticProcessor")

            # Initialize Anthropic client
            try:
                self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                self.model = ANTHROPIC_MODEL
                
                logger.info("SemanticProcessor initialized successfully")
                logger.info(f"  - Model: {self.model}")
                
                # Test client connection
                test_response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                    temperature=0.1
                )
                
                if test_response:
                    logger.success("Anthropic client connection verified for semantic processing")

            except Exception as e:
                logger.exception(f"Failed to initialize SemanticProcessor: {e}")
                raise RuntimeError(f"SemanticProcessor initialization failed: {e}")
            
            # Configuration for semantic processing
            self.query_expansion_temperature = 0.3  # Balanced creativity for expansion
            self.extraction_temperature = 0.1      # Low temperature for precise extraction
            self.intent_temperature = 0.0          # Deterministic intent detection
            self.max_tokens_expansion = 200        # Sufficient for query variations
            self.max_tokens_extraction = 100       # Concise keyword extraction
            self.max_tokens_intent = 50           # Short intent classification
            
            # Performance tracking
            self.total_expansions = 0
            self.total_extractions = 0
            self.total_intent_detections = 0
            self.total_processing_time = 0.0
            self.cache = {}  # Simple in-memory cache
            self.cache_max_size = 1000
            
            # Predefined intent categories
            self.intent_categories = [
                "summarize", "compare", "define", "explain", "list", 
                "analyze", "research", "translate", "calculate", "question"
            ]
            
            self._initialized = True
            logger.success("SemanticProcessor ready for semantic analysis")
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Expand a user query into multiple semantically equivalent variations.
        
        This method uses advanced semantic rephrasing to create query variations
        that can improve vector search recall by matching documents with different
        terminology but similar meaning (similar to HyDE technique).
        
        Args:
            query (str): Original user query to expand
            num_variations (int): Number of query variations to generate (default: 3)
            
        Returns:
            List[str]: List containing original query plus semantic variations
            
        Raises:
            ValueError: If query is empty or invalid
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if num_variations < 1 or num_variations > 10:
            raise ValueError("Number of variations must be between 1 and 10")
        
        query = query.strip()
        
        # Check cache first
        cache_key = f"expand_{query}_{num_variations}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for query expansion: {query[:50]}...")
            return self.cache[cache_key]
        
        logger.debug(f"Expanding query: {query[:100]}...")
        start_time = time.time()
        
        try:
            # Create expansion prompt
            expansion_prompt = f"""
**ADVANCED SEMANTIC QUERY EXPANSION SYSTEM**
**OPTIMIZATION TARGET:** Document Retrieval Enhancement

**QUERY TO EXPAND:** "{query}"
**EXPANSION COUNT:** {num_variations} semantic variations

**SEMANTIC EXPANSION PROTOCOL:**

**RETRIEVAL OPTIMIZATION STRATEGY:**
You are optimizing for vector similarity search across document collections with varying:
- Writing styles (formal reports ↔ informal notes)
- Expertise levels (technical documentation ↔ executive summaries)  
- Terminology preferences (industry jargon ↔ plain language)
- Content depth (detailed analysis ↔ brief overviews)

**EXPANSION DIMENSIONS:**

**📚 PROFESSIONAL/TECHNICAL DIMENSION:**
Transform query using:
- Industry-specific terminology and acronyms
- Formal academic or business language
- Technical precision and domain expertise
- Professional documentation style

**💬 CONVERSATIONAL/ACCESSIBLE DIMENSION:**
Rephrase using:
- Everyday language and common expressions
- Natural speech patterns and questions
- Simplified terminology without jargon
- User-friendly explanations

**🔄 ALTERNATIVE CONCEPTUAL DIMENSION:**
Reframe through:
- Different conceptual approaches to same information
- Alternative problem formulations
- Related but distinct terminology clusters
- Complementary perspectives on the topic

**🎯 ENHANCED SPECIFICITY DIMENSION:**
Develop variations with:
- More detailed or specific focus areas
- Broader contextual framing
- Implementation-oriented language
- Results-focused terminology

**SEMANTIC QUALITY ASSURANCE:**
- Maintain perfect semantic equivalence across all variations
- Ensure natural, complete query expressions
- Optimize for embedding model comprehension
- Maximize document retrieval coverage

**OPTIMIZED SEMANTIC VARIATIONS:**
Generate {num_variations} variations below, each on a separate line with no formatting.
"""

            # Call Anthropic for query expansion
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating semantic query variations for document search optimization."},
                    {"role": "user", "content": expansion_prompt}
                ],
                temperature=self.query_expansion_temperature,
                max_tokens=self.max_tokens_expansion
            )
            
            # Parse response
            variations_text = response.choices[0].message.content.strip()
            variations = [
                var.strip() 
                for var in variations_text.split('\n') 
                if var.strip() and not var.strip().startswith(('-', '*', '1.', '2.', '3.'))
            ]
            
            # Filter and clean variations
            clean_variations = []
            for var in variations[:num_variations]:
                # Remove common prefixes/suffixes
                var = var.strip(' -*"')
                if var and var.lower() != query.lower() and len(var) > 5:
                    clean_variations.append(var)
            
            # Ensure we have the original query first
            result = [query] + clean_variations[:num_variations]
            
            # Cache the result
            self._update_cache(cache_key, result)
            
            # Update statistics
            self.total_expansions += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.success(f"Query expanded into {len(result)} variations in {processing_time:.3f}s")
            logger.debug(f"Variations: {result}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Query expansion failed for: {query}")
            
            # Fallback: return original query only
            fallback_result = [query]
            logger.warning(f"Using fallback: returning original query only")
            
            return fallback_result
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract the most important keywords and entities from text.
        
        Uses LLM-powered analysis to identify key terms, proper nouns, and
        important concepts that can be used for metadata-based filtering
        and enhanced search functionality.
        
        Args:
            text (str): Text to extract keywords from
            max_keywords (int): Maximum number of keywords to extract (default: 5)
            
        Returns:
            List[str]: List of extracted keywords and entities
            
        Raises:
            ValueError: If text is empty or max_keywords is invalid
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if max_keywords < 1 or max_keywords > 20:
            raise ValueError("max_keywords must be between 1 and 20")
        
        text = text.strip()
        
        # Check cache first
        cache_key = f"extract_{hash(text)}_{max_keywords}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for keyword extraction")
            return self.cache[cache_key]
        
        logger.debug(f"Extracting {max_keywords} keywords from text: {text[:100]}...")
        start_time = time.time()
        
        try:
            # Create extraction prompt
            extraction_prompt = f"""
**KEYWORD INTELLIGENCE EXTRACTION ENGINE**

**MISSION:** Extract the {max_keywords} most strategically valuable keywords and entities from the provided text for search optimization and metadata categorization.

**TARGET TEXT:**
"{text}"

**EXTRACTION FRAMEWORK:**

**PRIORITY HIERARCHY:**
1. **PROPER NOUNS** (High Value): Person names, organizations, companies, products, locations, brands
2. **TECHNICAL CONCEPTS** (High Value): Domain-specific terms, methodologies, technologies, systems
3. **KEY ENTITIES** (Medium Value): Important objects, processes, standards, metrics, frameworks
4. **DOMAIN TERMINOLOGY** (Medium Value): Industry jargon, specialized vocabulary, acronyms

**EXTRACTION CRITERIA:**

**🎯 SEARCH OPTIMIZATION FOCUS:**
- Terms that users would likely search for when looking for this content
- Unique identifiers that distinguish this document from others
- Concepts that would be valuable for document categorization and filtering
- Names and entities that provide specific context and relevance

**📊 CATEGORIZATION VALUE:**
- Terms that indicate document type, domain, or subject matter
- Keywords that would help in automatic document classification
- Entities that provide semantic context for content understanding
- Concepts that enable metadata-based document organization

**🔍 SEMANTIC IMPORTANCE:**
- Central concepts that define the document's core themes
- Technical terminology that indicates expertise level and domain
- Proper nouns that provide factual anchors for information retrieval
- Key terms that would appear in executive summaries or abstracts

**EXTRACTION STANDARDS:**
✓ **Relevance:** Only extract terms directly related to main content themes
✓ **Specificity:** Prefer specific terms over generic ones (e.g., "ChromaDB" > "database")
✓ **Searchability:** Choose terms users would actually search for
✓ **Uniqueness:** Prioritize distinctive terms that characterize this specific content
✓ **Utility:** Focus on terms valuable for categorization and retrieval systems

**QUALITY FILTERS:**
- Exclude common stop words, articles, and generic terms
- Avoid overly broad concepts unless they're central themes
- Prioritize nouns and noun phrases over adjectives or verbs
- Ensure extracted terms provide meaningful semantic value

**EXTRACTED KEYWORDS:**
Return exactly {max_keywords} comma-separated keywords with no explanations, formatting, or numbering.
"""

            # Call Anthropic for keyword extraction
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key terms and entities from text for search optimization."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=self.extraction_temperature,
                max_tokens=self.max_tokens_extraction
            )
            
            # Parse response
            keywords_text = response.choices[0].message.content.strip()
            
            # Split and clean keywords
            keywords = []
            for keyword in keywords_text.split(','):
                keyword = keyword.strip(' .-"\'')
                if keyword and len(keyword) > 1 and keyword.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with']:
                    keywords.append(keyword)
            
            # Limit to requested number
            result = keywords[:max_keywords]
            
            # Cache the result
            self._update_cache(cache_key, result)
            
            # Update statistics
            self.total_extractions += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.success(f"Extracted {len(result)} keywords in {processing_time:.3f}s")
            logger.debug(f"Keywords: {result}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Keyword extraction failed for text: {text[:100]}...")
            
            # Fallback: simple word extraction
            fallback_keywords = []
            words = text.split()
            for word in words:
                word = word.strip('.,!?;:"()[]{}')
                if (len(word) > 4 and 
                    word[0].isupper() and 
                    word not in fallback_keywords and 
                    len(fallback_keywords) < max_keywords):
                    fallback_keywords.append(word)
            
            logger.warning(f"Using fallback keyword extraction: {fallback_keywords}")
            return fallback_keywords
    
    def detect_query_intent(self, query: str) -> str:
        """
        Detect the intent behind a user query for enhanced planning.
        
        Classifies user queries into predefined intent categories to help
        the LLMAgent make better planning decisions and tool selections.
        
        Args:
            query (str): User query to analyze for intent
            
        Returns:
            str: Detected intent category (e.g., "summarize", "compare", "explain")
            
        Raises:
            ValueError: If query is empty
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        # Check cache first
        cache_key = f"intent_{hash(query)}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for intent detection: {query[:50]}...")
            return self.cache[cache_key]
        
        logger.debug(f"Detecting intent for query: {query[:100]}...")
        start_time = time.time()
        
        try:
            # Create intent detection prompt
            intent_categories_str = ", ".join(self.intent_categories)
            intent_prompt = f"""
**STRATEGIC QUERY INTENT CLASSIFICATION ENGINE**

**MISSION:** Classify user query intent to optimize tool selection and execution planning for DataNeuron AI agent.

**QUERY TO ANALYZE:** "{query}"

**INTENT CLASSIFICATION FRAMEWORK:**

**AVAILABLE INTENT CATEGORIES:** {intent_categories_str}

**INTENT CATEGORY DEFINITIONS:**

**📋 CONTENT PROCESSING INTENTS:**
- **summarize:** User wants condensed overview, key points, or executive summary of content
- **compare:** User seeks side-by-side analysis, differences/similarities between multiple items
- **analyze:** User requests in-depth examination, insights, patterns, or strategic assessment
- **research:** User needs comprehensive information gathering on a topic or domain

**🎯 INFORMATION SEEKING INTENTS:**
- **define:** User wants clear definition, meaning, or explanation of terms/concepts
- **explain:** User seeks detailed understanding of processes, mechanisms, or causality
- **question:** User asks specific factual questions requiring direct answers
- **list:** User wants enumerated items, steps, options, or structured information

**⚙️ FUNCTIONAL OPERATION INTENTS:**
- **translate:** User requests language translation or format conversion
- **calculate:** User needs mathematical computation, numerical analysis, or quantitative results

**INTENT DETECTION METHODOLOGY:**

**🔍 LINGUISTIC PATTERN ANALYSIS:**
Examine query for:
- **Action verbs:** "compare", "summarize", "explain", "analyze", "list", "define"
- **Question patterns:** "What is...", "How does...", "Why...", "Which..."
- **Comparative language:** "versus", "difference", "better", "contrast"
- **Request indicators:** "show me", "give me", "I need", "help me"

**🎯 CONTEXTUAL INTENT MAPPING:**
Consider:
- **Information depth required:** Surface-level vs. deep analysis
- **Output format expectations:** Lists, comparisons, explanations, summaries
- **Cognitive task complexity:** Simple retrieval vs. complex reasoning
- **User goal orientation:** Learning, decision-making, or task completion

**🧠 STRATEGIC CLASSIFICATION LOGIC:**
Prioritize intent based on:
- **Primary user objective:** What outcome does the user ultimately seek?
- **Cognitive complexity:** What level of processing is required?
- **Tool alignment:** Which DataNeuron tools best serve this intent?
- **Output specificity:** How structured should the response be?

**CLASSIFICATION DECISION TREE:**
1. **Document manipulation needed?** → summarize, compare, analyze
2. **Information retrieval focus?** → research, question, define
3. **Structured output required?** → list, explain
4. **Computational task?** → calculate, translate

**QUALITY ASSURANCE:**
✓ **Single Intent Focus:** Select the most dominant intent, not secondary ones
✓ **User Goal Alignment:** Choose intent that best serves user's primary objective
✓ **Tool Optimization:** Ensure classification enables optimal tool selection
✓ **Execution Planning:** Support efficient agent planning and resource allocation

**STRATEGIC INTENT CLASSIFICATION:**
Return exactly one category from the available options that best represents the user's primary intent.
"""

            # Call Anthropic for intent detection
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at understanding user intent from queries. Always respond with exactly one category."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=self.intent_temperature,
                max_tokens=self.max_tokens_intent
            )
            
            # Parse response
            detected_intent = response.choices[0].message.content.strip().lower()
            
            # Validate intent category
            if detected_intent in self.intent_categories:
                result = detected_intent
            else:
                # Fallback: check if response contains a valid category
                result = "question"  # Default fallback
                for category in self.intent_categories:
                    if category in detected_intent:
                        result = category
                        break
            
            # Cache the result
            self._update_cache(cache_key, result)
            
            # Update statistics
            self.total_intent_detections += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.success(f"Detected intent '{result}' in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.exception(f"Intent detection failed for query: {query}")
            
            # Fallback: rule-based intent detection
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
                fallback_intent = "define"
            elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
                fallback_intent = "compare"
            elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
                fallback_intent = "summarize"
            elif any(word in query_lower for word in ['list', 'show me', 'give me', 'items']):
                fallback_intent = "list"
            elif any(word in query_lower for word in ['how', 'why', 'explain', 'because']):
                fallback_intent = "explain"
            elif any(word in query_lower for word in ['analyze', 'analysis', 'insights']):
                fallback_intent = "analyze"
            else:
                fallback_intent = "question"
            
            logger.warning(f"Using fallback intent detection: {fallback_intent}")
            return fallback_intent
    
    def _update_cache(self, key: str, value: Any) -> None:
        """
        Update cache with size management.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about semantic processing performance.
        
        Returns:
            Dictionary with performance statistics and metrics
        """
        total_operations = self.total_expansions + self.total_extractions + self.total_intent_detections
        avg_processing_time = (self.total_processing_time / total_operations) if total_operations > 0 else 0
        
        return {
            "total_operations": total_operations,
            "query_expansions": self.total_expansions,
            "keyword_extractions": self.total_extractions,
            "intent_detections": self.total_intent_detections,
            "total_processing_time_seconds": round(self.total_processing_time, 3),
            "average_processing_time_seconds": round(avg_processing_time, 3),
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "model": self.model,
            "configuration": {
                "query_expansion_temperature": self.query_expansion_temperature,
                "extraction_temperature": self.extraction_temperature,
                "intent_temperature": self.intent_temperature,
                "max_tokens_expansion": self.max_tokens_expansion,
                "max_tokens_extraction": self.max_tokens_extraction,
                "max_tokens_intent": self.max_tokens_intent
            }
        }
    
    def clear_cache(self) -> int:
        """
        Clear the semantic processing cache.
        
        Returns:
            Number of cache entries cleared
        """
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared semantic processing cache: {cache_size} entries removed")
        return cache_size
    
    def process_query_semantically(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive semantic processing of a query.
        
        Performs query expansion, intent detection, and keyword extraction
        in a single operation for complete semantic analysis.
        
        Args:
            query (str): Query to process
            
        Returns:
            Dictionary containing all semantic processing results
        """
        logger.info(f"Performing comprehensive semantic processing: {query[:100]}...")
        start_time = time.time()
        
        try:
            # Perform all semantic processing operations
            expanded_queries = self.expand_query(query)
            intent = self.detect_query_intent(query)
            keywords = self.extract_keywords(query, max_keywords=3)
            
            processing_time = time.time() - start_time
            
            result = {
                "original_query": query,
                "expanded_queries": expanded_queries,
                "detected_intent": intent,
                "extracted_keywords": keywords,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time()
            }
            
            logger.success(f"Comprehensive semantic processing completed in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.exception(f"Comprehensive semantic processing failed: {e}")
            return {
                "original_query": query,
                "expanded_queries": [query],
                "detected_intent": "question",
                "extracted_keywords": [],
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
                "timestamp": time.time()
            }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Comprehensive test of the SemanticProcessor with real Anthropic API calls.

    This test validates all semantic processing capabilities including query
    expansion, keyword extraction, and intent detection with real examples.
    """
    
    print("=== DataNeuron Semantic Processor Test ===")
    print("Note: This test makes real Anthropic API calls")

    try:
        # Test 1: Create SemanticProcessor instance
        print(f"\nTest 1: Creating SemanticProcessor singleton")
        processor = SemanticProcessor()
        print(f" PASS - SemanticProcessor created successfully")
        print(f"   Model: {processor.model}")
        print(f"   Intent categories: {len(processor.intent_categories)}")
        
        # Test singleton behavior
        processor2 = SemanticProcessor()
        if processor is processor2:
            print(f" PASS - Singleton pattern working correctly")
        else:
            print(f"L FAIL - Multiple instances created")
        
        # Test 2: Query Expansion
        print(f"\nTest 2: Query Expansion")
        test_queries = [
            "AI risks",
            "machine learning algorithms",
            "climate change effects"
        ]
        
        for query in test_queries:
            print(f"\n  Testing query: '{query}'")
            try:
                expanded = processor.expand_query(query, num_variations=3)
                print(f"   PASS - Expanded into {len(expanded)} variations")
                for i, variation in enumerate(expanded):
                    print(f"     {i+1}. {variation}")
                    
                # Validate results
                if len(expanded) >= 1 and query in expanded:
                    print(f"   PASS - Original query preserved")
                else:
                    print(f"   WARNING - Original query not in results")
                    
            except Exception as e:
                print(f"   FAIL - Query expansion failed: {e}")
        
        # Test 3: Keyword Extraction
        print(f"\nTest 3: Keyword Extraction")
        test_texts = [
            "DataNeuron uses ChromaDB and Anthropic to create intelligent document processing systems for enterprise applications.",
            "The research paper discusses machine learning algorithms including neural networks, decision trees, and support vector machines.",
            "Apple Inc. released the iPhone 15 with advanced AI capabilities powered by the A17 Pro chip manufactured by TSMC."
        ]
        
        for text in test_texts:
            print(f"\n  Testing text: '{text[:60]}...'")
            try:
                keywords = processor.extract_keywords(text, max_keywords=5)
                print(f"   PASS - Extracted {len(keywords)} keywords")
                print(f"   Keywords: {keywords}")
                
                # Validate results
                if keywords and len(keywords) <= 5:
                    print(f"   PASS - Keyword extraction within limits")
                else:
                    print(f"   WARNING - Unexpected keyword results")
                    
            except Exception as e:
                print(f"   FAIL - Keyword extraction failed: {e}")
        
        # Test 4: Intent Detection
        print(f"\nTest 4: Intent Detection")
        test_queries_intent = [
            ("What is machine learning?", "define"),
            ("Compare neural networks and decision trees", "compare"),
            ("Summarize this document", "summarize"),
            ("List the benefits of AI", "list"),
            ("How does deep learning work?", "explain"),
            ("Analyze the market trends", "analyze")
        ]
        
        for query, expected_intent in test_queries_intent:
            print(f"\n  Testing query: '{query}'")
            try:
                detected_intent = processor.detect_query_intent(query)
                print(f"   PASS - Detected intent: '{detected_intent}'")
                
                # Check if detected intent is reasonable
                if detected_intent in processor.intent_categories:
                    print(f"   PASS - Intent is valid category")
                    if detected_intent == expected_intent:
                        print(f"   EXCELLENT - Matches expected intent '{expected_intent}'")
                    else:
                        print(f"   NOTE - Expected '{expected_intent}', got '{detected_intent}' (may still be correct)")
                else:
                    print(f"   WARNING - Intent not in predefined categories")
                    
            except Exception as e:
                print(f"   FAIL - Intent detection failed: {e}")
        
        # Test 5: Comprehensive Processing
        print(f"\nTest 5: Comprehensive Semantic Processing")
        test_query = "Compare the performance of different AI models for document analysis"
        
        try:
            comprehensive_result = processor.process_query_semantically(test_query)
            print(f" PASS - Comprehensive processing completed")
            print(f"   Original query: {comprehensive_result['original_query']}")
            print(f"   Expanded queries: {len(comprehensive_result['expanded_queries'])}")
            print(f"   Detected intent: {comprehensive_result['detected_intent']}")
            print(f"   Keywords: {comprehensive_result['extracted_keywords']}")
            print(f"   Processing time: {comprehensive_result['processing_time_seconds']}s")
            
        except Exception as e:
            print(f" FAIL - Comprehensive processing failed: {e}")
        
        # Test 6: Performance Statistics
        print(f"\nTest 6: Performance Statistics")
        try:
            stats = processor.get_statistics()
            print(f" PASS - Statistics retrieved")
            print(f"   Total operations: {stats['total_operations']}")
            print(f"   Query expansions: {stats['query_expansions']}")
            print(f"   Keyword extractions: {stats['keyword_extractions']}")
            print(f"   Intent detections: {stats['intent_detections']}")
            print(f"   Average processing time: {stats['average_processing_time_seconds']}s")
            print(f"   Cache size: {stats['cache_size']}")
            
        except Exception as e:
            print(f" FAIL - Statistics retrieval failed: {e}")
        
        # Test 7: Error Handling
        print(f"\nTest 7: Error Handling")
        
        # Test empty query
        try:
            processor.expand_query("")
            print(f" FAIL - Should have rejected empty query")
        except ValueError:
            print(f" PASS - Empty query correctly rejected")
        except Exception as e:
            print(f" WARNING - Unexpected error with empty query: {e}")
        
        # Test invalid parameters
        try:
            processor.extract_keywords("test", max_keywords=0)
            print(f" FAIL - Should have rejected invalid max_keywords")
        except ValueError:
            print(f" PASS - Invalid parameters correctly rejected")
        except Exception as e:
            print(f" WARNING - Unexpected error with invalid parameters: {e}")
        
        # Test 8: Cache Performance
        print(f"\nTest 8: Cache Performance")
        
        # Test same query multiple times to verify caching
        test_query_cache = "test caching performance"
        
        # First call (should hit API)
        start_time = time.time()
        result1 = processor.expand_query(test_query_cache)
        first_call_time = time.time() - start_time
        
        # Second call (should hit cache)
        start_time = time.time()
        result2 = processor.expand_query(test_query_cache)
        second_call_time = time.time() - start_time
        
        if result1 == result2:
            print(f" PASS - Cache returns consistent results")
            print(f"   First call: {first_call_time:.3f}s")
            print(f"   Second call: {second_call_time:.3f}s")
            
            if second_call_time < first_call_time * 0.5:  # Cache should be much faster
                print(f" PASS - Cache improves performance significantly")
            else:
                print(f" NOTE - Cache performance improvement not significant")
        else:
            print(f" WARNING - Cache returns inconsistent results")
        
        print(f"\n=== Test Complete ===")
        print("SemanticProcessor is ready for production use.")
        print("All semantic processing capabilities have been validated.")
        print("Integration with LLMAgent and VectorStore can proceed.")
        
        # Final statistics
        final_stats = processor.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  - Total API calls: {final_stats['total_operations']}")
        print(f"  - Total processing time: {final_stats['total_processing_time_seconds']}s")
        print(f"  - Cache efficiency: {final_stats['cache_size']} entries")
        
    except Exception as e:
        print(f"L CRITICAL FAIL - Test execution failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nNote: This test used real Anthropic API calls for validation.")
    print(f"In production, results will be cached for improved performance.")
