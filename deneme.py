# Add these debug logs to tools/synthesis_tools.py in _perform_llm_synthesis method

def _perform_llm_synthesis(self, processed_results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """
    Enhanced LLM synthesis with debug logging.
    """
    try:
        # DEBUG: Log inputs
        logger.info(f"[DEBUG] SynthesizeResultsTool called with:")
        logger.info(f"[DEBUG] - original_query: {original_query[:100]}...")
        logger.info(f"[DEBUG] - processed_results keys: {list(processed_results.keys())}")
        logger.info(f"[DEBUG] - processed_results: {str(processed_results)[:500]}...")
        
        # Prepare detailed context
        context = f"User Query: {original_query}\n\n"
        context += "EXTRACTED DOCUMENT CONTENT:\n\n"
        
        for tool_name, result in processed_results.items():
            logger.info(f"[DEBUG] Processing result from: {tool_name}")
            context += f"=== {tool_name.upper()} RESULTS ===\n"
            context += f"Success: {result['success']}\n"
            context += f"Summary: {result['summary']}\n"
            
            if result['key_data']:
                # Extract actual content from key_data
                key_data = result['key_data']
                logger.info(f"[DEBUG] key_data type: {type(key_data)}")
                
                if isinstance(key_data, dict):
                    if 'content' in key_data:
                        actual_content = str(key_data['content'])
                        logger.info(f"[DEBUG] Found content, length: {len(actual_content)}")
                        context += f"ACTUAL DOCUMENT CONTENT:\n{actual_content}\n"
                    elif 'analysis' in key_data:
                        context += f"Analysis: {str(key_data['analysis'])}\n"
                    else:
                        context += f"Data: {str(key_data)}\n"
                else:
                    context += f"Data: {str(key_data)}\n"
            context += "\n---\n\n"

        logger.info(f"[DEBUG] Context prepared, length: {len(context)}")
        logger.info(f"[DEBUG] Context preview: {context[:300]}...")

        # CONFIDENT synthesis prompt (your enhanced version)
        synthesis_prompt = f"""
**CONFIDENT DOCUMENT ANALYST - NO UNCERTAINTY ALLOWED**

**USER QUESTION:** "{original_query}"

**ACTUAL DOCUMENT CONTENT AVAILABLE:**
{context}

**STRICT DIRECTIVE:**
You have the COMPLETE document content above. You are NOT speculating or guessing. You have the actual document text. Provide a DEFINITIVE, SPECIFIC response about what the document contains.

**FORBIDDEN WORDS/PHRASES:**
- "may not directly answer"
- "likely", "might", "appears", "seems"
- "further analysis recommended"
- "may be necessary"
- "It is important to note"
- "refinements may be necessary"

**REQUIRED RESPONSE STYLE:**

**For Turkish Questions:**
Start with: "Bu PDF, [specific document type] içerir ve şu konuları ele alır:"

**For English Questions:**  
Start with: "This PDF contains [specific content] and covers the following:"

**CONTENT EXTRACTION REQUIREMENTS:**
1. **Document Type**: Exactly what kind of document this is
2. **Main Subject**: The specific project/topic being discussed
3. **Key Details**: Specific technologies, objectives, features mentioned
4. **Scope**: What the project includes and excludes
5. **Technical Stack**: Exact technologies mentioned

**RESPONSE TEMPLATE FOR TURKISH:**
"Bu PDF, [tam belge türü] belgesidir ve [spesifik proje adı] hakkındadır.

Belgenin içeriği:
• [Spesifik detay 1]
• [Spesifik detay 2] 
• [Spesifik detay 3]

Proje kapsamı: [exact scope from document]
Teknoloji yığını: [exact technologies mentioned]
Hedef: [exact objective from document]"

**CONFIDENCE COMMAND:**
You MUST extract specific information from the document content provided. Make definitive statements about what the document contains. You have access to the full document text - use it to provide comprehensive, specific answers.

**DEFINITIVE DOCUMENT ANALYSIS:**
"""

        logger.info(f"[DEBUG] Calling OpenAI with prompt length: {len(synthesis_prompt)}")

        # Call OpenAI with strict confident directive
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a confident document analysis expert with DIRECT ACCESS to complete document content.

ABSOLUTE REQUIREMENTS:
- Make DEFINITIVE statements: "This document contains", "The project includes", "The objective is"
- NEVER use uncertain language: avoid "may", "might", "likely", "appears", "seems", "could be"
- Extract SPECIFIC details: names, technologies, objectives, timelines, requirements
- Use ONLY the provided document content - no speculation
- Respond in the SAME language as the user's question
- Provide comprehensive, detailed analysis of actual document contents

You have successfully extracted the complete document content. Analyze it definitively."""
                },
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.0,  # Maximum determinism
            max_tokens=2000
        )
        
        analysis_text = response.choices[0].message.content.strip()
        logger.info(f"[DEBUG] OpenAI response length: {len(analysis_text)}")
        logger.info(f"[DEBUG] OpenAI response preview: {analysis_text[:200]}...")
        
        # Extract key findings
        key_findings = self._extract_key_findings_from_analysis(analysis_text)
        logger.info(f"[DEBUG] Extracted {len(key_findings)} key findings")
        
        # Calculate confidence
        confidence_score = 0.95  # High confidence since we have full content
        
        result = {
            "synthesis": analysis_text,
            "key_findings": key_findings,
            "confidence_score": confidence_score
        }
        
        logger.info(f"[DEBUG] SynthesizeResultsTool completed successfully")
        return result
        
    except Exception as e:
        logger.exception(f"[DEBUG] Enhanced LLM synthesis failed: {e}")
        return self._perform_fallback_synthesis(processed_results, original_query)