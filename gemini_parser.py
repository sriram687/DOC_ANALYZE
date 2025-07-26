import json
import asyncio
import logging
from typing import List, Dict, Any

import google.generativeai as genai

from config import config

logger = logging.getLogger(__name__)

class GeminiParser:
    """Handles Gemini API interactions for query parsing and logic evaluation"""
    
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract structured intent from natural language query"""
        prompt = f"""
        Analyze the following natural language query and extract structured information:
        
        Query: "{query}"
        
        Please return a JSON object with the following structure:
        {{
            "intent": "coverage_check|policy_lookup|compliance_check|general_inquiry",
            "target": "specific item/procedure/condition being asked about",
            "focus": ["list", "of", "key", "aspects", "to", "focus", "on"],
            "question_type": "yes_no|conditions|explanation|comparison",
            "entities": ["list", "of", "named", "entities", "found"]
        }}
        
        Only return the JSON object, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error parsing query intent: {str(e)}")
            # Fallback basic parsing
            return {
                "intent": "general_inquiry",
                "target": query[:50],
                "focus": ["coverage", "conditions"],
                "question_type": "explanation",
                "entities": []
            }
    
    async def evaluate_clause_relevance(self, query: str, chunks: List[str]) -> List[Dict[str, Any]]:
        """Use Gemini to assess and rank chunk relevance"""
        prompt = f"""
        Query: "{query}"
        
        Please evaluate the relevance of each text chunk to the query and extract key information:
        
        Text Chunks:
        {chr(10).join([f"CHUNK_{i}: {chunk}" for i, chunk in enumerate(chunks)])}
        
        For each chunk, return a JSON array with objects containing:
        {{
            "chunk_id": "CHUNK_X",
            "relevance_score": 0.0-1.0,
            "key_phrases": ["relevant", "phrases", "found"],
            "contains_conditions": true/false,
            "summary": "brief summary of relevant content"
        }}
        
        Only return the JSON array, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error evaluating clause relevance: {str(e)}")
            # Fallback scoring
            return [{"chunk_id": f"CHUNK_{i}", "relevance_score": 0.5, 
                    "key_phrases": [], "contains_conditions": False, 
                    "summary": "Could not analyze"} for i in range(len(chunks))]
    
    async def generate_final_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                                  query_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final structured answer based on relevant chunks"""
        chunks_text = "\n".join([f"CLAUSE_{chunk['chunk_id']}: {chunk['text']}" 
                                for chunk in relevant_chunks])
        
        prompt = f"""
        Based on the following relevant document clauses, provide a comprehensive answer to the user's query.
        
        Query: "{query}"
        Query Intent: {json.dumps(query_intent)}
        
        Relevant Clauses:
        {chunks_text}
        
        Please provide a response in the following JSON format:
        {{
            "answer": "Direct answer to the query",
            "conditions": ["list", "of", "any", "conditions", "or", "requirements"],
            "evidence": [
                {{
                    "clause_id": "unique_identifier",
                    "text": "relevant excerpt from the clause",
                    "relevance": "why this clause is relevant"
                }}
            ],
            "confidence": 0.0-1.0,
            "caveats": ["any", "limitations", "or", "uncertainties"]
        }}
        
        Be thorough but concise. If the answer cannot be determined from the provided clauses, state that clearly.
        Only return the JSON object, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return {
                "answer": "Unable to process the query due to technical issues.",
                "conditions": [],
                "evidence": [],
                "confidence": 0.0,
                "caveats": ["Technical error occurred during processing"]
            }