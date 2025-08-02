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
        self.model = genai.GenerativeModel(config.GEMINI_CHAT_MODEL)
    
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=500
                    )
                )

                # Check if response exists and has content
                if not response:
                    logger.warning(f"No response from Gemini (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    raise ValueError("No response from Gemini after retries")

                # Clean and parse the response
                if hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                    logger.debug(f"Raw Gemini response: {response_text}")

                    # Check if response is empty
                    if not response_text:
                        logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        raise ValueError("Empty response from Gemini after retries")

                    # Try to extract JSON from the response
                    if response_text.startswith('```json'):
                        # Remove markdown code blocks
                        response_text = response_text.replace('```json', '').replace('```', '').strip()
                    elif response_text.startswith('```'):
                        # Remove any code blocks
                        response_text = response_text.replace('```', '').strip()

                    # Try to find JSON object in the response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_text = response_text[start_idx:end_idx]
                        return json.loads(json_text)
                    else:
                        # If no JSON found, try parsing the whole response
                        return json.loads(response_text)
                else:
                    logger.warning(f"No text in Gemini response (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    raise ValueError("No text in Gemini response after retries")

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error (attempt {attempt + 1}): {str(e)}")
                logger.error(f"Response text: {response_text if 'response_text' in locals() else 'No response text'}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error parsing query intent (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        # Fallback basic parsing
        logger.info("Using fallback query intent parsing")
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
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=2000
                    )
                )

                if response and hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                    if response_text:
                        # Try to extract JSON from the response
                        if response_text.startswith('```json'):
                            response_text = response_text.replace('```json', '').replace('```', '').strip()
                        elif response_text.startswith('```'):
                            response_text = response_text.replace('```', '').strip()

                        # Try to find JSON array in the response
                        start_idx = response_text.find('[')
                        end_idx = response_text.rfind(']') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_text = response_text[start_idx:end_idx]
                            return json.loads(json_text)
                        else:
                            return json.loads(response_text)

                logger.warning(f"Empty or invalid response from Gemini (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in clause relevance (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error evaluating clause relevance (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        # Fallback scoring after all retries failed
        logger.warning("Using fallback clause relevance scoring")
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
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1500
                    )
                )

                if response and hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                    if response_text:
                        # Try to extract JSON from the response
                        if response_text.startswith('```json'):
                            response_text = response_text.replace('```json', '').replace('```', '').strip()
                        elif response_text.startswith('```'):
                            response_text = response_text.replace('```', '').strip()

                        # Try to find JSON object in the response
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_text = response_text[start_idx:end_idx]
                            return json.loads(json_text)
                        else:
                            return json.loads(response_text)

                logger.warning(f"Empty or invalid response from Gemini for final answer (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in final answer (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error generating final answer (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        # Fallback response after all retries failed
        logger.warning("Using fallback final answer")
        return {
            "answer": "Unable to process the query due to technical issues. Please try again.",
            "conditions": [],
            "evidence": [],
            "confidence": 0.0,
            "caveats": ["Technical error occurred during processing"]
        }