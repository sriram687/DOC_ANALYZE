#!/usr/bin/env python3
"""
Generic document processing for structured decision making across all domains
Supports insurance, legal, HR, contracts, compliance, and other document types
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.generativeai as genai
from config import config

logger = logging.getLogger(__name__)

@dataclass
class StructuredQuery:
    """Generic structured query for any document type"""
    # Personal/Entity Information
    age: Optional[int] = None
    gender: Optional[str] = None
    name: Optional[str] = None
    employee_id: Optional[str] = None

    # Action/Procedure/Request
    action: Optional[str] = None  # procedure, request, claim, application
    action_type: Optional[str] = None  # surgery, leave, loan, etc.

    # Location/Department
    location: Optional[str] = None
    department: Optional[str] = None

    # Time/Duration
    duration: Optional[str] = None  # policy duration, employment period, etc.
    date: Optional[str] = None

    # Financial
    amount: Optional[float] = None
    currency: Optional[str] = "INR"

    # Document Context
    document_type: Optional[str] = None  # insurance, contract, policy, etc.

    # Additional extracted entities
    entities: Dict[str, Any] = None

    # Original query
    raw_query: str = ""

@dataclass
class DocumentDecision:
    """Generic decision response for any document type"""
    decision: str  # "APPROVED", "REJECTED", "PENDING", "COMPLIANT", "NON_COMPLIANT", etc.
    amount: Optional[float] = None
    justification: str = ""
    clauses_used: List[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    decision_type: str = "GENERAL"  # INSURANCE, LEGAL, HR, CONTRACT, etc.

class DocumentQueryProcessor:
    """Process queries and make decisions based on any type of document"""
    
    def __init__(self):
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_CHAT_MODEL)
        else:
            self.model = None
            logger.warning("Gemini API key not available - using fallback processing")
    
    def parse_document_query(self, query: str, document_type: str = "general") -> StructuredQuery:
        """Parse natural language query into structured format for any document type"""
        
        # Enhanced regex patterns for multiple document types
        age_pattern = r'(\d{1,3})\s*(?:year|yr|y)?(?:\s*old)?(?:\s*male|\s*female|\s*M|\s*F)?'
        gender_pattern = r'(?:(\d{1,3})\s*(?:year|yr|y)?\s*(?:old)?\s*)?(male|female|M|F)\b'
        amount_pattern = r'(?:₹|Rs\.?|INR|\$|USD|EUR)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        employee_id_pattern = r'(?:emp|employee|staff)\s*(?:id|number|no)?\s*:?\s*([A-Z0-9]+)'
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'

        parsed = StructuredQuery(raw_query=query, document_type=document_type)
        query_lower = query.lower()
        
        # Extract age
        age_match = re.search(age_pattern, query, re.IGNORECASE)
        if age_match:
            try:
                parsed.age = int(age_match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Extract gender
        gender_match = re.search(gender_pattern, query, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(2).upper()
            if gender in ['M', 'MALE']:
                parsed.gender = 'Male'
            elif gender in ['F', 'FEMALE']:
                parsed.gender = 'Female'
        
        # Extract actions/procedures based on document type
        actions = {
            'insurance': [
                'knee surgery', 'hip replacement', 'heart surgery', 'cataract',
                'appendectomy', 'gallbladder', 'hernia', 'bypass', 'angioplasty',
                'chemotherapy', 'dialysis', 'physiotherapy', 'consultation'
            ],
            'hr': [
                'leave application', 'sick leave', 'maternity leave', 'vacation',
                'resignation', 'promotion', 'transfer', 'salary review',
                'performance review', 'training request', 'overtime'
            ],
            'legal': [
                'contract review', 'compliance check', 'audit', 'violation',
                'breach', 'termination', 'renewal', 'amendment', 'dispute'
            ],
            'general': [
                'application', 'request', 'claim', 'approval', 'review',
                'assessment', 'evaluation', 'processing', 'verification'
            ]
        }

        # Get relevant actions for document type
        relevant_actions = actions.get(document_type, actions['general'])
        relevant_actions.extend(actions['general'])  # Always include general terms

        for action in relevant_actions:
            if action in query_lower:
                parsed.action = action.title()
                parsed.action_type = action.split()[-1].title()  # Last word as type
                break
        
        # Extract locations (cities, countries, offices)
        locations = [
            # Indian cities
            'mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata',
            'hyderabad', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
            # International
            'new york', 'london', 'singapore', 'dubai', 'tokyo', 'sydney',
            # Office locations
            'head office', 'branch office', 'regional office', 'headquarters'
        ]

        for location in locations:
            if location in query_lower:
                parsed.location = location.title()
                break

        # Extract departments
        departments = [
            'hr', 'human resources', 'finance', 'accounting', 'it', 'technology',
            'marketing', 'sales', 'operations', 'legal', 'compliance',
            'administration', 'procurement', 'logistics', 'customer service'
        ]

        for dept in departments:
            if dept in query_lower:
                parsed.department = dept.title()
                break
        
        # Extract policy duration
        duration_patterns = [
            r'(\d+)\s*(?:month|months|mon)\s*(?:old\s*)?policy',
            r'(\d+)\s*(?:year|years|yr)\s*(?:old\s*)?policy',
            r'policy\s*(?:of\s*)?(\d+)\s*(?:month|months|year|years)'
        ]
        
        for pattern in duration_patterns:
            duration_match = re.search(pattern, query_lower)
            if duration_match:
                duration_num = duration_match.group(1)
                if 'month' in duration_match.group(0):
                    parsed.policy_duration = f"{duration_num} months"
                else:
                    parsed.policy_duration = f"{duration_num} years"
                break
        
        # Extract amount
        amount_match = re.search(amount_pattern, query)
        if amount_match:
            try:
                amount_str = amount_match.group(1).replace(',', '')
                parsed.amount_claimed = float(amount_str)
            except (ValueError, IndexError):
                pass
        
        return parsed
    
    async def make_document_decision(self, query: StructuredQuery, evidence_chunks: List[Dict]) -> DocumentDecision:
        """Make decision based on parsed query and document evidence"""
        
        if not evidence_chunks:
            return DocumentDecision(
                decision="INSUFFICIENT_INFO",
                justification="No relevant clauses found for this query.",
                confidence=0.0,
                decision_type=query.document_type.upper() if query.document_type else "GENERAL"
            )
        
        # Prepare context for decision making
        context = self._prepare_decision_context(query, evidence_chunks)
        
        if self.model:
            try:
                decision = await self._gemini_decision(context, query, evidence_chunks)
                return decision
            except Exception as e:
                logger.error(f"Gemini decision failed: {e}")
        
        # Fallback rule-based decision
        return self._rule_based_decision(query, evidence_chunks)
    
    def _prepare_decision_context(self, query: StructuredQuery, evidence_chunks: List[Dict]) -> str:
        """Prepare context for decision making"""

        document_type_title = (query.document_type or 'Document').upper()

        context = f"""
        {document_type_title} EVALUATION

        REQUEST DETAILS:
        - Age: {query.age or 'Not specified'}
        - Gender: {query.gender or 'Not specified'}
        - Name: {query.name or 'Not specified'}
        - Employee ID: {query.employee_id or 'Not specified'}
        - Action/Request: {query.action or 'Not specified'}
        - Action Type: {query.action_type or 'Not specified'}
        - Location: {query.location or 'Not specified'}
        - Department: {query.department or 'Not specified'}
        - Duration: {query.duration or 'Not specified'}
        - Date: {query.date or 'Not specified'}
        - Amount: {query.currency} {query.amount or 'Not specified'}
        - Document Type: {query.document_type or 'General'}
        - Original Query: {query.raw_query}

        RELEVANT DOCUMENT CLAUSES:
        """
        
        for i, chunk in enumerate(evidence_chunks[:5], 1):
            context += f"\nClause {i}: {chunk.get('text', '')}\n"
        
        return context
    
    async def _gemini_decision(self, context: str, query: StructuredQuery, evidence_chunks: List[Dict]) -> DocumentDecision:
        """Use Gemini to make document-based decision"""
        
        document_type = query.document_type or 'document'

        # Generic decision criteria based on document type
        criteria_map = {
            'insurance': [
                "Is the procedure/treatment covered?",
                "Are there any age restrictions?",
                "Are there location restrictions?",
                "Is the policy duration sufficient?",
                "Are there any exclusions?"
            ],
            'hr': [
                "Is the employee eligible for this request?",
                "Are there any tenure requirements?",
                "Does this comply with company policy?",
                "Are there any departmental restrictions?",
                "Is the request within allowed limits?"
            ],
            'legal': [
                "Does this comply with legal requirements?",
                "Are there any regulatory restrictions?",
                "Is this within contractual terms?",
                "Are there any compliance issues?",
                "Does this meet audit standards?"
            ],
            'contract': [
                "Is this within contract scope?",
                "Are the terms and conditions met?",
                "Is this within the validity period?",
                "Are there any breach conditions?",
                "Does this require amendments?"
            ]
        }

        criteria = criteria_map.get(document_type, [
            "Is the request valid according to the document?",
            "Are all requirements met?",
            "Are there any restrictions that apply?",
            "Is this within allowed parameters?",
            "Does this comply with stated policies?"
        ])

        criteria_text = '\n        '.join([f"{i+1}. {c}" for i, c in enumerate(criteria)])

        # Generic decision options based on document type
        decision_options = {
            'insurance': '"APPROVED" | "REJECTED" | "PENDING"',
            'hr': '"APPROVED" | "REJECTED" | "PENDING" | "REQUIRES_MANAGER_APPROVAL"',
            'legal': '"COMPLIANT" | "NON_COMPLIANT" | "REQUIRES_REVIEW"',
            'contract': '"VALID" | "INVALID" | "REQUIRES_AMENDMENT"'
        }

        decisions = decision_options.get(document_type, '"APPROVED" | "REJECTED" | "PENDING"')

        prompt = f"""
        {context}

        Based on the request details and document clauses above, make a decision for this {document_type} query.

        Consider:
        {criteria_text}

        Respond with a JSON object containing:
        {{
            "decision": {decisions},
            "amount": <approved_amount_or_null>,
            "justification": "<detailed_explanation>",
            "clauses_referenced": [<list_of_clause_numbers_used>],
            "confidence": <0.0_to_1.0>
        }}
        """
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                result = json.loads(json_text)
                
                # Map clauses
                clauses_used = []
                for clause_num in result.get('clauses_referenced', []):
                    if isinstance(clause_num, int) and 1 <= clause_num <= len(evidence_chunks):
                        clauses_used.append({
                            'clause_number': clause_num,
                            'text': evidence_chunks[clause_num - 1].get('text', ''),
                            'relevance_score': evidence_chunks[clause_num - 1].get('relevance_score', 0)
                        })
                
                return DocumentDecision(
                    decision=result.get('decision', 'PENDING'),
                    amount=result.get('amount'),
                    justification=result.get('justification', ''),
                    clauses_used=clauses_used,
                    confidence=result.get('confidence', 0.5),
                    decision_type=query.document_type.upper() if query.document_type else "GENERAL"
                )
                
        except Exception as e:
            logger.error(f"Error in Gemini decision: {e}")
        
        # Fallback if Gemini fails
        return self._rule_based_decision(query, evidence_chunks)
    
    def _rule_based_decision(self, query: StructuredQuery, evidence_chunks: List[Dict]) -> DocumentDecision:
        """Simple rule-based decision as fallback"""
        
        # Simple keyword-based rules
        all_text = ' '.join([chunk.get('text', '').lower() for chunk in evidence_chunks])
        
        # Generic rule-based logic
        if query.action:
            action_lower = query.action.lower()
            positive_keywords = ['covered', 'allowed', 'approved', 'eligible', 'valid', 'compliant']
            negative_keywords = ['excluded', 'not covered', 'rejected', 'invalid', 'non-compliant']

            if any(keyword in all_text for keyword in positive_keywords):
                decision = "APPROVED"
                justification = f"{query.action} appears to be allowed based on document clauses."
                confidence = 0.7
            elif any(keyword in all_text for keyword in negative_keywords):
                decision = "REJECTED"
                justification = f"{query.action} appears to be restricted based on document clauses."
                confidence = 0.7
            else:
                decision = "PENDING"
                justification = f"Unclear whether {query.action} is allowed - requires manual review."
                confidence = 0.4
        else:
            decision = "PENDING"
            justification = "Insufficient information to make a decision."
            confidence = 0.3
        
        # Map evidence to clauses
        clauses_used = []
        for i, chunk in enumerate(evidence_chunks[:3], 1):
            clauses_used.append({
                'clause_number': i,
                'text': chunk.get('text', ''),
                'relevance_score': chunk.get('relevance_score', 0)
            })
        
        return DocumentDecision(
            decision=decision,
            justification=justification,
            clauses_used=clauses_used,
            confidence=confidence,
            decision_type=query.document_type.upper() if query.document_type else "GENERAL"
        )
