import re
import logging
from functools import lru_cache
from typing import List, Dict, Any

# Optional NLTK support
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """Optimize and classify queries for better processing"""

    def __init__(self):
        self.intent_patterns = {
            'coverage_check': [
                r'cover(s|ed|age)?', r'includ(e|ed|es)', r'eligible', r'benefit'
            ],
            'policy_lookup': [
                r'policy', r'rule(s)?', r'regulation(s)?', r'guideline(s)?'
            ],
            'compliance_check': [
                r'complian(ce|t)', r'requirement(s)?', r'mandator(y)?', r'legal'
            ],
            'condition_inquiry': [
                r'condition(s)?', r'requirement(s)?', r'criteria', r'prerequisite(s)?'
            ],
            'exclusion_check': [
                r'exclusion(s)?', r'not covered', r'except', r'exclude(d)?'
            ]
        }

    @lru_cache(maxsize=1000)
    def classify_intent(self, query: str) -> str:
        """Classify query intent using pattern matching"""
        query_lower = query.lower()
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score

        if not any(intent_scores.values()):
            return 'general_inquiry'

        return max(intent_scores, key=intent_scores.get)

    def optimize_query(self, query: str) -> str:
        """Optimize query for better search results"""
        if not NLTK_AVAILABLE:
            return query

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        stop_words = set(stopwords.words('english'))
        # Keep important question words
        important_words = {'what', 'when', 'where', 'how', 'why', 'which', 'who'}
        stop_words -= important_words

        tokens = word_tokenize(query.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

        return ' '.join(filtered_tokens)