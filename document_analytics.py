import re
from typing import Dict, Any, List
from collections import Counter

class DocumentAnalytics:
    """Analyze document patterns and provide insights"""
    
    def __init__(self):
        self.document_stats = {}
        self.query_patterns = {}
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze document characteristics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Basic statistics
        stats = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'readability_score': self._calculate_readability(text),
            'document_type': self._classify_document_type(text),
            'key_topics': self._extract_key_topics(text),
            'complexity_score': self._calculate_complexity(text)
        }
        
        return stats
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower().strip('.,!?";')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content patterns"""
        text_lower = text.lower()
        
        type_indicators = {
            'insurance_policy': ['coverage', 'premium', 'deductible', 'claim', 'beneficiary'],
            'hr_policy': ['employee', 'leave', 'benefits', 'vacation', 'salary'],
            'legal_contract': ['agreement', 'terms', 'conditions', 'party', 'breach'],
            'compliance_doc': ['compliance', 'regulation', 'audit', 'requirement', 'standard'],
            'medical_document': ['patient', 'diagnosis', 'treatment', 'medical', 'prescription']
        }
        
        type_scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(text_lower.count(indicator) for indicator in indicators)
            type_scores[doc_type] = score
        
        if not any(type_scores.values()):
            return 'general_document'
        
        return max(type_scores, key=type_scores.get)
    
    def _extract_key_topics(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key topics using simple TF-IDF approach"""
        # Simple tokenization and filtering
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Remove common words
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'may', 'will',
            'shall', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want',
            'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here',
            'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than',
            'them', 'well', 'were'
        }
        
        filtered_words = [word for word in words if word not in common_words]
        word_freq = Counter(filtered_words)
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate document complexity score (0-100)"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Factors contributing to complexity
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_richness = unique_words / max(len(words), 1)
        
        # Legal/technical terms increase complexity
        complex_terms = [
            'notwithstanding', 'hereinafter', 'aforementioned', 'whereas', 
            'pursuant', 'hereunder', 'thereof', 'hereof', 'compliance',
            'regulation', 'statute', 'ordinance', 'liability', 'indemnity'
        ]
        
        complex_term_count = sum(text.lower().count(term) for term in complex_terms)
        
        # Normalize and combine factors
        complexity = (
            (avg_word_length - 4) * 10 +  # Longer words = more complex
            (avg_sentence_length - 15) * 2 +  # Longer sentences = more complex
            vocabulary_richness * 30 +  # Higher vocabulary = more complex
            complex_term_count * 5  # Legal terms = more complex
        )
        
        return max(0, min(100, complexity))