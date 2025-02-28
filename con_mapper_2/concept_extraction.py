"""
Enhanced concept extraction module for the Philosophical Concept Map Generator.

This module provides advanced concept extraction capabilities using NLP techniques
to identify philosophical concepts in text with better accuracy and relevance scoring.
"""
import os
import re
import string
import json
from collections import Counter
from typing import List, Dict, Set, Tuple, Any, Optional

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import utility modules
try:
    from logging_utils import get_logger, log_execution
    from error_handling import handle_error, NLPProcessingError
    from config import config
except ImportError:
    # Fallback for standalone usage
    from logging import getLogger
    def get_logger(name):
        return getLogger(name)
    
    def log_execution(logger=None):
        def decorator(func):
            return func
        return decorator
    
    def handle_error(error, raise_exception=False, **kwargs):
        print(f"Error: {error}")
    
    class NLPProcessingError(Exception):
        pass
    
    # Simple config
    class SimpleConfig:
        def get(self, key, default=None):
            return default
    
    config = SimpleConfig()

# Initialize logger
logger = get_logger("concept_extraction")

# Initialize NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    error_msg = f"Error loading spaCy model: {e}"
    logger.error(error_msg)
    raise NLPProcessingError(error_msg, original_exception=e)

# Core philosophical terms loaded from config or default list
PHILOSOPHICAL_TERMS = config.get("PHILOSOPHICAL_TERMS", [
    "Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", 
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology", 
    "Determinism", "Free Will", "Consciousness", "Virtue Ethics", 
    "Deontology", "Utilitarianism", "Moral Realism", "Relativism",
    "Ontology", "Dualism", "Materialism", "Idealism", "Pragmatism",
    "Positivism", "Skepticism", "Nihilism", "Subjectivism", "Objectivism"
])

# Load philosophical dictionary if available
PHILOSOPHICAL_DICT_PATH = os.path.join(
    config.get("DATA_DIR", "data"), 
    "philosophical_dictionary.json"
)

try:
    if os.path.exists(PHILOSOPHICAL_DICT_PATH):
        with open(PHILOSOPHICAL_DICT_PATH, 'r', encoding='utf-8') as f:
            PHILOSOPHICAL_DICT = json.load(f)
            logger.info(f"Loaded philosophical dictionary with {len(PHILOSOPHICAL_DICT)} concepts")
    else:
        PHILOSOPHICAL_DICT = {}
        logger.warning("Philosophical dictionary not found")
except Exception as e:
    PHILOSOPHICAL_DICT = {}
    logger.error(f"Error loading philosophical dictionary: {e}")


@log_execution()
def extract_concepts_basic(text: str) -> List[str]:
    """
    Basic concept extraction using NER and keyword matching.
    
    Args:
        text: The text to extract concepts from
        
    Returns:
        List of extracted concepts
    """
    if not text:
        return []
    
    # Process with spaCy
    max_length = config.get("MAX_TEXT_LENGTH", 20000)
    doc = nlp(text[:max_length])
    
    # NER extraction
    ner_concepts = set()
    relevant_entity_types = config.get("RELEVANT_ENTITY_TYPES", 
                                     ["ORG", "PERSON", "NORP", "MISC", "GPE"])
    
    for ent in doc.ents:
        if ent.label_ in relevant_entity_types:
            ner_concepts.add(ent.text)
    
    # Keyword matching
    keyword_concepts = set()
    text_lower = text.lower()
    for term in PHILOSOPHICAL_TERMS:
        if term.lower() in text_lower:
            keyword_concepts.add(term)
    
    # Dictionary matching
    dict_concepts = set()
    for concept in PHILOSOPHICAL_DICT.keys():
        if concept.lower() in text_lower:
            dict_concepts.add(concept)
    
    # Combine results
    all_concepts = sorted(list(ner_concepts.union(keyword_concepts).union(dict_concepts)))
    return all_concepts


def extract_noun_chunks(doc) -> Set[str]:
    """
    Extract normalized noun chunks from a spaCy document.
    
    Args:
        doc: spaCy document
        
    Returns:
        Set of normalized noun chunks
    """
    chunks = set()
    for chunk in doc.noun_chunks:
        # Filter out chunks with stopwords or determiners only
        if not all(token.is_stop or token.pos_ == "DET" for token in chunk):
            # Normalize by removing determiners and adjectives if necessary
            filtered_chunk = " ".join([token.text for token in chunk 
                                     if not (token.pos_ == "DET")])
            if filtered_chunk:
                chunks.add(filtered_chunk)
    return chunks


def filter_candidate_concepts(candidates: Set[str], min_length: int = 3) -> Set[str]:
    """
    Filter candidate concepts based on various criteria.
    
    Args:
        candidates: Set of candidate concepts
        min_length: Minimum length for a concept (in characters)
        
    Returns:
        Filtered set of concepts
    """
    filtered = set()
    for candidate in candidates:
        # Skip if too short
        if len(candidate) < min_length:
            continue
            
        # Skip if just numbers or punctuation
        if all(c in string.punctuation or c.isdigit() or c.isspace() for c in candidate):
            continue
            
        # Skip common stopwords and irrelevant terms
        if candidate.lower() in {"the", "a", "an", "this", "that", "these", "those", 
                                "it", "they", "them", "he", "she", "his", "her"}:
            continue
            
        filtered.add(candidate)
    
    return filtered


@log_execution()
def extract_concepts_advanced(text: str, primary_concept: str = None) -> List[Dict[str, Any]]:
    """
    Advanced concept extraction with relevance scoring.
    
    Args:
        text: The text to extract concepts from
        primary_concept: The primary concept for relevance scoring
        
    Returns:
        List of dictionaries with concept info including relevance scores
    """
    if not text:
        return []
    
    # Process with spaCy
    max_length = config.get("MAX_TEXT_LENGTH", 20000)
    doc = nlp(text[:max_length])
    
    # Extract candidate concepts using multiple methods
    candidates = set()
    
    # 1. NER extraction
    relevant_entity_types = config.get("RELEVANT_ENTITY_TYPES", 
                                     ["ORG", "PERSON", "NORP", "MISC", "GPE"])
    for ent in doc.ents:
        if ent.label_ in relevant_entity_types:
            candidates.add(ent.text)
    
    # 2. Noun chunk extraction
    noun_chunks = extract_noun_chunks(doc)
    candidates.update(noun_chunks)
    
    # 3. Keyword matching
    text_lower = text.lower()
    for term in PHILOSOPHICAL_TERMS:
        if term.lower() in text_lower:
            candidates.add(term)
    
    # 4. Dictionary matching
    for concept in PHILOSOPHICAL_DICT.keys():
        if concept.lower() in text_lower:
            candidates.add(concept)
    
    # Filter candidates
    filtered_candidates = filter_candidate_concepts(candidates)
    
    # Score candidates for relevance
    scored_concepts = score_concepts(text, filtered_candidates, primary_concept)
    
    # Sort by relevance score
    sorted_concepts = sorted(scored_concepts, key=lambda x: x['relevance_score'], reverse=True)
    
    return sorted_concepts


def score_concepts(text: str, candidates: Set[str], primary_concept: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Score candidate concepts for relevance.
    
    Args:
        text: The source text
        candidates: Set of candidate concepts
        primary_concept: The primary concept for contextualized scoring
        
    Returns:
        List of dictionaries with concept info including relevance scores
    """
    # Prepare the document
    sentences = [sent.text for sent in nlp(text).sents]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    
    # If no valid sentences, return empty-scored concepts
    if not sentences:
        return [{'concept': c, 'relevance_score': 0.0, 'frequency': 0} for c in candidates]
    
    try:
        # Calculate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        logger.warning(f"Error calculating TF-IDF: {e}. Using basic scoring instead.")
        # Fall back to basic frequency scoring
        return score_concepts_basic(text, candidates)
    
    scored_concepts = []
    text_lower = text.lower()
    
    for concept in candidates:
        # Calculate raw frequency
        concept_lower = concept.lower()
        frequency = text_lower.count(concept_lower)
        
        # Skip concepts that don't actually appear in the text
        if frequency == 0:
            continue
        
        # Calculate TF-IDF score if possible
        tfidf_score = 0.0
        try:
            # Get the index of the concept in the vocabulary
            indices = [i for i, feature in enumerate(feature_names) 
                     if concept_lower in feature or feature in concept_lower]
            
            if indices:
                # Average the TF-IDF scores for matching features
                tfidf_scores = [tfidf_matrix[:, idx].mean() for idx in indices]
                tfidf_score = np.mean(tfidf_scores)
        except Exception as e:
            logger.debug(f"Couldn't calculate TF-IDF for '{concept}': {e}")
        
        # Calculate co-occurrence with primary concept if provided
        cooccurrence_score = 0.0
        if primary_concept:
            # Count sentences where both concepts appear
            cooccurrence_count = sum(1 for sent in sentences 
                                  if concept_lower in sent.lower() 
                                  and primary_concept.lower() in sent.lower())
            
            # Normalize by frequency
            if frequency > 0:
                cooccurrence_score = cooccurrence_count / frequency
        
        # Look up in philosophical dictionary for domain score
        domain_score = 0.0
        if concept in PHILOSOPHICAL_DICT:
            domain_score = 1.0
        elif any(concept_lower in term.lower() or term.lower() in concept_lower 
               for term in PHILOSOPHICAL_TERMS):
            domain_score = 0.7
        
        # Calculate final relevance score (weighted combination)
        relevance_score = (
            0.3 * min(frequency / 10, 1.0) +  # Frequency (capped at 10)
            0.3 * float(tfidf_score) +        # TF-IDF score
            0.2 * cooccurrence_score +        # Co-occurrence score
            0.2 * domain_score                # Domain relevance score
        )
        
        # Add to results
        scored_concepts.append({
            'concept': concept,
            'relevance_score': relevance_score,
            'frequency': frequency,
            'tfidf_score': float(tfidf_score),
            'domain_score': domain_score,
            'cooccurrence_score': cooccurrence_score
        })
    
    return scored_concepts


def score_concepts_basic(text: str, candidates: Set[str]) -> List[Dict[str, Any]]:
    """
    Basic concept scoring based on frequency and dictionary lookup.
    Used as a fallback if TF-IDF fails.
    
    Args:
        text: The source text
        candidates: Set of candidate concepts
        
    Returns:
        List of dictionaries with concept info including basic relevance scores
    """
    scored_concepts = []
    text_lower = text.lower()
    
    for concept in candidates:
        # Calculate raw frequency
        concept_lower = concept.lower()
        frequency = text_lower.count(concept_lower)
        
        # Skip concepts that don't actually appear in the text
        if frequency == 0:
            continue
        
        # Calculate domain score based on dictionary
        domain_score = 0.0
        if concept in PHILOSOPHICAL_DICT:
            domain_score = 1.0
        elif any(concept_lower in term.lower() or term.lower() in concept_lower 
                for term in PHILOSOPHICAL_TERMS):
            domain_score = 0.7
        
        # Calculate simple relevance score
        relevance_score = (
            0.7 * min(frequency / 10, 1.0) +  # Frequency (capped at 10)
            0.3 * domain_score                # Domain relevance score
        )
        
        # Add to results
        scored_concepts.append({
            'concept': concept,
            'relevance_score': relevance_score,
            'frequency': frequency,
            'domain_score': domain_score
        })
    
    return scored_concepts


@log_execution()
def filter_concepts(concepts: List[Dict[str, Any]], 
                   min_score: float = 0.0, 
                   max_concepts: int = None) -> List[Dict[str, Any]]:
    """
    Filter concepts based on relevance score and limit the number.
    
    Args:
        concepts: List of concept dictionaries with relevance scores
        min_score: Minimum relevance score to include
        max_concepts: Maximum number of concepts to include
        
    Returns:
        Filtered list of concept dictionaries
    """
    # Filter by minimum score
    filtered = [c for c in concepts if c['relevance_score'] >= min_score]
    
    # Sort by relevance score
    sorted_concepts = sorted(filtered, key=lambda x: x['relevance_score'], reverse=True)
    
    # Limit number of concepts if specified
    if max_concepts is not None and max_concepts > 0:
        sorted_concepts = sorted_concepts[:max_concepts]
    
    return sorted_concepts


@log_execution()
def build_philosophical_dictionary() -> Dict[str, Dict[str, Any]]:
    """
    Build or update the philosophical dictionary from various sources.
    
    Returns:
        Dictionary of philosophical concepts with metadata
    """
    # Start with core philosophical terms
    dictionary = {term: {"category": "Core Term", "relevance": 1.0} for term in PHILOSOPHICAL_TERMS}
    
    # Add terms from Stanford Encyclopedia of Philosophy if available
    sep_terms_path = os.path.join(config.get("DATA_DIR", "data"), "sep_terms.json")
    if os.path.exists(sep_terms_path):
        try:
            with open(sep_terms_path, 'r', encoding='utf-8') as f:
                sep_terms = json.load(f)
                for term, data in sep_terms.items():
                    dictionary[term] = {
                        "category": "SEP Entry",
                        "relevance": 0.9,
                        **data
                    }
            logger.info(f"Added {len(sep_terms)} terms from Stanford Encyclopedia of Philosophy")
        except Exception as e:
            logger.error(f"Error loading SEP terms: {e}")
    
    # Save the dictionary
    try:
        with open(PHILOSOPHICAL_DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2)
        logger.info(f"Saved philosophical dictionary with {len(dictionary)} concepts")
    except Exception as e:
        logger.error(f"Error saving philosophical dictionary: {e}")
    
    return dictionary


if __name__ == "__main__":
    # Example usage
    build_philosophical_dictionary()
    
    test_text = """
    Ethics, also called moral philosophy, is the discipline concerned with what is morally good and bad and morally right and wrong. The term is also applied to any system or theory of moral values or principles.
    
    Utilitarianism is the view that the right moral action is the one that maximizes utility, which is usually defined in terms of happiness or pleasure.
    """
    
    print("Basic concept extraction:")
    basic_concepts = extract_concepts_basic(test_text)
    print(basic_concepts)
    
    print("\nAdvanced concept extraction:")
    advanced_concepts = extract_concepts_advanced(test_text, "Ethics")
    for concept in advanced_concepts:
        print(f"{concept['concept']}: {concept['relevance_score']:.2f} (freq: {concept['frequency']})")