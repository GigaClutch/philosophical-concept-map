"""
Enhanced relationship extraction module for the Philosophical Concept Map Generator.

This module provides advanced capabilities for extracting semantic relationships
between philosophical concepts, including relationship types and directionality.
"""
import re
import string
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc, Span

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
logger = get_logger("relationship_extraction")

# Initialize NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    error_msg = f"Error loading spaCy model: {e}"
    logger.error(error_msg)
    raise NLPProcessingError(error_msg, original_exception=e)


# Define relationship patterns
RELATIONSHIP_PATTERNS = [
    # Format: (pattern regex, relationship type, is_directed, direction_correct)
    
    # Definitional patterns
    (r"(\w+) is (?:a type of|a kind of|a form of) (\w+)", "is_a", True, True),
    (r"(\w+) are (?:types of|kinds of|forms of) (\w+)", "is_a", True, True),
    (r"(\w+) is defined as (\w+)", "is_defined_as", True, True),
    
    # Causal patterns
    (r"(\w+) causes (\w+)", "causes", True, True),
    (r"(\w+) leads to (\w+)", "leads_to", True, True),
    (r"(\w+) results in (\w+)", "results_in", True, True),
    (r"(\w+) is caused by (\w+)", "causes", True, False),  # Reversed
    
    # Oppositional patterns
    (r"(\w+) opposes (\w+)", "opposes", True, True),
    (r"(\w+) contradicts (\w+)", "contradicts", True, True),
    (r"(\w+) is contrary to (\w+)", "is_contrary_to", True, True),
    
    # Similarity patterns
    (r"(\w+) is similar to (\w+)", "is_similar_to", False, True),
    (r"(\w+) resembles (\w+)", "resembles", False, True),
    (r"(\w+) is related to (\w+)", "is_related_to", False, True),
    
    # Temporal patterns
    (r"(\w+) precedes (\w+)", "precedes", True, True),
    (r"(\w+) follows (\w+)", "follows", True, True),
    (r"(\w+) developed after (\w+)", "developed_after", True, True),
    (r"(\w+) influenced (\w+)", "influenced", True, True),
    (r"(\w+) was influenced by (\w+)", "influenced", True, False),  # Reversed
    
    # Part-whole patterns
    (r"(\w+) is part of (\w+)", "is_part_of", True, True),
    (r"(\w+) contains (\w+)", "contains", True, True),
    (r"(\w+) includes (\w+)", "includes", True, True),
    
    # General association patterns
    (r"(\w+) and (\w+)", "associated_with", False, True),
    (r"(?:both) (\w+) and (\w+)", "associated_with", False, True)
]


class RelationshipType:
    """Predefined relationship types with metadata."""
    
    # Definitional relationships
    IS_A = "is_a"  # Hyponymy
    PART_OF = "part_of"  # Meronymy
    DEFINED_AS = "defined_as"
    
    # Causality relationships
    CAUSES = "causes"
    LEADS_TO = "leads_to"
    INFLUENCES = "influences"
    
    # Oppositional relationships
    OPPOSES = "opposes"
    CONTRADICTS = "contradicts"
    CONTRASTS_WITH = "contrasts_with"
    
    # Similarity relationships
    SIMILAR_TO = "similar_to"
    AGREES_WITH = "agrees_with"
    RELATED_TO = "related_to"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONTEMPORARY_WITH = "contemporary_with"
    
    # Generic relationship
    ASSOCIATED_WITH = "associated_with"
    MENTIONED_WITH = "mentioned_with"
    
    # Map verb phrases to relationship types
    VERB_PHRASE_MAP = {
        # Definitional verbs
        "is": IS_A,
        "are": IS_A,
        "is defined as": DEFINED_AS,
        "is a type of": IS_A,
        "is part of": PART_OF,
        "belongs to": PART_OF,
        
        # Causal verbs
        "causes": CAUSES,
        "leads to": LEADS_TO,
        "results in": LEADS_TO,
        "influences": INFLUENCES,
        "affects": INFLUENCES,
        "impacts": INFLUENCES,
        
        # Oppositional verbs
        "opposes": OPPOSES,
        "contradicts": CONTRADICTS,
        "denies": CONTRADICTS,
        "contrasts with": CONTRASTS_WITH,
        "differs from": CONTRASTS_WITH,
        
        # Similarity verbs
        "is similar to": SIMILAR_TO,
        "resembles": SIMILAR_TO,
        "agrees with": AGREES_WITH,
        "is related to": RELATED_TO,
        "connects to": RELATED_TO,
        
        # Temporal verbs
        "precedes": PRECEDES,
        "comes before": PRECEDES,
        "follows": FOLLOWS,
        "comes after": FOLLOWS,
        "is contemporary with": CONTEMPORARY_WITH
    }
    
    # Map dependency paths to relationship types
    DEP_PATH_MAP = {
        # Subject-verb-object paths
        "nsubj:ROOT:dobj": INFLUENCES,
        "nsubj:ROOT:prep:pobj": RELATED_TO,
        
        # Conjunction paths
        "conj": ASSOCIATED_WITH
    }
    
    # Relationship properties
    PROPERTIES = {
        # Format: relation_type: (is_directed, is_symmetric, is_transitive)
        IS_A: (True, False, True),
        PART_OF: (True, False, True),
        DEFINED_AS: (True, False, False),
        CAUSES: (True, False, False),
        LEADS_TO: (True, False, True),
        INFLUENCES: (True, False, False),
        OPPOSES: (True, True, False),
        CONTRADICTS: (True, True, False),
        CONTRASTS_WITH: (True, True, False),
        SIMILAR_TO: (False, True, False),
        AGREES_WITH: (False, True, False),
        RELATED_TO: (False, True, False),
        PRECEDES: (True, False, True),
        FOLLOWS: (True, False, True),
        CONTEMPORARY_WITH: (False, True, False),
        ASSOCIATED_WITH: (False, True, False),
        MENTIONED_WITH: (False, True, False)
    }
    
    @classmethod
    def is_directed(cls, relation_type: str) -> bool:
        """Check if a relationship type is directed."""
        return cls.PROPERTIES.get(relation_type, (False, False, False))[0]
    
    @classmethod
    def is_symmetric(cls, relation_type: str) -> bool:
        """Check if a relationship type is symmetric."""
        return cls.PROPERTIES.get(relation_type, (False, False, False))[1]
    
    @classmethod
    def is_transitive(cls, relation_type: str) -> bool:
        """Check if a relationship type is transitive."""
        return cls.PROPERTIES.get(relation_type, (False, False, False))[2]
    
    @classmethod
    def get_inverse(cls, relation_type: str) -> str:
        """Get the inverse of a relationship type."""
        # Define inverse relationships
        INVERSES = {
            cls.IS_A: "has_type",
            cls.PART_OF: "has_part",
            cls.CAUSES: "caused_by",
            cls.LEADS_TO: "follows_from",
            cls.INFLUENCES: "influenced_by",
            cls.PRECEDES: cls.FOLLOWS,
            cls.FOLLOWS: cls.PRECEDES
        }
        
        # Return the inverse if defined, otherwise return the original
        # (for symmetric relationships)
        return INVERSES.get(relation_type, relation_type)


@log_execution()
def extract_basic_relationships(input_concept: str, wiki_text: str, 
                              extracted_concepts: List[str]) -> Dict[Any, Dict[str, Any]]:
    """
    Extract basic co-occurrence relationships between concepts.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        
    Returns:
        Dictionary of relationships between concept pairs
    """
    if not wiki_text:
        return {}
    
    logger.info(f"Extracting basic relationships for '{input_concept}'")
    
    # Process text with spaCy
    max_length = config.get("MAX_TEXT_LENGTH", 20000)
    doc = nlp(wiki_text[:max_length])
    sentences = list(doc.sents)
    
    relationship_data = {}
    input_concept_lower = input_concept.lower()
    
    # Process each sentence
    for sentence in sentences:
        sentence_text = sentence.text.lower()
        
        # Check if input concept is mentioned
        if input_concept_lower in sentence_text:
            # Find concepts that co-occur in this sentence
            for concept in extracted_concepts:
                concept_lower = concept.lower()
                
                # Skip self-relationships
                if concept_lower == input_concept_lower:
                    continue
                
                # Check if the concept appears in this sentence
                if concept_lower in sentence_text:
                    # Create a unique key for this concept pair
                    pair_key = frozenset({input_concept, concept})
                    
                    # Initialize data structure if this is the first occurrence
                    if pair_key not in relationship_data:
                        relationship_data[pair_key] = {
                            "count": 0,
                            "sentences": [],
                            "relationship_type": RelationshipType.MENTIONED_WITH,
                            "is_directed": False,
                            "direction": None  # None for undirected
                        }
                    
                    # Update relationship data
                    relationship_data[pair_key]["count"] += 1
                    
                    # Store the sentence if we haven't stored too many already
                    max_sentences = config.get("MAX_SENTENCES_PER_CONCEPT", 5)
                    if len(relationship_data[pair_key]["sentences"]) < max_sentences:
                        relationship_data[pair_key]["sentences"].append(sentence.text)
    
    logger.info(f"Found {len(relationship_data)} basic relationships")
    return relationship_data


def extract_verb_phrase(doc: Doc, subj_index: int, obj_index: int) -> str:
    """
    Extract the verb phrase connecting a subject and object.
    
    Args:
        doc: The spaCy document
        subj_index: Index of the subject token
        obj_index: Index of the object token
        
    Returns:
        The verb phrase as a string
    """
    # Determine the range to consider
    start = min(subj_index, obj_index)
    end = max(subj_index, obj_index)
    
    # Find the verb(s) between subject and object
    verbs = []
    for i in range(start, end + 1):
        token = doc[i]
        if token.pos_ == "VERB":
            verbs.append(token)
    
    if not verbs:
        return ""
    
    # For simple cases, just return the verb
    if len(verbs) == 1:
        verb = verbs[0]
        children = list(verb.children)
        
        # Include particles and prepositions
        verb_phrase = verb.text
        for child in children:
            if child.dep_ in ["prt", "prep"] and child.i > verb.i:
                verb_phrase += " " + child.text
        
        return verb_phrase.lower()
    
    # For multiple verbs, try to find the main one
    root_verbs = [v for v in verbs if v.dep_ == "ROOT"]
    if root_verbs:
        main_verb = root_verbs[0]
    else:
        # Just use the first verb if no root
        main_verb = verbs[0]
        
    # Get the verb phrase
    verb_phrase = main_verb.text
    children = list(main_verb.children)
    
    # Include particles and prepositions
    for child in children:
        if child.dep_ in ["prt", "prep"]:
            verb_phrase += " " + child.text
    
    return verb_phrase.lower()


def get_dependency_path(doc: Doc, subj_token: spacy.tokens.Token, 
                      obj_token: spacy.tokens.Token) -> str:
    """
    Get the dependency path between subject and object tokens.
    
    Args:
        doc: The spaCy document
        subj_token: The subject token
        obj_token: The object token
        
    Returns:
        String representation of the dependency path
    """
    # Get the dependency path (simplified implementation)
    if subj_token.head == obj_token:
        return f"{subj_token.dep_}:{obj_token.dep_}"
    elif obj_token.head == subj_token:
        return f"{obj_token.dep_}:{subj_token.dep_}"
    elif subj_token.head == obj_token.head:
        return f"{subj_token.dep_}:{subj_token.head.dep_}:{obj_token.dep_}"
    
    # For more complex cases, just indicate a general connection
    return "connected"


def analyze_sentence_structure(sentence: Span, concept1: str, concept2: str) -> Dict[str, Any]:
    """
    Analyze the syntactic structure of a sentence to extract relationship information.
    
    Args:
        sentence: spaCy sentence span
        concept1: First concept
        concept2: Second concept
        
    Returns:
        Dictionary with relationship information
    """
    result = {
        "verb_phrase": "",
        "dependency_path": "",
        "is_directed": False,
        "direction": None  # None, True (1→2), or False (2→1)
    }
    
    # Find the tokens corresponding to the concepts
    concept1_lower = concept1.lower()
    concept2_lower = concept2.lower()
    
    # Check if either concept is a multi-word phrase
    if " " in concept1_lower or " " in concept2_lower:
        # For multi-word concepts, we need to check for the entire phrase
        text_lower = sentence.text.lower()
        
        # Find the approximate positions of the concepts in the text
        pos1 = text_lower.find(concept1_lower)
        pos2 = text_lower.find(concept2_lower)
        
        if pos1 == -1 or pos2 == -1:
            return result
        
        # Use character offset to find the closest token
        concept1_tokens = [token for token in sentence 
                         if token.idx <= pos1 and token.idx + len(token) >= pos1]
        concept2_tokens = [token for token in sentence 
                         if token.idx <= pos2 and token.idx + len(token) >= pos2]
        
        if not concept1_tokens or not concept2_tokens:
            return result
            
        concept1_token = concept1_tokens[0]
        concept2_token = concept2_tokens[0]
    else:
        # For single-word concepts, we can search the tokens directly
        concept1_tokens = [token for token in sentence if token.text.lower() == concept1_lower]
        concept2_tokens = [token for token in sentence if token.text.lower() == concept2_lower]
        
        if not concept1_tokens or not concept2_tokens:
            return result
            
        concept1_token = concept1_tokens[0]
        concept2_token = concept2_tokens[0]
    
    # Extract the verb phrase
    verb_phrase = extract_verb_phrase(sentence, concept1_token.i, concept2_token.i)
    result["verb_phrase"] = verb_phrase
    
    # Get dependency path
    dep_path = get_dependency_path(sentence, concept1_token, concept2_token)
    result["dependency_path"] = dep_path
    
    # Determine directionality
    if verb_phrase:
        # Look up the verb phrase in the relationship type map
        rel_type = RelationshipType.VERB_PHRASE_MAP.get(verb_phrase, RelationshipType.MENTIONED_WITH)
        is_directed = RelationshipType.is_directed(rel_type)
        
        result["is_directed"] = is_directed
        
        # For directed relationships, determine the direction based on syntax
        if is_directed:
            # Check if concept1 is the subject
            if concept1_token.dep_ in ["nsubj", "nsubjpass"]:
                result["direction"] = True  # 1 → 2
            # Check if concept2 is the subject
            elif concept2_token.dep_ in ["nsubj", "nsubjpass"]:
                result["direction"] = False  # 2 → 1
    
    return result


@log_execution()
def extract_advanced_relationships(input_concept: str, wiki_text: str, 
                                 extracted_concepts: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """
    Extract advanced semantic relationships between concepts.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of concept dictionaries from advanced extraction
        
    Returns:
        Dictionary of relationships with semantic information
    """
    if not wiki_text:
        return {}
    
    logger.info(f"Extracting advanced relationships for '{input_concept}'")
    
    # Process text with spaCy
    max_length = config.get("MAX_TEXT_LENGTH", 20000)
    doc = nlp(wiki_text[:max_length])
    sentences = list(doc.sents)
    
    # Extract concept strings from dictionaries
    concept_strings = [c['concept'] for c in extracted_concepts]
    
    relationship_data = {}
    input_concept_lower = input_concept.lower()
    
    # Process each sentence
    for sentence in sentences:
        sentence_text = sentence.text.lower()
        
        # Check if input concept is mentioned
        if input_concept_lower in sentence_text:
            # Find concepts that co-occur in this sentence
            for concept in concept_strings:
                concept_lower = concept.lower()
                
                # Skip self-relationships
                if concept_lower == input_concept_lower:
                    continue
                
                # Check if the concept appears in this sentence
                if concept_lower in sentence_text:
                    # Create a unique key for this concept pair
                    pair_key = frozenset({input_concept, concept})
                    
                    # Initialize data structure if this is the first occurrence
                    if pair_key not in relationship_data:
                        relationship_data[pair_key] = {
                            "count": 0,
                            "sentences": [],
                            "relationship_type": RelationshipType.MENTIONED_WITH,
                            "is_directed": False,
                            "direction": None,  # None for undirected
                            "verbs": Counter(),
                            "examples": []
                        }
                    
                    # Update relationship data
                    rel_data = relationship_data[pair_key]
                    rel_data["count"] += 1
                    
                    # Analyze sentence structure
                    structure_info = analyze_sentence_structure(sentence, input_concept, concept)
                    
                    # Extract verb phrase
                    if structure_info["verb_phrase"]:
                        rel_data["verbs"][structure_info["verb_phrase"]] += 1
                    
                    # Determine relationship type based on verb phrase
                    verb_phrase = structure_info["verb_phrase"]
                    if verb_phrase:
                        rel_type = RelationshipType.VERB_PHRASE_MAP.get(
                            verb_phrase, RelationshipType.MENTIONED_WITH)
                        
                        # Set relationship type if it's more specific than the current one
                        if rel_type != RelationshipType.MENTIONED_WITH:
                            rel_data["relationship_type"] = rel_type
                            rel_data["is_directed"] = RelationshipType.is_directed(rel_type)
                            
                            # Set direction based on sentence analysis
                            if rel_data["is_directed"]:
                                direction = structure_info["direction"]
                                if direction is not None:
                                    rel_data["direction"] = direction
                    
                    # Store the sentence if we haven't stored too many already
                    max_sentences = config.get("MAX_SENTENCES_PER_CONCEPT", 5)
                    if len(rel_data["sentences"]) < max_sentences:
                        rel_data["sentences"].append(sentence.text)
                        
                        # If we have a verb phrase, store this as an example
                        if verb_phrase and len(rel_data["examples"]) < max_sentences:
                            rel_data["examples"].append({
                                "sentence": sentence.text,
                                "verb_phrase": verb_phrase,
                                "dependency_path": structure_info["dependency_path"]
                            })
    
    # For each relationship, determine the most common verb
    for pair_key, rel_data in relationship_data.items():
        verbs = rel_data["verbs"]
        if verbs:
            most_common_verb = verbs.most_common(1)[0][0]
            rel_data["most_common_verb"] = most_common_verb
            
            # Update relationship type based on most common verb if not already set
            if rel_data["relationship_type"] == RelationshipType.MENTIONED_WITH:
                rel_type = RelationshipType.VERB_PHRASE_MAP.get(
                    most_common_verb, RelationshipType.MENTIONED_WITH)
                
                if rel_type != RelationshipType.MENTIONED_WITH:
                    rel_data["relationship_type"] = rel_type
                    rel_data["is_directed"] = RelationshipType.is_directed(rel_type)
    
    logger.info(f"Found {len(relationship_data)} advanced relationships")
    return relationship_data


@log_execution()
def extract_pattern_based_relationships(text: str, concepts: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Extract relationships using pattern matching.
    
    Args:
        text: The text to analyze
        concepts: List of concept strings
        
    Returns:
        Dictionary of relationships between concept pairs
    """
    relationships = {}
    
    # Prepare concepts for pattern matching
    concept_patterns = {c: r'\b' + re.escape(c) + r'\b' for c in concepts}
    
    # For each relationship pattern
    for pattern, rel_type, is_directed, direction_correct in RELATIONSHIP_PATTERNS:
        # Find all matches
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Get the concepts in the match
            concept1_text = match.group(1)
            concept2_text = match.group(2)
            
            # Find which concepts these match
            matching_concepts1 = [c for c in concepts 
                                if concept1_text.lower() in c.lower() 
                                or c.lower() in concept1_text.lower()]
            
            matching_concepts2 = [c for c in concepts 
                                if concept2_text.lower() in c.lower() 
                                or c.lower() in concept2_text.lower()]
            
            # If both match known concepts
            if matching_concepts1 and matching_concepts2:
                # Use the closest matching concepts
                concept1 = matching_concepts1[0]
                concept2 = matching_concepts2[0]
                
                # Skip self-relationships
                if concept1 == concept2:
                    continue
                
                # Create the relationship key
                # For directed relationships, use a tuple instead of frozenset to preserve direction
                if is_directed:
                    if direction_correct:
                        rel_key = (concept1, concept2)
                    else:
                        rel_key = (concept2, concept1)
                else:
                    rel_key = frozenset([concept1, concept2])
                
                # Initialize or update the relationship data
                if rel_key not in relationships:
                    relationships[rel_key] = {
                        "count": 0,
                        "relationship_type": rel_type,
                        "is_directed": is_directed,
                        "matches": []
                    }
                
                relationships[rel_key]["count"] += 1
                
                # Store the match context (text before and after the match)
                match_start = max(0, match.start() - 50)
                match_end = min(len(text), match.end() + 50)
                context = text[match_start:match_end]
                
                # Add to matches if we don't have too many
                if len(relationships[rel_key]["matches"]) < 5:
                    relationships[rel_key]["matches"].append({
                        "text": context,
                        "pattern": pattern
                    })
    
    logger.info(f"Found {len(relationships)} pattern-based relationships")
    return relationships


@log_execution()
def merge_relationships(relationships1: Dict, relationships2: Dict) -> Dict:
    """
    Merge two relationship dictionaries.
    
    Args:
        relationships1: First relationship dictionary
        relationships2: Second relationship dictionary
        
    Returns:
        Merged relationship dictionary
    """
    merged = relationships1.copy()
    
    # Add or merge relationships from the second dictionary
    for key, rel_data in relationships2.items():
        if key in merged:
            # Merge counts
            merged[key]["count"] += rel_data["count"]
            
            # Take the more specific relationship type
            if merged[key]["relationship_type"] == RelationshipType.MENTIONED_WITH and \
               rel_data["relationship_type"] != RelationshipType.MENTIONED_WITH:
                merged[key]["relationship_type"] = rel_data["relationship_type"]
                merged[key]["is_directed"] = rel_data["is_directed"]
            
            # Merge other fields if they exist
            if "sentences" in merged[key] and "sentences" in rel_data:
                merged[key]["sentences"].extend(rel_data["sentences"])
                
            if "verbs" in merged[key] and "verbs" in rel_data:
                for verb, count in rel_data["verbs"].items():
                    merged[key]["verbs"][verb] += count
                    
            if "examples" in merged[key] and "examples" in rel_data:
                merged[key]["examples"].extend(rel_data["examples"])
                
            if "matches" in rel_data:
                if "matches" not in merged[key]:
                    merged[key]["matches"] = []
                merged[key]["matches"].extend(rel_data["matches"])
        else:
            # Add the relationship as is
            merged[key] = rel_data
    
    return merged


def filter_relationships(relationships: Dict, min_count: int = 1) -> Dict:
    """
    Filter relationships based on occurrence count.
    
    Args:
        relationships: Dictionary of relationships
        min_count: Minimum occurrence count to include
        
    Returns:
        Filtered relationship dictionary
    """
    return {key: data for key, data in relationships.items() if data["count"] >= min_count}


@log_execution()
def process_relationships(input_concept: str, wiki_text: str, 
                        extracted_concepts: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """
    Process relationships between concepts using multiple methods.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of concept dictionaries from advanced extraction
        
    Returns:
        Dictionary of relationships with semantic information
    """
    if not wiki_text:
        return {}
    
    # Extract the concept strings from the dictionaries
    concept_strings = [c['concept'] for c in extracted_concepts]
    
    # 1. Extract advanced relationships
    advanced_relations = extract_advanced_relationships(input_concept, wiki_text, extracted_concepts)
    
    # 2. Extract pattern-based relationships
    pattern_relations = extract_pattern_based_relationships(wiki_text, concept_strings)
    
    # 3. Merge the relationships
    merged_relations = merge_relationships(advanced_relations, pattern_relations)
    
    # 4. Filter out relationships with low counts
    min_count = config.get("MIN_RELATIONSHIP_COUNT", 1)
    filtered_relations = filter_relationships(merged_relations, min_count=min_count)
    
    return filtered_relations


if __name__ == "__main__":
    # Example usage
    test_text = """
    Ethics, also called moral philosophy, is the discipline concerned with what is morally good and bad and morally right and wrong. The term is also applied to any system or theory of moral values or principles.
    
    Utilitarianism is a type of consequentialism that states that the right moral action is the one that maximizes utility.
    Deontology opposes utilitarianism and emphasizes moral duties and rules.
    Virtue ethics was developed after utilitarianism and focuses on character rather than actions or consequences.
    """
    
    test_concepts = [
        {"concept": "Ethics", "relevance_score": 1.0},
        {"concept": "Utilitarianism", "relevance_score": 0.9},
        {"concept": "Deontology", "relevance_score": 0.8},
        {"concept": "Virtue Ethics", "relevance_score": 0.7},
        {"concept": "Consequentialism", "relevance_score": 0.6}
    ]
    
    print("Advanced relationship extraction:")
    advanced_rels = extract_advanced_relationships("Ethics", test_text, test_concepts)
    for key, data in advanced_rels.items():
        if isinstance(key, frozenset):
            concepts = list(key)
        else:
            concepts = key
            
        print(f"Relationship between {concepts}")
        print(f"  Type: {data['relationship_type']}")
        print(f"  Count: {data['count']}")
        if "verbs" in data and data["verbs"]:
            print(f"  Common verbs: {data['verbs'].most_common(3)}")
        print()
    
    print("\nPattern-based relationship extraction:")
    pattern_rels = extract_pattern_based_relationships(test_text, [c['concept'] for c in test_concepts])
    for key, data in pattern_rels.items():
        if isinstance(key, frozenset):
            concepts = list(key)
        else:
            concepts = key
            
        print(f"Relationship between {concepts}")
        print(f"  Type: {data['relationship_type']}")
        print(f"  Directed: {data['is_directed']}")
        print(f"  Count: {data['count']}")
        print()
    
    print("\nFull relationship processing:")
    all_rels = process_relationships("Ethics", test_text, test_concepts)
    for key, data in all_rels.items():
        if isinstance(key, frozenset):
            concepts = list(key)
        else:
            concepts = key
            
        print(f"Relationship between {concepts}")
        print(f"  Type: {data['relationship_type']}")
        print(f"  Count: {data['count']}")
        print()