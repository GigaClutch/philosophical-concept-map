"""
Philosophical Concept Map Generator - Core Module

This module contains the main functionality for extracting philosophical concepts,
analyzing their relationships, and creating visual concept maps.
"""
import os
import sys
import json
import time
import wikipedia
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
import tempfile

# Try to import utility modules
try:
    from error_handling import (
        handle_error, 
        WikipediaError, 
        NLPProcessingError, 
        VisualizationError, 
        DataStorageError
    )
    from logging_utils import get_logger, log_execution, log_to_ui
    from config import config
    
    # Initialize logger
    logger = get_logger("concept_map")
    
    # Use configuration settings
    DEFAULT_CACHE_DIR = config.get("CACHE_DIR", "wiki_cache")
    DEFAULT_OUTPUT_DIR = config.get("OUTPUT_DIR", "output")
    DEFAULT_THRESHOLD = config.get("DEFAULT_THRESHOLD", 1.0)
    MAX_TEXT_LENGTH = config.get("MAX_TEXT_LENGTH", 20000)
    MAX_CONCEPTS_DISPLAY = config.get("MAX_CONCEPTS_DISPLAY", 15) 
    
except ImportError:
    # Fallback for standalone usage without utility modules
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("concept_map")
    
    def handle_error(error, raise_exception=False, **kwargs):
        logger.error(f"Error: {error}")
    
    def log_execution(logger=None):
        def decorator(func):
            return func
        return decorator
    
    def log_to_ui(message, level=None):
        print(f"UI: {message}")
    
    # Default configuration
    DEFAULT_CACHE_DIR = "wiki_cache"
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_THRESHOLD = 1.0
    MAX_TEXT_LENGTH = 20000
    MAX_CONCEPTS_DISPLAY = 15
    
    # Define exception classes
    class WikipediaError(Exception): pass
    class NLPProcessingError(Exception): pass
    class VisualizationError(Exception): pass
    class DataStorageError(Exception): pass

# Initialize NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    error_msg = f"Error loading spaCy model: {e}"
    logger.error(error_msg)
    logger.error("Please install it with: python -m spacy download en_core_web_sm")
    raise NLPProcessingError(error_msg)

# Core philosophical terms to detect
PHILOSOPHICAL_TERMS = [
    "Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", 
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology", 
    "Determinism", "Free Will", "Consciousness", "Virtue Ethics", 
    "Deontology", "Utilitarianism", "Moral Realism", "Relativism",
    "Ontology", "Dualism", "Materialism", "Idealism", "Pragmatism",
    "Positivism", "Skepticism", "Nihilism", "Subjectivism", "Objectivism"
]


@log_execution()
def get_wikipedia_content(concept_name, cache_dir=DEFAULT_CACHE_DIR):
    """
    Fetches Wikipedia content with caching to reduce API calls.
    
    Args:
        concept_name: The philosophical concept to retrieve
        cache_dir: Directory for caching Wikipedia content
        
    Returns:
        str: Wikipedia article content or None if retrieval failed
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{concept_name.replace(' ', '_')}.txt")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        logger.info(f"Loading cached content for '{concept_name}'")
        log_to_ui(f"Loading cached content for '{concept_name}'")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
            # Continue to fetch from Wikipedia
    
    # Fetch from Wikipedia
    try:
        logger.info(f"Fetching Wikipedia content for '{concept_name}'")
        log_to_ui(f"Fetching Wikipedia content for '{concept_name}'")
        
        # Use auto_suggest=False to prevent Wikipedia from changing our search term
        page = wikipedia.page(concept_name, auto_suggest=False)
        content = page.content
        
        logger.info(f"Successfully retrieved page: {page.title} ({len(content)} characters)")
        log_to_ui(f"Retrieved Wikipedia page ({len(content)} characters)")
        
        # Cache the content
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Error caching content: {e}")
            # Continue even if caching fails
        
        return content
        
    except wikipedia.exceptions.PageError as e:
        error_msg = f"Wikipedia page not found for '{concept_name}'"
        logger.error(error_msg)
        log_to_ui(error_msg)
        return None
        
    except wikipedia.exceptions.DisambiguationError as e:
        options = ', '.join(e.options[:5])
        error_msg = f"Disambiguation error for '{concept_name}'. Options: {options}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        return None
        
    except Exception as e:
        error_msg = f"Error retrieving Wikipedia content for '{concept_name}': {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        return None


@log_execution()
def extract_all_concepts(wiki_text):
    """
    Extracts concepts using both NER and a predefined list of philosophical terms.
    
    Args:
        wiki_text: The Wikipedia article text
        
    Returns:
        list: Extracted philosophical concepts
    """
    if not wiki_text:
        logger.warning("Cannot extract concepts from empty text")
        return []
    
    logger.info("Extracting concepts from text...")
    log_to_ui("Extracting concepts from text...")
    
    try:
        # NER extraction - limit text for faster processing
        processing_text = wiki_text[:MAX_TEXT_LENGTH]
        doc = nlp(processing_text)
        
        ner_concepts = set()
        relevant_entity_types = ["ORG", "PERSON", "NORP", "MISC", "GPE"]
        
        for ent in doc.ents:
            if ent.label_ in relevant_entity_types:
                ner_concepts.add(ent.text)
        
        logger.info(f"Extracted {len(ner_concepts)} concepts using NER")
        
        # Keyword-based extraction
        keyword_concepts = set()
        text_lower = wiki_text.lower()
        for term in PHILOSOPHICAL_TERMS:
            if term.lower() in text_lower:
                keyword_concepts.add(term)
        
        logger.info(f"Extracted {len(keyword_concepts)} concepts using keyword matching")
        
        # Combine both approaches
        all_concepts = sorted(list(ner_concepts.union(keyword_concepts)))
        logger.info(f"Total unique concepts extracted: {len(all_concepts)}")
        log_to_ui(f"Found {len(all_concepts)} related concepts")
        
        return all_concepts
        
    except Exception as e:
        error_msg = f"Error extracting concepts: {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        raise NLPProcessingError(error_msg)


@log_execution()
def extract_rich_relationships(input_concept, wiki_text, extracted_concepts):
    """
    Extracts relationships with richer semantic content.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        
    Returns:
        dict: Dictionary of concept relationships
    """
    if not wiki_text:
        logger.warning("Cannot extract relationships from empty text")
        return {}
    
    logger.info(f"Analyzing relationships for '{input_concept}'...")
    log_to_ui(f"Analyzing relationships between concepts...")
    
    try:
        # Process limited text for efficiency
        processing_text = wiki_text[:MAX_TEXT_LENGTH]
        doc = nlp(processing_text)
        sentences = list(doc.sents)
        
        relationship_data = {}
        input_concept_lower = input_concept.lower()
        
        for sentence in sentences:
            sentence_text = sentence.text.lower()
            
            # Check if input concept is mentioned in this sentence
            if input_concept_lower in sentence_text:
                # Find concepts that co-occur in this sentence
                for concept in extracted_concepts:
                    concept_lower = concept.lower()
                    
                    if concept_lower != input_concept_lower and concept_lower in sentence_text:
                        # Create a unique key for this concept pair
                        pair_key = frozenset({input_concept, concept})
                        
                        # Initialize data structure if this is the first occurrence
                        if pair_key not in relationship_data:
                            relationship_data[pair_key] = {
                                "count": 0,
                                "sentences": []
                            }
                        
                        # Update relationship data
                        relationship_data[pair_key]["count"] += 1
                        relationship_data[pair_key]["sentences"].append(sentence.text)
        
        logger.info(f"Found {len(relationship_data)} relationships")
        log_to_ui(f"Found {len(relationship_data)} relationships")
        
        return relationship_data
        
    except Exception as e:
        error_msg = f"Error extracting relationships: {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        raise NLPProcessingError(error_msg)


@log_execution()
def generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data):
    """
    Generate a textual summary of the concept map.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        relationship_data: Dictionary of concept relationships
        
    Returns:
        str: Markdown-formatted summary text
    """
    if not wiki_text or not extracted_concepts or not relationship_data:
        logger.warning("Insufficient data to generate a summary")
        return "Insufficient data to generate a summary."
    
    logger.info(f"Generating summary for '{input_concept}'")
    log_to_ui(f"Generating summary for '{input_concept}'")
    
    try:
        # Basic statistics
        concept_count = len(extracted_concepts)
        relationship_count = len(relationship_data)
        
        # Find most related concepts
        concept_relevance = {}
        for pair, data in relationship_data.items():
            pair_list = list(pair)
            if pair_list[0] == input_concept:
                concept_relevance[pair_list[1]] = concept_relevance.get(pair_list[1], 0) + data["count"]
            elif pair_list[1] == input_concept:
                concept_relevance[pair_list[0]] = concept_relevance.get(pair_list[0], 0) + data["count"]
        
        # Sort concepts by relevance
        top_concepts = sorted(concept_relevance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate summary text
        summary = [
            f"# Concept Map Summary for '{input_concept}'",
            f"\nThe concept map for '{input_concept}' contains {concept_count} related philosophical concepts and {relationship_count} relationships.",
            "\n## Most Closely Related Concepts:"
        ]
        
        for concept, count in top_concepts:
            # Find the relationship data
            rel_data = relationship_data.get(frozenset({input_concept, concept}))
            if rel_data:
                summary.append(f"- **{concept}** (mentioned together {count} times)")
                
                # Include a sample sentence
                if rel_data["sentences"]:
                    example = rel_data["sentences"][0]
                    if len(example) > 100:
                        example = example[:97] + "..."
                    summary.append(f"  - Example: \"{example}\"")
        
        return "\n".join(summary)
        
    except Exception as e:
        error_msg = f"Error generating summary: {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        return f"Error generating summary: {str(e)}"


@log_execution()
def save_results(input_concept, wiki_text, extracted_concepts, relationship_data, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Save all results to files for future reference.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        relationship_data: Dictionary of concept relationships
        output_dir: Directory to save results
        
    Returns:
        str: Path to the concept directory
    """
    logger.info(f"Saving results for '{input_concept}'")
    log_to_ui(f"Saving results for '{input_concept}'")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        concept_dir = os.path.join(output_dir, input_concept.replace(" ", "_"))
        os.makedirs(concept_dir, exist_ok=True)
        
        # Save Wikipedia text
        with open(os.path.join(concept_dir, "wiki_text.txt"), 'w', encoding='utf-8') as f:
            f.write(wiki_text)
        
        # Save extracted concepts
        with open(os.path.join(concept_dir, "extracted_concepts.json"), 'w', encoding='utf-8') as f:
            json.dump(extracted_concepts, f, indent=2)
        
        # Save relationship data (converting frozenset keys to strings)
        serializable_relationship_data = {}
        for key, value in relationship_data.items():
            # Convert frozenset to a string representation for JSON serialization
            string_key = str(sorted(list(key)))
            serializable_relationship_data[string_key] = value
        
        with open(os.path.join(concept_dir, "relationships.json"), 'w', encoding='utf-8') as f:
            json.dump(serializable_relationship_data, f, indent=2)
        
        # Generate and save summary
        try:
            summary = generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data)
            with open(os.path.join(concept_dir, "summary.md"), 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception as e:
            logger.warning(f"Could not generate summary: {e}")
        
        logger.info(f"All results saved to {concept_dir}")
        log_to_ui(f"Results saved to {concept_dir}")
        
        return concept_dir
        
    except Exception as e:
        error_msg = f"Error saving results: {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        raise DataStorageError(error_msg)


@log_execution()
def create_visualization(concept, relationship_data, extracted_concepts, threshold=DEFAULT_THRESHOLD):
    """
    Create a visualization of the concept map.
    
    Args:
        concept: The primary philosophical concept
        relationship_data: Dictionary of concept relationships
        extracted_concepts: List of extracted concepts
        threshold: Minimum relevance score for concepts to include
        
    Returns:
        matplotlib.figure.Figure: The generated visualization figure
    """
    logger.info(f"Creating visualization for '{concept}' with threshold {threshold}")
    log_to_ui(f"Creating visualization with threshold {threshold:.1f}")
    
    try:
        # Create graph
        G = nx.Graph()
        G.add_node(concept)
        
        # Calculate relevance scores for concepts
        concept_relevance = {}
        for pair_key, data in relationship_data.items():
            pair_list = list(pair_key)
            if pair_list[0] == concept:
                concept_relevance[pair_list[1]] = data["count"]
            elif pair_list[1] == concept:
                concept_relevance[pair_list[0]] = data["count"]
        
        # Filter concepts by threshold and limit to top concepts for clarity
        filtered_concepts = [(c, score) for c, score in concept_relevance.items() if score >= threshold]
        top_concepts = sorted(filtered_concepts, key=lambda x: x[1], reverse=True)[:MAX_CONCEPTS_DISPLAY]
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Add nodes and edges to graph
        for related_concept, score in top_concepts:
            G.add_node(related_concept)
            G.add_edge(concept, related_concept, weight=score)
        
        # Generate layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color="skyblue", 
                              node_size=3000, 
                              alpha=0.8)
        
        # Draw edges with width based on weight
        if G.edges():
            edge_weights = [G[u][v]['weight']/2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, 
                                 width=edge_weights, 
                                 alpha=0.5, 
                                 edge_color="gray")
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f"Concept Map for '{concept}'")
        plt.axis('off')
        
        # Save visualization to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), f"{concept.replace(' ', '_')}_concept_map.png")
        plt.savefig(temp_file)
        logger.info(f"Visualization saved to temporary file: {temp_file}")
        
        log_to_ui(f"Visualization created successfully")
        return fig
        
    except Exception as e:
        error_msg = f"Error creating visualization: {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        raise VisualizationError(error_msg)


@log_execution()
def process_concept(concept_name, cache_dir=DEFAULT_CACHE_DIR, output_dir=DEFAULT_OUTPUT_DIR, threshold=DEFAULT_THRESHOLD):
    """
    Process a philosophical concept through the entire pipeline.
    
    Args:
        concept_name: The philosophical concept to process
        cache_dir: Directory for caching Wikipedia content
        output_dir: Directory to save results
        threshold: Minimum relevance score for concepts in visualization
        
    Returns:
        dict: Results including paths to saved files and visualization data
    """
    logger.info(f"Processing concept: '{concept_name}'")
    log_to_ui(f"Processing concept: '{concept_name}'")
    
    try:
        # Get Wikipedia content
        wiki_text = get_wikipedia_content(concept_name, cache_dir=cache_dir)
        if not wiki_text:
            error_msg = f"Could not retrieve Wikipedia content for '{concept_name}'"
            logger.error(error_msg)
            log_to_ui(error_msg)
            return {"error": error_msg}
        
        # Extract concepts
        extracted_concepts = extract_all_concepts(wiki_text)
        
        # Extract relationships
        relationship_data = extract_rich_relationships(concept_name, wiki_text, extracted_concepts)
        
        # Save results
        result_dir = save_results(concept_name, wiki_text, extracted_concepts, relationship_data, output_dir)
        
        # Create visualization
        visualization = create_visualization(concept_name, relationship_data, extracted_concepts, threshold)
        
        # Return results
        return {
            "concept": concept_name,
            "result_directory": result_dir,
            "concept_count": len(extracted_concepts),
            "relationship_count": len(relationship_data),
            "visualization": visualization
        }
        
    except Exception as e:
        error_msg = f"Error processing concept '{concept_name}': {e}"
        logger.error(error_msg)
        log_to_ui(error_msg)
        return {"error": error_msg}


def main():
    """Main function for command-line usage"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate philosophical concept maps")
    parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Relevance threshold for filtering concepts")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get concept from argument or prompt
    concept = args.concept
    if not concept:
        concept = input("Enter a philosophical concept: ")
    
    logger.info(f"Generating concept map for: '{concept}'...")
    print(f"Generating concept map for: '{concept}'...")
    
    # Process the concept
    results = process_concept(concept, args.cache_dir, args.output, args.threshold)
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print("Concept map generation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())