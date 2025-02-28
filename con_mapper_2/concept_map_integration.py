"""
Integration module for the enhanced Philosophical Concept Map Generator.

This module integrates the new concept extraction and relationship analysis
capabilities into the main concept map generation pipeline.
"""
import os
import sys
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Import utility modules
from logging_utils import get_logger, log_execution
from error_handling import handle_error, NLPProcessingError, WikipediaError
from config import config
from performance_utils import memoize, profile

# Import enhanced NLP modules
try:
    from concept_extraction import extract_concepts_advanced, filter_concepts
    from relationship_extraction import process_relationships
    from concept_comparator import ConceptComparator
    
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    # Fall back to original functions if enhanced modules are not available
    ENHANCED_MODULES_AVAILABLE = False
    print(f"Enhanced modules not available: {e}. Using original implementations.")

# Import original functions for fallback
from concept_map_refactored import (
    get_wikipedia_content,
    extract_all_concepts as extract_concepts_basic,
    extract_rich_relationships as extract_relationships_basic,
    generate_summary as generate_summary_basic,
    save_results as save_results_basic,
    create_visualization as create_visualization_basic
)

# Initialize logger
logger = get_logger("concept_map_integration")


@memoize(memory=True, disk=True, key_prefix="wikipedia")
@log_execution()
def get_wiki_content(concept: str) -> str:
    """
    Get Wikipedia content for a concept with caching.
    
    Args:
        concept: The concept to retrieve
        
    Returns:
        Wikipedia article content
    """
    return get_wikipedia_content(concept)


@log_execution()
def extract_concepts(wiki_text: str, concept: str, enhanced: bool = True) -> List[Dict[str, Any]]:
    """
    Extract concepts using either enhanced or basic extraction.
    
    Args:
        wiki_text: The Wikipedia article text
        concept: The primary concept
        enhanced: Whether to use enhanced extraction
        
    Returns:
        List of concept dictionaries
    """
    if not wiki_text:
        return []
    
    if enhanced and ENHANCED_MODULES_AVAILABLE:
        # Use advanced extraction
        logger.info("Using enhanced concept extraction")
        concepts = extract_concepts_advanced(wiki_text, concept)
        
        # Filter concepts based on relevance score
        min_score = config.get("MIN_CONCEPT_SCORE", 0.2)
        max_concepts = config.get("MAX_CONCEPTS", 100)
        filtered = filter_concepts(concepts, min_score=min_score, max_concepts=max_concepts)
        
        return filtered
    else:
        # Fall back to basic extraction
        logger.info("Using basic concept extraction")
        basic_concepts = extract_concepts_basic(wiki_text)
        
        # Convert to dictionary format for compatibility
        return [{"concept": c, "relevance_score": 1.0} for c in basic_concepts]


@log_execution()
def extract_relationships(concept: str, wiki_text: str, concepts: List[Dict[str, Any]], 
                        enhanced: bool = True) -> Dict[Any, Dict[str, Any]]:
    """
    Extract relationships using either enhanced or basic extraction.
    
    Args:
        concept: The primary concept
        wiki_text: The Wikipedia article text
        concepts: List of concept dictionaries
        enhanced: Whether to use enhanced extraction
        
    Returns:
        Dictionary of concept relationships
    """
    if not wiki_text:
        return {}
    
    if enhanced and ENHANCED_MODULES_AVAILABLE:
        # Use advanced relationship extraction
        logger.info("Using enhanced relationship extraction")
        return process_relationships(concept, wiki_text, concepts)
    else:
        # Fall back to basic extraction
        logger.info("Using basic relationship extraction")
        # Convert concepts to list of strings for compatibility
        concept_strings = [c["concept"] for c in concepts]
        return extract_relationships_basic(concept, wiki_text, concept_strings)


@log_execution()
def generate_enhanced_summary(concept: str, wiki_text: str, concepts: List[Dict[str, Any]], 
                            relationships: Dict[Any, Dict[str, Any]]) -> str:
    """
    Generate an enhanced summary of the concept map.
    
    Args:
        concept: The primary concept
        wiki_text: The Wikipedia article text
        concepts: List of concept dictionaries
        relationships: Dictionary of concept relationships
        
    Returns:
        Markdown-formatted summary text
    """
    # Basic information
    concept_count = len(concepts)
    relationship_count = len(relationships)
    
    # Start building the summary
    summary = [
        f"# Concept Map Summary for '{concept}'",
        f"\nThe concept map for '{concept}' contains {concept_count} related philosophical concepts and {relationship_count} relationships.",
    ]
    
    # Add top concepts by relevance
    summary.append("\n## Top Related Concepts by Relevance")
    
    # Sort concepts by relevance score
    top_concepts = sorted(concepts, key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]
    
    for concept_data in top_concepts:
        concept_name = concept_data["concept"]
        relevance = concept_data.get("relevance_score", 0)
        summary.append(f"- **{concept_name}** (relevance: {relevance:.2f})")
    
    # Add relationship types
    summary.append("\n## Relationship Types")
    relationship_types = {}
    
    for rel_key, rel_data in relationships.items():
        rel_type = rel_data.get("relationship_type", "related_to")
        if rel_type not in relationship_types:
            relationship_types[rel_type] = 0
        relationship_types[rel_type] += 1
    
    for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"- **{rel_type}**: {count} relationships")
    
    # Add example relationships
    summary.append("\n## Example Relationships")
    
    # Find relationships with the most interesting types (not just 'mentioned_with')
    interesting_rels = [
        (key, data) for key, data in relationships.items() 
        if data.get("relationship_type", "mentioned_with") != "mentioned_with"
    ]
    
    # Sort by relationship type
    sorted_rels = sorted(interesting_rels, key=lambda x: x[1].get("relationship_type", ""))
    
    # Take up to 5 examples
    for rel_key, rel_data in sorted_rels[:5]:
        # Get the related concept (not the primary concept)
        if isinstance(rel_key, frozenset):
            related_concept = next(iter(c for c in rel_key if c != concept))
        elif isinstance(rel_key, tuple):
            related_concept = rel_key[1] if rel_key[0] == concept else rel_key[0]
        else:
            continue
        
        rel_type = rel_data.get("relationship_type", "related_to")
        is_directed = rel_data.get("is_directed", False)
        direction = rel_data.get("direction", None)
        
        relationship_str = f"**{concept}** {rel_type} **{related_concept}**"
        if is_directed and direction is not None:
            if not direction:  # direction is False, meaning 2 → 1
                relationship_str = f"**{related_concept}** {rel_type} **{concept}**"
        
        summary.append(f"- {relationship_str}")
        
        # Add an example sentence if available
        if "sentences" in rel_data and rel_data["sentences"]:
            example = rel_data["sentences"][0]
            if len(example) > 100:
                example = example[:97] + "..."
            summary.append(f"  - Example: \"{example}\"")
    
    return "\n".join(summary)


@log_execution()
def save_enhanced_results(concept: str, wiki_text: str, concepts: List[Dict[str, Any]], 
                        relationships: Dict[Any, Dict[str, Any]], output_dir: str) -> str:
    """
    Save enhanced results to files.
    
    Args:
        concept: The primary concept
        wiki_text: The Wikipedia article text
        concepts: List of concept dictionaries
        relationships: Dictionary of concept relationships
        output_dir: Directory to save results
        
    Returns:
        Path to the concept directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    concept_dir = os.path.join(output_dir, concept.replace(" ", "_"))
    os.makedirs(concept_dir, exist_ok=True)
    
    # Save Wikipedia text
    with open(os.path.join(concept_dir, "wiki_text.txt"), 'w', encoding='utf-8') as f:
        f.write(wiki_text)
    
    # Save extracted concepts
    with open(os.path.join(concept_dir, "extracted_concepts.json"), 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2)
    
    # Save relationship data (converting frozenset keys to strings)
    serializable_relationship_data = {}
    for key, value in relationships.items():
        # Convert frozenset or tuple to a string representation for JSON serialization
        if isinstance(key, frozenset):
            string_key = str(sorted(list(key)))
        elif isinstance(key, tuple):
            string_key = str(list(key))
        else:
            string_key = str(key)
            
        serializable_relationship_data[string_key] = value
    
    with open(os.path.join(concept_dir, "relationships.json"), 'w', encoding='utf-8') as f:
        json.dump(serializable_relationship_data, f, indent=2)
    
    # Generate and save summary
    try:
        summary = generate_enhanced_summary(concept, wiki_text, concepts, relationships)
        with open(os.path.join(concept_dir, "summary.md"), 'w', encoding='utf-8') as f:
            f.write(summary)
    except Exception as e:
        logger.warning(f"Could not generate summary: {e}")
    
    logger.info(f"All results saved to {concept_dir}")
    return concept_dir


@profile
@log_execution()
def create_enhanced_visualization(concept: str, relationships: Dict[Any, Dict[str, Any]], 
                                concepts: List[Dict[str, Any]], threshold: float = 1.0) -> plt.Figure:
    """
    Create an enhanced visualization of the concept map.
    
    Args:
        concept: The primary concept
        relationships: Dictionary of concept relationships
        concepts: List of concept dictionaries
        threshold: Minimum relevance threshold for concepts
        
    Returns:
        matplotlib Figure object
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    logger.info(f"Creating enhanced visualization for '{concept}' with threshold {threshold}")
    
    # Create graph
    G = nx.Graph()
    G.add_node(concept, type="main", relevance=1.0)
    
    # Filter concepts by relevance threshold
    filtered_concepts = [c for c in concepts if c.get("relevance_score", 0) >= threshold]
    
    # Sort by relevance and limit to reasonable number
    top_concepts = sorted(filtered_concepts, key=lambda x: x.get("relevance_score", 0), reverse=True)
    max_display = config.get("MAX_CONCEPTS_DISPLAY", 15)
    top_concepts = top_concepts[:max_display]
    
    # Map concept to relevance for easy lookup
    concept_relevance = {c["concept"]: c.get("relevance_score", 0) for c in top_concepts}
    
    # Add concepts as nodes
    for concept_data in top_concepts:
        concept_name = concept_data["concept"]
        relevance = concept_data.get("relevance_score", 0)
        
        G.add_node(concept_name, relevance=relevance, type="related")
    
    # Add edges based on relationships
    for rel_key, rel_data in relationships.items():
        # Extract the two concepts in the relationship
        if isinstance(rel_key, frozenset):
            concepts_in_rel = list(rel_key)
        elif isinstance(rel_key, tuple):
            concepts_in_rel = list(rel_key)
        else:
            # Try to parse from string representation
            try:
                concepts_in_rel = eval(rel_key)
                if not isinstance(concepts_in_rel, (list, tuple)):
                    continue
            except:
                continue
        
        # Check if both concepts are in the graph
        if concepts_in_rel[0] in G and concepts_in_rel[1] in G:
            # Get relationship properties
            rel_type = rel_data.get("relationship_type", "related_to")
            count = rel_data.get("count", 1)
            is_directed = rel_data.get("is_directed", False)
            
            # Create the edge
            G.add_edge(concepts_in_rel[0], concepts_in_rel[1], 
                     type=rel_type, 
                     weight=count, 
                     directed=is_directed)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create a layout that puts the main concept in the center
    pos = nx.spring_layout(G, seed=42)
    pos[concept] = (0.5, 0.5)  # Force the main concept to be centered
    
    # Create lists of nodes by type
    main_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "main"]
    related_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "related"]
    
    # Size nodes by relevance
    node_sizes = []
    for node in related_nodes:
        relevance = G.nodes[node].get("relevance", 0.5)
        size = 500 + (2500 * relevance)  # Scale from 500 to 3000 based on relevance
        node_sizes.append(size)
    
    # Create a colormap for edge types
    edge_colors = []
    edge_widths = []
    for u, v, attr in G.edges(data=True):
        rel_type = attr.get("type", "related_to")
        weight = attr.get("weight", 1)
        
        # Map relationship types to colors
        if rel_type == "is_a" or rel_type == "part_of":
            edge_colors.append("green")
        elif rel_type == "opposes" or rel_type == "contradicts":
            edge_colors.append("red")
        elif rel_type in ["causes", "leads_to", "influences"]:
            edge_colors.append("blue")
        elif rel_type in ["similar_to", "agrees_with"]:
            edge_colors.append("purple")
        else:
            edge_colors.append("gray")
        
        # Scale edge width based on weight
        edge_widths.append(1 + weight * 0.5)
    
    # Draw the graph
    # Draw the main node (center concept)
    nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, 
                          node_color="gold", 
                          node_size=3000, 
                          alpha=0.9)
    
    # Draw the related nodes with sizes based on relevance
    node_collection = nx.draw_networkx_nodes(G, pos, nodelist=related_nodes, 
                                           node_color="skyblue", 
                                           node_size=node_sizes, 
                                           alpha=0.7)
    
    # Add a colorbar showing relevance scores
    relevance_values = [G.nodes[n].get("relevance", 0) for n in related_nodes]
    if relevance_values:
        # Map the relevance values to colors
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        norm = mcolors.Normalize(min(relevance_values), max(relevance_values))
        sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
        sm.set_array([])
        
        # Map relevance to actual colors
        if node_collection is not None:
            node_collection.set_array(relevance_values)
            node_collection.set_cmap(cm.Blues)
            node_collection.set_norm(norm)
            
            # Add colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Relevance Score')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          edge_color=edge_colors, 
                          alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add title and disable axis
    plt.title(f"Enhanced Concept Map for '{concept}'")
    plt.axis("off")
    
    return fig


@log_execution()
def process_concept_enhanced(concept: str, cache_dir: str = None, output_dir: str = None, 
                           threshold: float = None, use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Process a philosophical concept using the enhanced pipeline.
    
    Args:
        concept: The philosophical concept to process
        cache_dir: Directory for caching Wikipedia content
        output_dir: Directory to save results
        threshold: Minimum relevance threshold for concepts in visualization
        use_enhanced: Whether to use enhanced extraction and analysis
        
    Returns:
        Dictionary with processing results
    """
    # Use default directories if not specified
    if cache_dir is None:
        cache_dir = config.get("CACHE_DIR", "wiki_cache")
    
    if output_dir is None:
        output_dir = config.get("OUTPUT_DIR", "output")
    
    if threshold is None:
        threshold = config.get("DEFAULT_THRESHOLD", 1.0)
    
    logger.info(f"Processing concept '{concept}' with threshold {threshold}")
    
    try:
        # Step 1: Get Wikipedia content
        wiki_text = get_wiki_content(concept)
        if not wiki_text:
            error_msg = f"Could not retrieve Wikipedia content for '{concept}'"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Step 2: Extract concepts
        extracted_concepts = extract_concepts(wiki_text, concept, enhanced=use_enhanced)
        logger.info(f"Extracted {len(extracted_concepts)} concepts for '{concept}'")
        
        # Step 3: Extract relationships
        relationships = extract_relationships(concept, wiki_text, extracted_concepts, enhanced=use_enhanced)
        logger.info(f"Extracted {len(relationships)} relationships for '{concept}'")
        
        # Step 4: Save results
        result_dir = save_enhanced_results(concept, wiki_text, extracted_concepts, relationships, output_dir)
        
        # Step 5: Create visualization
        visualization = create_enhanced_visualization(concept, relationships, extracted_concepts, threshold)
        
        # Save visualization
        viz_path = os.path.join(result_dir, f"{concept.replace(' ', '_')}_concept_map.png")
        visualization.savefig(viz_path)
        logger.info(f"Visualization saved to {viz_path}")
        
        return {
            "concept": concept,
            "wiki_text": wiki_text,
            "extracted_concepts": extracted_concepts,
            "relationships": relationships,
            "result_directory": result_dir,
            "visualization": visualization
        }
    
    except Exception as e:
        error_msg = f"Error processing concept '{concept}': {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {"error": error_msg}


@log_execution()
def compare_concepts(concept_names: List[str], input_dir: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Compare multiple philosophical concepts.
    
    Args:
        concept_names: List of concepts to compare
        input_dir: Directory containing concept data
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    if not ENHANCED_MODULES_AVAILABLE:
        return {"error": "Enhanced modules are required for concept comparison"}
    
    # Use default directories if not specified
    if input_dir is None:
        input_dir = config.get("OUTPUT_DIR", "output")
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "comparison")
    
    logger.info(f"Comparing concepts: {', '.join(concept_names)}")
    
    try:
        # Create comparator
        comparator = ConceptComparator()
        
        # Load concepts
        for concept in concept_names:
            result = comparator.load_concept_from_files(concept, input_dir)
            if "error" in result:
                return {"error": f"Error loading concept '{concept}': {result['error']}"}
        
        # Compare concepts
        comparison = comparator.compare_concepts(concept_names)
        if "error" in comparison:
            return {"error": f"Error comparing concepts: {comparison['error']}"}
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        with open(os.path.join(output_dir, "comparison_results.json"), 'w', encoding='utf-8') as f:
            # Convert non-serializable objects to serializable formats
            serializable_results = comparison.copy()
            serializable_results.pop("comparison_graph", None)  # Remove the NetworkX graph
            
            # Convert frozensets to lists
            if "shared_relationships" in serializable_results:
                for rel in serializable_results["shared_relationships"]:
                    for appearance in rel["appearances"]:
                        if "key" in appearance and isinstance(appearance["key"], frozenset):
                            appearance["key"] = sorted(list(appearance["key"]))
            
            json.dump(serializable_results, f, indent=2)
        
        # Save visualization
        visualization_path = os.path.join(output_dir, "comparison_visualization.png")
        visualization = comparator.visualize_comparison(visualization_path)
        
        # Save summary
        summary = comparator.generate_comparison_summary()
        with open(os.path.join(output_dir, "comparison_summary.md"), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Augment the results with additional data
        comparison["result_directory"] = output_dir
        comparison["visualization"] = visualization
        comparison["summary"] = summary
        
        return comparison
    
    except Exception as e:
        error_msg = f"Error comparing concepts: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {"error": error_msg}


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Philosophical Concept Map Generator")
    
    # Main arguments
    parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
    parser.add_argument("--threshold", type=float, default=None, help="Relevance threshold for filtering concepts")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--basic", action="store_true", help="Use basic extraction (no enhanced features)")
    
    # Comparison mode
    parser.add_argument("--compare", nargs='+', help="List of concepts to compare")
    
    args = parser.parse_args()
    
    # Comparison mode
    if args.compare:
        result = compare_concepts(args.compare, output_dir=args.output)
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
            
        print(f"Comparison complete. Results saved to {result['result_directory']}")
        return 0
    
    # Regular concept processing mode
    concept = args.concept
    if not concept:
        concept = input("Enter a philosophical concept: ")
    
    result = process_concept_enhanced(
        concept,
        cache_dir=args.cache_dir,
        output_dir=args.output,
        threshold=args.threshold,
        use_enhanced=not args.basic
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    print(f"Processing complete. Results saved to {result['result_directory']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())