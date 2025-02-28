"""
Concept Comparator module for the Philosophical Concept Map Generator.

This module provides functionality for comparing multiple philosophical concepts
to identify relationships, similarities, differences, and common themes between them.
"""
import os
import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import utility modules
try:
    from logging_utils import get_logger, log_execution
    from error_handling import handle_error, NLPProcessingError
    from config import config
    from concept_extraction import extract_concepts_advanced
    from relationship_extraction import process_relationships
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
    
    # Placeholder functions
    def extract_concepts_advanced(text, primary_concept=None):
        return [{"concept": "Concept1"}, {"concept": "Concept2"}]
    
    def process_relationships(input_concept, wiki_text, extracted_concepts):
        return {}

# Initialize logger
logger = get_logger("concept_comparator")


class ConceptComparator:
    """
    Analyzes and compares multiple philosophical concepts to identify relationships,
    similarities, differences, and common themes.
    """
    
    def __init__(self):
        """Initialize the concept comparator."""
        self.concept_data = {}
        self.comparison_results = None
    
    @log_execution()
    def add_concept(self, concept_name: str, wiki_text: str) -> Dict[str, Any]:
        """
        Add a concept to the comparator.
        
        Args:
            concept_name: Name of the concept
            wiki_text: Wikipedia text content for the concept
            
        Returns:
            Dictionary with processed concept data
        """
        logger.info(f"Adding concept '{concept_name}' to comparator")
        
        try:
            # Extract concepts with advanced extraction
            extracted_concepts = extract_concepts_advanced(wiki_text, concept_name)
            logger.info(f"Extracted {len(extracted_concepts)} related concepts for '{concept_name}'")
            
            # Extract relationships
            relationships = process_relationships(concept_name, wiki_text, extracted_concepts)
            logger.info(f"Extracted {len(relationships)} relationships for '{concept_name}'")
            
            # Store concept data
            self.concept_data[concept_name] = {
                "wiki_text": wiki_text,
                "extracted_concepts": extracted_concepts,
                "relationships": relationships
            }
            
            return self.concept_data[concept_name]
            
        except Exception as e:
            error_msg = f"Error adding concept '{concept_name}': {e}"
            logger.error(error_msg)
            handle_error(error_msg)
            return {"error": error_msg}
    
    def load_concept_from_files(self, concept_name: str, directory: str) -> Dict[str, Any]:
        """
        Load a concept from saved files.
        
        Args:
            concept_name: Name of the concept
            directory: Directory containing the saved files
            
        Returns:
            Dictionary with loaded concept data
        """
        logger.info(f"Loading concept '{concept_name}' from {directory}")
        
        try:
            concept_dir = os.path.join(directory, concept_name.replace(" ", "_"))
            
            # Check if directory exists
            if not os.path.exists(concept_dir):
                error_msg = f"Directory not found for concept '{concept_name}'"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Load wiki text
            wiki_text_path = os.path.join(concept_dir, "wiki_text.txt")
            if not os.path.exists(wiki_text_path):
                error_msg = f"Wiki text file not found for concept '{concept_name}'"
                logger.error(error_msg)
                return {"error": error_msg}
                
            with open(wiki_text_path, 'r', encoding='utf-8') as f:
                wiki_text = f.read()
            
            # Load extracted concepts
            concepts_path = os.path.join(concept_dir, "extracted_concepts.json")
            if not os.path.exists(concepts_path):
                # If file doesn't exist, extract concepts from wiki text
                extracted_concepts = extract_concepts_advanced(wiki_text, concept_name)
            else:
                with open(concepts_path, 'r', encoding='utf-8') as f:
                    concepts_data = json.load(f)
                    
                    # Check if it's the new format (list of dicts) or old format (list of strings)
                    if concepts_data and isinstance(concepts_data[0], str):
                        # Convert old format to new format
                        extracted_concepts = [{"concept": c, "relevance_score": 1.0} for c in concepts_data]
                    else:
                        extracted_concepts = concepts_data
            
            # Load relationships
            relationships_path = os.path.join(concept_dir, "relationships.json")
            if not os.path.exists(relationships_path):
                # If file doesn't exist, extract relationships from wiki text
                relationships = process_relationships(concept_name, wiki_text, extracted_concepts)
            else:
                with open(relationships_path, 'r', encoding='utf-8') as f:
                    relationships_data = json.load(f)
                    
                    # Convert string keys (from JSON) back to proper format
                    relationships = {}
                    for key_str, value in relationships_data.items():
                        # Check if it's a frozenset (list) or tuple (directional relationship)
                        try:
                            # Try to parse as a list (for frozenset)
                            key_list = eval(key_str)
                            if len(key_list) == 2:
                                key = frozenset(key_list)
                                relationships[key] = value
                        except:
                            # If parsing fails, just use the string key (for backward compatibility)
                            relationships[key_str] = value
            
            # Store concept data
            self.concept_data[concept_name] = {
                "wiki_text": wiki_text,
                "extracted_concepts": extracted_concepts,
                "relationships": relationships
            }
            
            return self.concept_data[concept_name]
            
        except Exception as e:
            error_msg = f"Error loading concept '{concept_name}': {e}"
            logger.error(error_msg)
            handle_error(error_msg)
            return {"error": error_msg}
    
    @log_execution()
    def compare_concepts(self, concept_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple philosophical concepts.
        
        Args:
            concept_names: List of concept names to compare (uses all loaded concepts if None)
            
        Returns:
            Dictionary with comparison results
        """
        # Use all loaded concepts if none specified
        if concept_names is None:
            concept_names = list(self.concept_data.keys())
        
        # Ensure we have at least two concepts
        if len(concept_names) < 2:
            error_msg = "Need at least two concepts to compare"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Ensure all concepts are loaded
        missing_concepts = [c for c in concept_names if c not in self.concept_data]
        if missing_concepts:
            error_msg = f"Concepts not loaded: {', '.join(missing_concepts)}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        logger.info(f"Comparing concepts: {', '.join(concept_names)}")
        
        try:
            # Extract data for comparison
            wiki_texts = {c: self.concept_data[c]["wiki_text"] for c in concept_names}
            concept_sets = {c: [item["concept"] for item in self.concept_data[c]["extracted_concepts"]] 
                          for c in concept_names}
            relationships = {c: self.concept_data[c]["relationships"] for c in concept_names}
            
            # Compute similarity matrix
            similarity_matrix = self._compute_concept_similarity(wiki_texts)
            
            # Find common and unique concepts
            common_concepts = self._find_common_concepts(concept_sets)
            unique_concepts = self._find_unique_concepts(concept_sets)
            
            # Identify key themes
            key_themes = self._identify_key_themes(wiki_texts)
            
            # Find shared relationships
            shared_relationships = self._find_shared_relationships(relationships, concept_names)
            
            # Generate comparison graph
            comparison_graph = self._generate_comparison_graph(concept_names, concept_sets, similarity_matrix, shared_relationships)
            
            # Store results
            self.comparison_results = {
                "concepts": concept_names,
                "similarity_matrix": similarity_matrix,
                "common_concepts": common_concepts,
                "unique_concepts": unique_concepts,
                "key_themes": key_themes,
                "shared_relationships": shared_relationships,
                "comparison_graph": comparison_graph
            }
            
            return self.comparison_results
            
        except Exception as e:
            error_msg = f"Error comparing concepts: {e}"
            logger.error(error_msg)
            handle_error(error_msg)
            return {"error": error_msg}
    
    def _compute_concept_similarity(self, wiki_texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Compute similarity between concepts based on text content.
        
        Args:
            wiki_texts: Dictionary mapping concept names to wiki text content
            
        Returns:
            Dictionary mapping each concept to its similarity scores with other concepts
        """
        concepts = list(wiki_texts.keys())
        texts = [wiki_texts[c] for c in concepts]
        
        # Compute TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix)
            
            # Convert to dictionary format
            similarity_dict = {}
            for i, c1 in enumerate(concepts):
                similarity_dict[c1] = {}
                for j, c2 in enumerate(concepts):
                    similarity_dict[c1][c2] = float(similarity[i, j])
            
            return similarity_dict
        except:
            # Fallback to simple Jaccard similarity if TF-IDF fails
            return self._compute_concept_overlap(wiki_texts)
    
    def _compute_concept_overlap(self, wiki_texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Compute similarity based on concept overlap.
        
        Args:
            wiki_texts: Dictionary mapping concept names to wiki text content
            
        Returns:
            Dictionary mapping each concept to its similarity scores with other concepts
        """
        concepts = list(wiki_texts.keys())
        similarity_dict = {}
        
        # Convert texts to word sets
        word_sets = {}
        for concept, text in wiki_texts.items():
            # Split into words and remove punctuation
            words = set(word.strip('.,;:?!()[]{}"\'-').lower() 
                       for word in text.split() 
                       if len(word.strip('.,;:?!()[]{}"\'-')) > 3)
            word_sets[concept] = words
        
        # Calculate Jaccard similarity for each pair
        for i, c1 in enumerate(concepts):
            similarity_dict[c1] = {}
            set1 = word_sets[c1]
            
            for j, c2 in enumerate(concepts):
                set2 = word_sets[c2]
                
                # Jaccard similarity: size of intersection divided by size of union
                if set1 or set2:  # Avoid division by zero
                    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                else:
                    similarity = 0
                    
                similarity_dict[c1][c2] = similarity
        
        return similarity_dict
    
    def _find_common_concepts(self, concept_sets: Dict[str, List[str]]) -> List[str]:
        """
        Find concepts that appear across multiple source concepts.
        
        Args:
            concept_sets: Dictionary mapping concept names to their related concepts
            
        Returns:
            List of concepts common to all source concepts
        """
        all_sets = list(concept_sets.values())
        if not all_sets:
            return []
        
        # Start with the first set and find intersection with others
        common = set(all_sets[0])
        for concept_set in all_sets[1:]:
            common = common.intersection(set(concept_set))
        
        return sorted(list(common))
    
    def _find_unique_concepts(self, concept_sets: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Find concepts unique to each source concept.
        
        Args:
            concept_sets: Dictionary mapping concept names to their related concepts
            
        Returns:
            Dictionary mapping each concept to its unique concepts
        """
        concepts = list(concept_sets.keys())
        unique_dict = {}
        
        for c in concepts:
            # Find concepts that only appear in this source
            current_set = set(concept_sets[c])
            other_sets = set()
            
            for other_c in concepts:
                if other_c != c:
                    other_sets.update(set(concept_sets[other_c]))
            
            unique_dict[c] = sorted(list(current_set - other_sets))
        
        return unique_dict
    
    def _identify_key_themes(self, wiki_texts: Dict[str, str]) -> List[Tuple[str, int]]:
        """
        Identify key themes across concepts by analyzing word frequency.
        
        Args:
            wiki_texts: Dictionary mapping concept names to wiki text content
            
        Returns:
            List of (word, frequency) tuples for key themes
        """
        # Extract all words from all texts
        all_words = []
        for text in wiki_texts.values():
            # Simple word extraction (could be enhanced with proper NLP)
            words = [word.strip('.,;:?!()[]{}"\'-').lower() 
                    for word in text.split() 
                    if len(word.strip('.,;:?!()[]{}"\'-')) > 3]
            
            # Filter out common words
            common_words = {'this', 'that', 'these', 'those', 'there', 'their', 'they', 
                          'which', 'about', 'because', 'would', 'could', 'should', 'from'}
            words = [w for w in words if w not in common_words]
            
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Return top themes
        return word_counts.most_common(20)
    
    def _find_shared_relationships(self, relationships: Dict[str, Dict], 
                                 concept_names: List[str]) -> List[Dict[str, Any]]:
        """
        Find relationships that are shared across multiple concepts.
        
        Args:
            relationships: Dictionary mapping concept names to their relationships
            concept_names: List of concept names
            
        Returns:
            List of shared relationship data
        """
        # Collect all concepts mentioned in relationships
        all_related_concepts = set()
        for concept in concept_names:
            for rel_key in relationships[concept]:
                if isinstance(rel_key, frozenset):
                    all_related_concepts.update(rel_key)
                elif isinstance(rel_key, tuple):
                    all_related_concepts.update(rel_key)
                else:
                    # Handle string keys from JSON deserialization
                    try:
                        related = eval(rel_key)
                        if isinstance(related, list):
                            all_related_concepts.update(related)
                    except:
                        pass
        
        # Remove the main concepts
        related_concepts = all_related_concepts - set(concept_names)
        
        # Find concepts that appear in multiple relationship sets
        shared_relationships = []
        
        for related in related_concepts:
            appearances = []
            
            for concept in concept_names:
                # Check if this related concept appears in this concept's relationships
                for rel_key in relationships[concept]:
                    if isinstance(rel_key, frozenset) and related in rel_key:
                        appearances.append({
                            "main_concept": concept,
                            "relationship": relationships[concept][rel_key],
                            "key": rel_key
                        })
                    elif isinstance(rel_key, tuple) and related in rel_key:
                        appearances.append({
                            "main_concept": concept,
                            "relationship": relationships[concept][rel_key],
                            "key": rel_key
                        })
            
            # If the related concept appears in multiple main concepts
            if len(appearances) > 1:
                shared_relationships.append({
                    "related_concept": related,
                    "appearances": appearances,
                    "count": len(appearances)
                })
        
        # Sort by count (most shared first)
        shared_relationships.sort(key=lambda x: x["count"], reverse=True)
        
        return shared_relationships
    
    def _generate_comparison_graph(self, concept_names: List[str], 
                                 concept_sets: Dict[str, List[str]],
                                 similarity_matrix: Dict[str, Dict[str, float]],
                                 shared_relationships: List[Dict[str, Any]]) -> nx.Graph:
        """
        Generate a graph showing relationships between concepts.
        
        Args:
            concept_names: List of concept names
            concept_sets: Dictionary mapping concept names to their related concepts
            similarity_matrix: Matrix of similarity scores between concepts
            shared_relationships: List of shared relationship data
            
        Returns:
            NetworkX graph representing the comparison
        """
        G = nx.Graph()
        
        # Add main concept nodes
        for concept in concept_names:
            G.add_node(concept, type="main", size=2000)
        
        # Add edges between main concepts based on similarity
        for i, c1 in enumerate(concept_names):
            for j, c2 in enumerate(concept_names):
                if i < j:  # Avoid duplicates
                    similarity = similarity_matrix[c1][c2]
                    if similarity > 0.1:  # Only add edges with some similarity
                        G.add_edge(c1, c2, weight=similarity, type="similarity")
        
        # Add shared concepts as nodes with edges to main concepts
        added_concepts = set()
        for rel_data in shared_relationships:
            related = rel_data["related_concept"]
            
            # Add the related concept node if it's not already added
            if related not in added_concepts and len(added_concepts) < 20:  # Limit to 20 shared concepts
                G.add_node(related, type="shared", size=1000)
                added_concepts.add(related)
                
                # Add edges to the main concepts
                for appearance in rel_data["appearances"]:
                    main_concept = appearance["main_concept"]
                    rel_type = appearance["relationship"].get("relationship_type", "related_to")
                    
                    G.add_edge(main_concept, related, type=rel_type, weight=1.0)
        
        # Add unique concepts for each main concept
        unique_dict = self._find_unique_concepts(concept_sets)
        for main_concept, uniques in unique_dict.items():
            for i, unique in enumerate(uniques):
                if i < 5:  # Limit to 5 unique concepts per main concept
                    G.add_node(unique, type="unique", size=500)
                    G.add_edge(main_concept, unique, type="unique_to", weight=0.5)
        
        return G
    
    @log_execution()
    def visualize_comparison(self, output_file: str = None) -> plt.Figure:
        """
        Visualize the concept comparison results.
        
        Args:
            output_file: Path to save the visualization (optional)
            
        Returns:
            matplotlib Figure object
        """
        if not self.comparison_results:
            logger.error("No comparison results to visualize")
            return None
        
        concepts = self.comparison_results["concepts"]
        similarity_matrix = self.comparison_results["similarity_matrix"]
        graph = self.comparison_results["comparison_graph"]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Similarity matrix heatmap
        ax1 = fig.add_subplot(121)
        sim_data = np.array([[similarity_matrix[c1][c2] for c2 in concepts] for c1 in concepts])
        im = ax1.imshow(sim_data, cmap="YlGnBu")
        ax1.set_xticks(np.arange(len(concepts)))
        ax1.set_yticks(np.arange(len(concepts)))
        ax1.set_xticklabels(concepts)
        ax1.set_yticklabels(concepts)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                text = ax1.text(j, i, f"{sim_data[i, j]:.2f}", 
                               ha="center", va="center", color="black" if sim_data[i, j] < 0.7 else "white")
        ax1.set_title("Concept Similarity Matrix")
        fig.colorbar(im, ax=ax1)
        
        # 2. Concept relationship graph
        ax2 = fig.add_subplot(122)
        pos = nx.spring_layout(graph, seed=42)
        
        # Get node types and sizes
        node_types = nx.get_node_attributes(graph, 'type')
        node_sizes = nx.get_node_attributes(graph, 'size')
        
        # Create node lists by type
        main_nodes = [n for n, t in node_types.items() if t == "main"]
        shared_nodes = [n for n, t in node_types.items() if t == "shared"]
        unique_nodes = [n for n, t in node_types.items() if t == "unique"]
        
        # Get sizes for each node list
        main_sizes = [node_sizes.get(n, 2000) for n in main_nodes]
        shared_sizes = [node_sizes.get(n, 1000) for n in shared_nodes]
        unique_sizes = [node_sizes.get(n, 500) for n in unique_nodes]
        
        # Draw nodes by type with different colors
        nx.draw_networkx_nodes(graph, pos, nodelist=main_nodes, node_color="skyblue", 
                              node_size=main_sizes, alpha=0.8, ax=ax2)
        nx.draw_networkx_nodes(graph, pos, nodelist=shared_nodes, node_color="lightgreen", 
                              node_size=shared_sizes, alpha=0.6, ax=ax2)
        nx.draw_networkx_nodes(graph, pos, nodelist=unique_nodes, node_color="salmon", 
                              node_size=unique_sizes, alpha=0.6, ax=ax2)
        
        # Get edge types and weights
        edge_types = nx.get_edge_attributes(graph, 'type')
        
        # Create edge lists by type
        similarity_edges = [e for e, t in edge_types.items() if t == "similarity"]
        relationship_edges = [e for e, t in edge_types.items() if t not in ["similarity", "unique_to"]]
        unique_edges = [e for e, t in edge_types.items() if t == "unique_to"]
        
        # Draw edges by type with different colors and widths
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        
        # Similarity edges between main concepts
        if similarity_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=similarity_edges, 
                                 width=[edge_weights[e] * 5 for e in similarity_edges], 
                                 alpha=0.7, edge_color="blue", ax=ax2)
        
        # Relationship edges to shared concepts
        if relationship_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=relationship_edges, 
                                 width=[edge_weights[e] * 3 for e in relationship_edges], 
                                 alpha=0.5, edge_color="green", ax=ax2)
        
        # Edges to unique concepts
        if unique_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=unique_edges, 
                                 width=[edge_weights[e] * 2 for e in unique_edges], 
                                 alpha=0.4, edge_color="red", ax=ax2)
        
        # Draw labels with different font sizes based on node type
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in main_nodes}, 
                              font_size=12, font_weight="bold", ax=ax2)
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in shared_nodes}, 
                              font_size=10, ax=ax2)
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in unique_nodes}, 
                              font_size=8, ax=ax2)
        
        ax2.set_title("Concept Relationship Graph")
        ax2.axis('off')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Visualization saved to {output_file}")
        
        return fig
    
    def generate_comparison_summary(self) -> str:
        """
        Generate a textual summary of the concept comparison.
        
        Returns:
            Markdown-formatted summary text
        """
        if not self.comparison_results:
            return "No comparison results available."
        
        concepts = self.comparison_results["concepts"]
        similarity_matrix = self.comparison_results["similarity_matrix"]
        common_concepts = self.comparison_results["common_concepts"]
        unique_concepts = self.comparison_results["unique_concepts"]
        key_themes = self.comparison_results["key_themes"]
        shared_relationships = self.comparison_results["shared_relationships"]
        
        # Build summary text
        summary = [
            f"# Comparison of {', '.join(concepts)}",
            "\n## Concept Similarity",
        ]
        
        # Add similarity information
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i < j:  # Avoid duplicates
                    similarity = similarity_matrix[c1][c2]
                    similarity_text = f"**{similarity:.2f}**" if similarity > 0.5 else f"{similarity:.2f}"
                    summary.append(f"- {c1} and {c2}: {similarity_text}")
        
        # Add common concepts
        summary.append("\n## Common Concepts")
        if common_concepts:
            for concept in common_concepts[:10]:  # Limit to top 10
                summary.append(f"- {concept}")
        else:
            summary.append("- No concepts common to all compared concepts.")
        
        # Add unique concepts
        summary.append("\n## Unique Concepts")
        for concept, uniques in unique_concepts.items():
            summary.append(f"\n### Unique to {concept}")
            if uniques:
                for unique in uniques[:5]:  # Limit to top 5
                    summary.append(f"- {unique}")
            else:
                summary.append("- No concepts unique to this concept.")
        
        # Add key themes
        summary.append("\n## Key Themes")
        for theme, count in key_themes[:10]:  # Limit to top 10
            summary.append(f"- {theme} ({count} occurrences)")
        
        # Add shared relationships
        summary.append("\n## Shared Relationships")
        if shared_relationships:
            for rel in shared_relationships[:5]:  # Limit to top 5
                related = rel["related_concept"]
                summary.append(f"\n### {related}")
                for appearance in rel["appearances"]:
                    main_concept = appearance["main_concept"]
                    rel_type = appearance["relationship"].get("relationship_type", "related_to")
                    summary.append(f"- Related to {main_concept} as '{rel_type}'")
        else:
            summary.append("- No significant shared relationships found.")
        
        return "\n".join(summary)


def compare_concepts_from_files(concept_names: List[str], input_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Compare multiple concepts from saved files.
    
    Args:
        concept_names: List of concept names to compare
        input_dir: Directory containing saved concept files
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Dictionary with comparison results
    """
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
    
    # Save outputs if output directory is specified
    if output_dir:
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
        comparator.visualize_comparison(visualization_path)
        
        # Save summary
        summary = comparator.generate_comparison_summary()
        with open(os.path.join(output_dir, "comparison_summary.md"), 'w', encoding='utf-8') as f:
            f.write(summary)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare philosophical concepts")
    parser.add_argument("--concepts", nargs='+', required=True, help="Concepts to compare")
    parser.add_argument("--input-dir", default="output", help="Input directory for concept files")
    parser.add_argument("--output-dir", help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "comparison")
    
    print(f"Comparing concepts: {', '.join(args.concepts)}")
    results = compare_concepts_from_files(args.concepts, args.input_dir, args.output_dir)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Comparison completed successfully")
        print(f"Results saved to {args.output_dir}")
        print("\nSimilarity Matrix:")
        for c1 in args.concepts:
            for c2 in args.concepts:
                if c1 != c2:
                    print(f"  {c1} - {c2}: {results['similarity_matrix'][c1][c2]:.2f}")
                    
        print("\nCommon Concepts:")
        for concept in results['common_concepts'][:5]:  # Show top 5
            print(f"  - {concept}")
            
        print("\nKey Themes:")
        for theme, count in results['key_themes'][:5]:  # Show top 5
            print(f"  - {theme} ({count})")