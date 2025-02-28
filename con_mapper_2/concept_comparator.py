import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import spacy

class ConceptComparator:
    """
    Analyzes and compares multiple philosophical concepts to identify relationships,
    similarities, differences, and common themes.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def compare_concepts(self, concept_data_dict):
        """
        Compare multiple philosophical concepts.
        
        Args:
            concept_data_dict: Dictionary of {concept_name: {wiki_text, extracted_concepts, relationships}}
        
        Returns:
            Comparison results including similarities, differences, and visualization data
        """
        if len(concept_data_dict) < 2:
            return {"error": "Need at least two concepts to compare"}
        
        # Extract data for comparison
        concepts = list(concept_data_dict.keys())
        extracted_concepts = {c: data["extracted_concepts"] for c, data in concept_data_dict.items()}
        
        # Compute simple overlap-based similarity instead of TF-IDF
        similarity_matrix = self._compute_concept_overlap(extracted_concepts)
        
        # Find common and unique concepts
        common_concepts = self._find_common_concepts(extracted_concepts)
        unique_concepts = self._find_unique_concepts(extracted_concepts)
        
        # Identify key themes
        key_themes = self._identify_key_themes(concept_data_dict)
        
        # Generate comparison graph
        comparison_graph = self._generate_comparison_graph(concepts, extracted_concepts, similarity_matrix)
        
        return {
            "concepts": concepts,
            "similarity_matrix": similarity_matrix,
            "common_concepts": common_concepts,
            "unique_concepts": unique_concepts,
            "key_themes": key_themes,
            "comparison_graph": comparison_graph
        }
    
    def _compute_concept_overlap(self, extracted_concepts):
        """Compute similarity based on concept overlap."""
        concepts = list(extracted_concepts.keys())
        similarity_dict = {}
        
        for i, c1 in enumerate(concepts):
            similarity_dict[c1] = {}
            set1 = set(extracted_concepts[c1])
            for j, c2 in enumerate(concepts):
                set2 = set(extracted_concepts[c2])
                # Jaccard similarity: size of intersection divided by size of union
                if set1 or set2:  # Avoid division by zero
                    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                else:
                    similarity = 0
                similarity_dict[c1][c2] = similarity
        
        return similarity_dict
    
    def _find_common_concepts(self, extracted_concepts):
        """Find concepts that appear in multiple source concepts."""
        all_concepts_sets = list(extracted_concepts.values())
        if not all_concepts_sets:
            return []
        
        # Start with the first set and find intersection with others
        common = set(all_concepts_sets[0])
        for concept_set in all_concepts_sets[1:]:
            common = common.intersection(set(concept_set))
        
        return list(common)
    
    def _find_unique_concepts(self, extracted_concepts):
        """Find concepts unique to each source concept."""
        concepts = list(extracted_concepts.keys())
        unique_dict = {}
        
        for c in concepts:
            # Find concepts that only appear in this source
            current_set = set(extracted_concepts[c])
            other_sets = set()
            for other_c in concepts:
                if other_c != c:
                    other_sets.update(set(extracted_concepts[other_c]))
            
            unique_dict[c] = list(current_set - other_sets)
        
        return unique_dict
    
    def _identify_key_themes(self, concept_data_dict):
        """Identify key themes across concepts by looking at frequent words."""
        # Extract all words from all texts
        all_words = []
        for data in concept_data_dict.values():
            doc = self.nlp(data["wiki_text"])
            # Only consider content words (not stopwords, punctuation, etc.)
            content_words = [token.text.lower() for token in doc 
                            if not token.is_stop and not token.is_punct 
                            and token.is_alpha and len(token.text) > 3]
            all_words.extend(content_words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Return top themes
        return word_counts.most_common(10)
    
    def _generate_comparison_graph(self, concepts, extracted_concepts, similarity_matrix):
        """Generate a graph showing relationships between concepts."""
        G = nx.Graph()
        
        # Add main concept nodes
        for concept in concepts:
            G.add_node(concept, type="main")
        
        # Add edges between main concepts based on similarity
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i < j:  # Avoid duplicates
                    similarity = similarity_matrix[c1][c2]
                    if similarity > 0.1:  # Only add edges with some similarity
                        G.add_edge(c1, c2, weight=similarity)
        
        return G
    
    def visualize_comparison(self, comparison_results, output_file=None):
        """Visualize concept comparison results."""
        concepts = comparison_results["concepts"]
        similarity_matrix = comparison_results["similarity_matrix"]
        graph = comparison_results["comparison_graph"]
        
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
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, nodelist=concepts, node_color="skyblue", 
                              node_size=3000, alpha=0.8, ax=ax2)
        
        # Draw edges with width based on similarity
        edge_weights = [graph[u][v]['weight'] * 5 for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5, ax=ax2)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax2)
        
        ax2.set_title("Concept Relationship Graph")
        ax2.axis('off')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        
        plt.show()
        
        return fig