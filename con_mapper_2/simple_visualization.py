"""
Simple visualization module for the concept map generator.
This provides sample data and visualization when Wikipedia access isn't working.
"""
import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import math
from pathlib import Path

def create_sample_data(concept_name="Justice"):
    """Create sample data for the given concept."""
    # Create sample wiki text
    wiki_text = f"""
    {concept_name} is a fundamental philosophical concept related to fairness and equality.
    In philosophical discussions, {concept_name} is often connected to Ethics, Morality, and Social Philosophy.
    
    Various philosophers including Plato, Aristotle, and John Rawls have developed theories of {concept_name}.
    """
    
    # Create sample concepts
    concepts = [
        {"concept": concept_name, "relevance_score": 1.0},
        {"concept": "Ethics", "relevance_score": 0.95},
        {"concept": "Fairness", "relevance_score": 0.9},
        {"concept": "Equality", "relevance_score": 0.85},
        {"concept": "Morality", "relevance_score": 0.8},
        {"concept": "Social Philosophy", "relevance_score": 0.75},
        {"concept": "Plato", "relevance_score": 0.7},
        {"concept": "Aristotle", "relevance_score": 0.65},
        {"concept": "John Rawls", "relevance_score": 0.6},
        {"concept": "Law", "relevance_score": 0.55},
        {"concept": "Rights", "relevance_score": 0.5}
    ]
    
    # Create sample relationships
    relationships = {}
    for c in concepts:
        if c["concept"] != concept_name:
            key = frozenset({concept_name, c["concept"]})
            relationships[key] = {
                "count": int(c["relevance_score"] * 10),
                "sentences": [f"{concept_name} is related to {c['concept']}."],
                "relationship_type": "related_to",
                "is_directed": False,
                "direction": None
            }
    
    # Create result directories
    output_dir = Path("results") / concept_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save wiki text
    with open(output_dir / "wiki_text.txt", 'w', encoding='utf-8') as f:
        f.write(wiki_text)
    
    # Save concepts
    with open(output_dir / "extracted_concepts.json", 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2)
    
    # Save relationships
    serializable_relationships = {}
    for key, value in relationships.items():
        string_key = str(sorted(list(key)))
        serializable_relationships[string_key] = value
    
    with open(output_dir / "relationships.json", 'w', encoding='utf-8') as f:
        json.dump(serializable_relationships, f, indent=2)
    
    # Create summary
    summary = f"""# Concept Map Summary for '{concept_name}'

The concept map for '{concept_name}' contains {len(concepts)} related philosophical concepts and {len(relationships)} relationships.

## Top Related Concepts by Relevance
- **Ethics** (relevance: 0.95)
- **Fairness** (relevance: 0.90)
- **Equality** (relevance: 0.85)
- **Morality** (relevance: 0.80)
- **Social Philosophy** (relevance: 0.75)

## Relationship Types
- **related_to**: {len(relationships)} relationships

## Example Relationships
- {concept_name} is related to Ethics
- {concept_name} is related to Fairness
- {concept_name} is related to Equality
"""
    
    with open(output_dir / "summary.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Sample data created successfully in {output_dir}")
    return concepts, relationships

def create_visualization(concept, relationship_data, extracted_concepts, threshold=1.0):
    """Create a simple visualization of the concept map."""
    # Create a graph
    G = nx.Graph()
    
    # Add the main concept
    G.add_node(concept, type="main")
    
    # Process concepts based on format
    if not extracted_concepts:
        # If no concepts provided, generate sample data
        print(f"No concepts provided, generating sample data for {concept}")
        extracted_concepts, relationship_data = create_sample_data(concept)
    
    # Filter concepts by threshold
    if isinstance(extracted_concepts[0], dict):
        # New format with dictionaries
        filtered_concepts = [c for c in extracted_concepts 
                           if c.get("relevance_score", 0) >= threshold]
        filtered_concepts = sorted(filtered_concepts, 
                                 key=lambda x: x.get("relevance_score", 0), 
                                 reverse=True)[:10]
    else:
        # Old format with strings
        filtered_concepts = [{"concept": c, "relevance_score": 0.5} for c in extracted_concepts[:10]]
    
    # Add related concepts as nodes and edges
    for concept_data in filtered_concepts:
        related_concept = concept_data.get("concept", concept_data)
        relevance = concept_data.get("relevance_score", 0.5) if isinstance(concept_data, dict) else 0.5
        
        if related_concept != concept:
            # Add node
            G.add_node(related_concept, type="related", relevance=relevance)
            
            # Add edge
            weight = 1.0
            # Try to get weight from relationship data
            for key, data in relationship_data.items():
                if isinstance(key, frozenset) and concept in key and related_concept in key:
                    weight = data.get("count", 1)
                    break
            
            G.add_edge(concept, related_concept, weight=weight)
    
    # Position nodes - central concept with others in a circle
    pos = {concept: (0.5, 0.5)}  # Center the main concept
    
    # Position related concepts in a circle
    nodes = [n for n in G.nodes if n != concept]
    num_nodes = len(nodes)
    
    if num_nodes > 0:
        radius = 0.4
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            pos[node] = (x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw nodes - main concept is larger and a different color
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[concept], 
                         node_color="skyblue", 
                         node_size=3000, 
                         alpha=0.8,
                         ax=ax)
    
    # Draw related concepts - colored by relevance
    related_nodes = [n for n in G.nodes if n != concept]
    if related_nodes:
        node_colors = [G.nodes[n].get("relevance", 0.5) for n in related_nodes]
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=related_nodes, 
                             node_color=node_colors, 
                             cmap=plt.cm.Blues, 
                             node_size=2000, 
                             alpha=0.6,
                             ax=ax)
    
    # Draw edges with varying widths based on weight
    if G.edges():
        edge_widths = [G[u][v].get('weight', 1) / 2 + 1 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                             width=edge_widths, 
                             alpha=0.6, 
                             edge_color="gray",
                             ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Set title and disable axis
    plt.title(f"Concept Map for '{concept}'")
    plt.axis('off')
    
    return fig

if __name__ == "__main__":
    # Test functionality
    create_sample_data("Justice")
    fig = create_visualization("Justice", {}, [])
    plt.show()