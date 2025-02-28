import sys
import wikipedia
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import Counter
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Slider

def create_interactive_visualization(input_concept, relationship_data, extracted_concepts, min_threshold=0, max_threshold=10):
    """
    Creates an interactive visualization with adjustable relevance threshold.
    """
    def update_graph(threshold):
        plt.clf()
        graph = nx.DiGraph()
        graph.add_node(input_concept, label=input_concept)
        
        # Calculate relevance scores
        concept_relevance_scores = Counter()
        for concept_pair, data in relationship_data.items():
            concept_pair_list = list(concept_pair)
            concept1, concept2 = concept_pair_list[0], concept_pair_list[1]
            if concept1 == input_concept:
                concept_relevance_scores[concept2] += data["count"]
            elif concept2 == input_concept:
                concept_relevance_scores[concept1] += data["count"]
        
        # Filter concepts based on threshold
        filtered_concepts = set()
        for concept in extracted_concepts:
            if concept_relevance_scores[concept] >= threshold:
                filtered_concepts.add(concept)
                graph.add_node(concept, label=concept)
                
                # Add relationships with most common verb as label
                if frozenset({input_concept, concept}) in relationship_data:
                    rel_data = relationship_data[frozenset({input_concept, concept})]
                    common_verb = rel_data["verbs"].most_common(1)[0][0] if rel_data["verbs"] else "related to"
                    weight = rel_data["count"]
                    graph.add_edge(input_concept, concept, 
                                   relation=common_verb, 
                                   weight=weight)
        
        # Color nodes based on relevance
        relevance_values = [concept_relevance_scores[node] if node != input_concept else max(concept_relevance_scores.values()) 
                           for node in graph.nodes()]
        norm = plt.Normalize(min(relevance_values), max(relevance_values))
        node_colors = [cm.viridis(norm(value)) for value in relevance_values]
        
        # Position nodes 
        pos = nx.spring_layout(graph, seed=42)
        
        # Make input concept central
        if input_concept in pos:
            central_node_pos = pos[input_concept]
            for node in graph.neighbors(input_concept):
                if 'weight' in graph[input_concept][node]:
                    weight = graph[input_concept][node]['weight']
                    if weight > 0:
                        distance_factor = 0.8 / math.log(1 + weight)
                    else:
                        distance_factor = 1.5
                    original_vector = pos[node] - central_node_pos
                    scaled_vector = original_vector * distance_factor
                    new_pos = central_node_pos + scaled_vector
                    pos[node] = new_pos
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color=node_colors)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
        
        # Draw edges with width proportional to weight
        edge_weights = [(u, v, d['weight']) for u, v, d in graph.edges(data=True) if 'weight' in d]
        
        if edge_weights:
            edge_widths = [weight * 0.2 for _, _, weight in edge_weights]
            nx.draw_networkx_edges(graph, pos, width=edge_widths, arrows=True, 
                                   edge_color="gray", alpha=0.5, connectionstyle='arc3,rad=0.2')
        
        # Edge labels
        edge_labels = {(u, v): d['relation'] for u, v, d in graph.edges(data=True) if 'relation' in d}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Concept Map for '{input_concept}' (Threshold={threshold:.1f})")
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Add a slider for adjusting the threshold
    slider_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    threshold_slider = Slider(
        slider_ax, 'Relevance Threshold', min_threshold, max_threshold, 
        valinit=min_threshold, valstep=0.1
    )
    
    # Update function for the slider
    def update(val):
        threshold = threshold_slider.val
        update_graph(threshold)
    
    threshold_slider.on_changed(update)
    
    # Initial plot
    update_graph(min_threshold)
    
    plt.savefig(f"interactive_concept_map_{input_concept}.png")
    plt.show()