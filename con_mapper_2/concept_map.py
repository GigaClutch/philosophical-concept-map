import sys
import wikipedia
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from collections import Counter
import argparse
import os
import time
import json
import tkinter as tk

# Initialize NLP
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please install it with: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Core philosophical terms to detect
philosophical_terms = [
    "Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", 
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology", 
    "Determinism", "Free Will", "Consciousness", "Virtue Ethics", 
    "Deontology", "Utilitarianism", "Moral Realism", "Relativism",
    "Ontology", "Dualism", "Materialism", "Idealism", "Pragmatism",
    "Positivism", "Skepticism", "Nihilism", "Subjectivism", "Objectivism"
]

def get_wikipedia_content(concept_name, cache_dir="wiki_cache"):
    """
    Fetches Wikipedia content with caching to reduce API calls.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{concept_name.replace(' ', '_')}.txt")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        print(f"Loading cached content for '{concept_name}'")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Otherwise, fetch from Wikipedia
    try:
        print(f"Fetching Wikipedia content for '{concept_name}'")
        # Use auto_suggest=False to prevent Wikipedia from changing our search term
        page = wikipedia.page(concept_name, auto_suggest=False)
        content = page.content
        print(f"Successfully retrieved page: {page.title} ({len(content)} characters)")
        
        # Cache the content
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    except wikipedia.exceptions.PageError as e:
        print(f"Wikipedia PageError for '{concept_name}': {e}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"\nDisambiguation error for '{concept_name}'. Options include: {', '.join(e.options[:5])}")
        return None
    except Exception as e:
        print(f"Unexpected error when fetching '{concept_name}': {str(e)}")
        return None

def extract_all_concepts(wiki_text):
    """
    Extracts concepts using both NER and a predefined list of philosophical terms.
    """
    if not wiki_text:
        return []
    
    print("Extracting concepts from text...")
    
    # NER extraction
    doc = nlp(wiki_text[:10000])  # Limit text for faster processing
    ner_concepts = set()
    relevant_entity_types = ["ORG", "PERSON", "NORP", "MISC", "GPE"]
    for ent in doc.ents:
        if ent.label_ in relevant_entity_types:
            ner_concepts.add(ent.text)
    
    # Keyword-based extraction
    keyword_concepts = set()
    for term in philosophical_terms:
        if term.lower() in wiki_text.lower():
            keyword_concepts.add(term)
    
    # Combine both approaches
    all_concepts = ner_concepts.union(keyword_concepts)
    return list(all_concepts)

def extract_rich_relationships(input_concept, wiki_text, extracted_concepts):
    """
    Extracts relationships with richer semantic content.
    """
    if not wiki_text:
        return {}
    
    print(f"Analyzing relationships for '{input_concept}'...")
    
    doc = nlp(wiki_text[:20000])  # Limit text for faster processing
    sentences = list(doc.sents)
    relationship_data = {}
    
    input_concept_lower = input_concept.lower()
    
    for sentence in sentences:
        sentence_text = sentence.text.lower()
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
    
    return relationship_data

def generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data):
    """
    Generate a textual summary of the concept map.
    """
    if not wiki_text or not extracted_concepts or not relationship_data:
        return "Insufficient data to generate a summary."
    
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
            # Get verbs if available
            verb_str = "related to"
            
            summary.append(f"- **{concept}** (mentioned together {count} times)")
            
            # Include a sample sentence
            if rel_data["sentences"]:
                example = rel_data["sentences"][0]
                if len(example) > 100:
                    example = example[:97] + "..."
                summary.append(f"  - Example: \"{example}\"")
    
    return "\n".join(summary)

def save_results(input_concept, wiki_text, extracted_concepts, relationship_data, output_dir="output"):
    """
    Save all results to files for future reference.
    """
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
        print(f"Warning: Could not generate summary: {e}")
    
    print(f"All results saved to {concept_dir}")
    return concept_dir

def create_visualization(concept, relationship_data, extracted_concepts, threshold=1.0):
    """
    Create a basic visualization of the concept map.
    """
    print("Creating visualization...")
    
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
    
    # Filter concepts by threshold and limit to top 15 for clarity
    filtered_concepts = [(c, score) for c, score in concept_relevance.items() if score >= threshold]
    top_concepts = sorted(filtered_concepts, key=lambda x: x[1], reverse=True)[:15]
    
    for related_concept, score in top_concepts:
        G.add_node(related_concept)
        G.add_edge(concept, related_concept, weight=score)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
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
    
    # Save and show the visualization
    output_file = f"{concept}_concept_map.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.show()

def main():
    print("Starting main function")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate philosophical concept maps")
    parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
    parser.add_argument("--threshold", type=float, default=1.0, help="Relevance threshold for filtering concepts")
    parser.add_argument("--compare", nargs='+', help="Compare multiple philosophical concepts")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    print(f"Parsed arguments: {args}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # GUI mode
    if args.gui:
        print("Launching GUI mode...")
        try:
            from gui import ConceptMapApp
            root = tk.Tk()
            app = ConceptMapApp(root)
            root.mainloop()
            return
        except Exception as e:
            print(f"Error launching GUI: {e}")
            import traceback
            traceback.print_exc()
    
    # Command line mode
    concept = args.concept
    if not concept:
        concept = input("Enter a philosophical concept: ")
    
    print(f"Generating concept map for: '{concept}'...")
    
    # Get Wikipedia content
    wiki_text = get_wikipedia_content(concept)
    if not wiki_text:
        print(f"Error: Could not retrieve Wikipedia page for '{concept}'.")
        return
    
    # Extract concepts
    extracted_concepts = extract_all_concepts(wiki_text)
    print(f"Found {len(extracted_concepts)} related concepts")
    
    # Extract relationships
    relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
    print(f"Found {len(relationship_data)} relationships")
    
    # Save results
    save_results(concept, wiki_text, extracted_concepts, relationship_data, args.output)
    
    # Create visualization
    create_visualization(concept, relationship_data, extracted_concepts, args.threshold)
    
    print("Concept map generation completed successfully.")

if __name__ == "__main__":
    main()