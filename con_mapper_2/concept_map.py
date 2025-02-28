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
from tkinter import Tk, filedialog
from concept_comparator import ConceptComparator

# Import your custom modules
# from knowledge_base import PhilosophicalKnowledgeBase
# from visualization import create_interactive_visualization

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# Core philosophical terms to detect
philosophical_terms = [
    "Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", 
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology", 
    "Determinism", "Free Will", "Consciousness", "Virtue Ethics", 
    "Deontology", "Utilitarianism", "Moral Realism", "Relativism",
    "Ontology", "Dualism", "Materialism", "Idealism", "Pragmatism",
    "Positivism", "Skepticism", "Nihilism", "Subjectivism", "Objectivism"
]

def extract_concepts_keyword_based(wiki_text, philosophical_terms):
    """
    Extract philosophical concepts based on a curated list of philosophical terms.
    """
    extracted_concepts = set()
    for term in philosophical_terms:
        if term.lower() in wiki_text.lower():
            extracted_concepts.add(term)
    return list(extracted_concepts)

def compare_philosophical_concepts(concept_names, output_file=None):
    """
    Compare multiple philosophical concepts and visualize their relationships.
    
    Args:
        concept_names: List of philosophical concept names to compare
        output_file: Optional file path to save visualization
    """
    print(f"Comparing philosophical concepts: {', '.join(concept_names)}")

    # At the beginning of your compare_philosophical_concepts function
    def compare_philosophical_concepts(concept_names, output_file=None):
        print(f"Starting comparison of: {', '.join(concept_names)}")
        
        # Fetch data for each concept
        concept_data = {}
        for concept in concept_names:
            print(f"Processing '{concept}'...")
            wiki_text = get_wikipedia_content(concept)
            if wiki_text:
                print(f"✓ Got Wikipedia content for '{concept}' ({len(wiki_text)} characters)")
                extracted_concepts = extract_all_concepts(wiki_text)
                print(f"✓ Extracted {len(extracted_concepts)} concepts")
                relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
                print(f"✓ Analyzed {len(relationship_data)} relationships")
                concept_data[concept] = {
                    "wiki_text": wiki_text,
                    "extracted_concepts": extracted_concepts,
                    "relationships": relationship_data
                }
            else:
                print(f"✗ Failed to retrieve Wikipedia content for '{concept}'. Skipping.")
        
        # Rest of your function remains the same...
    
    # Fetch data for each concept
    concept_data = {}
    for concept in concept_names:
        print(f"Processing '{concept}'...")
        wiki_text = get_wikipedia_content(concept)
        if wiki_text:
            extracted_concepts = extract_all_concepts(wiki_text)
            relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
            concept_data[concept] = {
                "wiki_text": wiki_text,
                "extracted_concepts": extracted_concepts,
                "relationships": relationship_data
            }
        else:
            print(f"Could not retrieve Wikipedia content for '{concept}'. Skipping.")
    
    # Perform comparison if we have at least 2 concepts
    if len(concept_data) >= 2:
        comparator = ConceptComparator()
        comparison_results = comparator.compare_concepts(concept_data)
        
        # Visualize the comparison
        comparator.visualize_comparison(comparison_results, output_file)
        
        print("Comparison completed. Key insights:")
        print(f"  Textual similarity: {comparison_results['similarity_matrix']}")
        print(f"  Common concepts: {', '.join(comparison_results['common_concepts'][:5])}")
        
        return comparison_results
    else:
        print("Need at least two valid concepts to compare.")
        return None

# Common operations
def get_wikipedia_content(concept_name, cache_dir="wiki_cache"):
    """
    Fetches Wikipedia content with caching to reduce API calls.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{concept_name.replace(' ', '_')}.txt")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Otherwise, fetch from Wikipedia
    try:
        page = wikipedia.page(concept_name, auto_suggest=False, redirect=True)
        content = page.content
        
        # Cache the content
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    except wikipedia.exceptions.PageError as e:
        print(f"Wikipedia PageError for '{concept_name}': {e}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"\nDisambiguation error for '{concept_name}'. Please choose a more specific topic:")
        for i, option in enumerate(e.options):
            print(f"  {i+1}. {option}")
        
        while True:
            try:
                choice_index = int(input(f"Enter the number of your choice (1-{len(e.options)}): ")) - 1
                if 0 <= choice_index < len(e.options):
                    chosen_concept = e.options[choice_index]
                    print(f"You chose: {chosen_concept}")
                    return get_wikipedia_content(chosen_concept)
                else:
                    print("Invalid choice. Please enter a number within the valid range.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        return None

def extract_all_concepts(wiki_text):
    """
    Extracts concepts using both NER and a predefined list of philosophical terms.
    """
    if not wiki_text:
        return []
    
    # NER extraction
    doc = nlp(wiki_text)
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
    
    doc = nlp(wiki_text)
    sentences = list(doc.sents)
    relationship_data = {}
    
    input_concept_lower = input_concept.lower()
    
    for sentence in sentences:
        sentence_text = sentence.text.lower()
        if input_concept_lower in sentence_text:
            # Process the sentence to extract semantic information
            sent_doc = nlp(sentence.text)
            
            # Find verbs in the sentence
            verbs = [token.text for token in sent_doc if token.pos_ == "VERB"]
            
            # Check which concepts co-occur in this sentence
            for concept in extracted_concepts:
                concept_lower = concept.lower()
                if concept_lower != input_concept_lower and concept_lower in sentence_text:
                    # Create a unique key for this concept pair
                    pair_key = frozenset({input_concept, concept})
                    
                    # Initialize data structure if this is the first occurrence
                    if pair_key not in relationship_data:
                        relationship_data[pair_key] = {
                            "count": 0,
                            "verbs": Counter(),
                            "sentences": []
                        }
                    
                    # Update relationship data
                    relationship_data[pair_key]["count"] += 1
                    for verb in verbs:
                        relationship_data[pair_key]["verbs"][verb] += 1
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
    concept_relevance = Counter()
    for pair, data in relationship_data.items():
        pair_list = list(pair)
        if pair_list[0] == input_concept:
            concept_relevance[pair_list[1]] += data["count"]
        elif pair_list[1] == input_concept:
            concept_relevance[pair_list[0]] += data["count"]
    
    top_concepts = concept_relevance.most_common(5)
    
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
            common_verbs = rel_data["verbs"].most_common(3)
            verb_str = ", ".join([f"'{v}'" for v, c in common_verbs]) if common_verbs else "related to"
            summary.append(f"- **{concept}** (mentioned together {count} times, common verbs: {verb_str})")
            
            # Include a sample sentence
            if rel_data["sentences"]:
                example = rel_data["sentences"][0]
                if len(example) > 100:
                    example = example[:97] + "..."
                summary.append(f"  - Example: \"{example}\"")
    
    # Add category information if available
    summary.append("\n## Concept Categories:")
    category_counts = Counter()
    
    # This would use your concept categories
    concept_categories = {
        "Justice": "Concept", "Ethics": "Concept", "Plato": "Philosopher",
        "Kant": "Philosopher", "Utilitarianism": "Ethical Theory"
        # Add more categories as needed
    }
    
    for concept in extracted_concepts:
        category = concept_categories.get(concept, "Uncategorized")
        category_counts[category] += 1
    
    for category, count in category_counts.items():
            summary.append(f"- **{category}**: {count} concepts")
        
        # Add recommendations for further exploration
    summary.append("\n## Recommended Concepts for Further Exploration:")
        
    # Find concepts that are 2nd degree connections (related to the top concepts)
    second_degree = set()
    for top_concept, _ in top_concepts:
        for pair, data in relationship_data.items():
            pair_list = list(pair)
            if top_concept in pair_list and input_concept not in pair_list:
                other_concept = pair_list[0] if pair_list[1] == top_concept else pair_list[1]
                if other_concept != input_concept and other_concept not in [c for c, _ in top_concepts]:
                    second_degree.add(other_concept)
        
        for concept in list(second_degree)[:5]:  # Take up to 5 recommendations
            summary.append(f"- **{concept}**")
        
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
        summary = generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data)
        with open(os.path.join(concept_dir, "summary.md"), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"All results saved to {concept_dir}")
        return concept_dir

    def main():
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate philosophical concept maps")
        parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
        parser.add_argument("--threshold", type=float, default=1.0, help="Relevance threshold for filtering concepts")
        parser.add_argument("--compare", nargs='+', help="Compare multiple philosophical concepts")
        parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
        parser.add_argument("--output", type=str, default="output", help="Output directory")
        args = parser.parse_args()
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Handle compare mode first
        if args.compare:
            print(f"Compare mode activated with concepts: {args.compare}")
            if len(args.compare) >= 2:
                try:
                    compare_philosophical_concepts(args.compare, os.path.join(args.output, "concept_comparison.png"))
                    print(f"Comparison complete. Output saved to {os.path.join(args.output, 'concept_comparison.png')}")
                    return
                except Exception as e:
                    print(f"Error during comparison: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Need at least two concepts to compare. Example: --compare Ethics Justice")
        
        # Rest of your main function for regular concept mapping...
        
        if args.gui:
            # Launch the GUI (you would import and use your ConceptMapApp here)
            print("Launching GUI mode...")
            root = Tk()
            # app = ConceptMapApp(root)
            root.mainloop()
            return
        
        # In the main() function, modify the comparison handling section:
        if args.compare:
            print(f"Compare argument detected: {args.compare}")
            if len(args.compare) >= 2:
                print(f"Attempting to compare: {args.compare}")
                compare_philosophical_concepts(args.compare, os.path.join(args.output, "concept_comparison.png"))
                return
            else:
                print("Need at least two concepts to compare. Example: --compare Ethics Justice")
        
        # Command line mode
        concept = args.concept
        if not concept:
            concept = input("Enter a philosophical concept: ")
        
        print(f"Generating concept map for: '{concept}'...")
        start_time = time.time()
        
        # Get Wikipedia content
        print("Fetching Wikipedia content...")
        wiki_text = get_wikipedia_content(concept)
        if not wiki_text:
            print(f"Error: Could not retrieve Wikipedia page for '{concept}'.")
            return
        
        # Extract concepts
        print("Extracting concepts...")
        extracted_concepts = extract_all_concepts(wiki_text)
        print(f"Found {len(extracted_concepts)} related concepts.")
        
        # Extract relationships
        print("Analyzing concept relationships...")
        relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
        print(f"Found {len(relationship_data)} relationships.")
        
        # Save results
        output_path = save_results(concept, wiki_text, extracted_concepts, relationship_data, args.output)
        
        # Generate visualization
        if args.interactive:
            print("Creating interactive visualization...")
            # You would call your interactive visualization function here
            # create_interactive_visualization(concept, relationship_data, extracted_concepts, args.threshold)
        else:
            print("Creating static visualization...")
            # Generate the static concept map
            # generate_concept_map(concept, extracted_concepts, relationship_data, relevance_threshold=args.threshold)
        
        elapsed_time = time.time() - start_time
        print(f"\nConcept map generation completed in {elapsed_time:.2f} seconds.")
        print(f"Results saved to: {output_path}")

    if __name__ == "__main__":
        main()