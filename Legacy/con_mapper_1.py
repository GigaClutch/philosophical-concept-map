import sys
import wikipedia
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import Counter

print(sys.executable)
print("Libraries imported successfully from virtual environment!")

# --- 1. Data Fetching ---
def get_wikipedia_content(concept_name):
    """
    Fetches the text content of a Wikipedia page for a given concept name.
    Handles disambiguation errors by prompting the user to choose a specific topic.
    """
    try:
        page = wikipedia.page(concept_name, auto_suggest=False, redirect=True)
        return page.content
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
                    return get_wikipedia_content(chosen_concept) # Recursive call for chosen concept
                else:
                    print("Invalid choice. Please enter a number within the valid range.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        return None

# --- 2. Concept Extraction (NER) ---
nlp = spacy.load("en_core_web_sm")

def extract_concepts_ner(wiki_text):
    """
    Extracts named entities (potential philosophical concepts) from Wikipedia text using spaCy's NER.
    Filters for specific entity types (ORG, PERSON, NORP, MISC).
    """
    if not wiki_text:
        return []
    doc = nlp(wiki_text)
    concepts = set()
    relevant_entity_types = ["ORG", "PERSON", "NORP", "MISC"] # Filter for these entity types
    for ent in doc.ents:
        if ent.label_ in relevant_entity_types:
            concepts.add(ent.text)
    return list(concepts)

# --- 3. Relationship Extraction (Sentence Co-occurrence with Frequency) ---
def extract_relationships_cooccurrence(input_concept, wiki_text, extracted_concepts_ner):
    """
    Extracts relationships between the input concept and other concepts based on sentence co-occurrence in the text.
    Counts the frequency of co-occurrence for each relationship pair.
    """
    if not wiki_text:
        return {}
    doc = nlp(wiki_text)
    sentences = list(doc.sents)
    relationship_counts = Counter()

    input_concept_lower = input_concept.lower()
    for sentence in sentences:
        sentence_text = sentence.text.lower()
        if input_concept_lower in sentence_text:
            for concept in extracted_concepts_ner:
                concept_lower = concept.lower()
                if concept_lower != input_concept_lower and concept_lower in sentence_text:
                    relationship_counts[frozenset({input_concept, concept})] += 1 # Count co-occurrences
    return relationship_counts

# --- 4. Concept Map Generation and Visualization ---


def generate_concept_map(input_concept, related_concepts_ner, relationship_counts, relevance_threshold=1):
    """
    Generates a concept map graph with relevance filtering.
    - Nodes represent concepts (input concept and NER-extracted concepts) that meet the relevance threshold.
    - Edges represent relationships ('mentions' for NER concepts, 'related_to_sentence' for co-occurrence) for filtered concepts.
    - Edge thickness is weighted by co-occurrence frequency.
    - Node colors are based on predefined concept categories (manual categorization).
    - Layout is relevance-scaled spring layout.
    - Concepts are filtered based on a relevance score threshold.
    """

    graph = nx.DiGraph()
    graph.add_node(input_concept, label=input_concept)

    # --- Calculate Relevance Scores ---
    concept_relevance_scores = Counter() # Use Counter to store relevance scores for each concept
    for concept_pair_frozenset, count in relationship_counts.items():
        concept_pair_list = list(concept_pair_frozenset)
        concept1 = concept_pair_list[0]
        concept2 = concept_pair_list[1]
        if concept1 == input_concept: # Relevance score is based on connections *to* the input concept
            concept_relevance_scores[concept2] += count
        elif concept2 == input_concept:
            concept_relevance_scores[concept1] += count

    # --- Filter Concepts and Relationships based on Relevance Threshold ---
    filtered_concepts = set()
    filtered_weighted_edges = []

    for concept in related_concepts_ner:
        if concept_relevance_scores[concept] >= relevance_threshold: # Apply relevance threshold
            filtered_concepts.add(concept)
            graph.add_node(concept, label=concept) # Add node only if it passes threshold
            graph.add_edge(input_concept, concept, relation="mentions") # Keep 'mentions' edges for filtered concepts

    for concept_pair_frozenset, count in relationship_counts.items():
        concept_pair_list = list(concept_pair_frozenset)
        concept1 = concept_pair_list[0]
        concept2 = concept_pair_list[1]
        if concept1 == input_concept and concept2 in filtered_concepts: # Filter relationships based on filtered concepts
            if concept2 in filtered_concepts: # Double check target is still filtered (redundant but safe)
                graph.add_edge(concept1, concept2, relation="related_to_sentence", weight=count) # Add weighted edge to graph
                filtered_weighted_edges.append((concept1, concept2, count)) # For visualization - use filtered edges!


    # --- Node Coloring (same as before) ---
    concept_categories = { # Manually defined categories for some concepts - extend as needed
        "Justice": "Concept", "Ethics": "Concept", "Plato": "Philosopher", "Kant": "Philosopher",
        "Utilitarianism": "Ethical Theory", "Consequentialism": "Ethical Theory", "Kantianism": "Philosophical School",
        "Slavery": "Concept", "Christian": "Religious Group", "Islamic": "Religious Group", "Ancient Greek": "Philosophical Tradition",
        "David Hume": "Philosopher", "John Locke": "Philosopher", "Jeremy Bentham": "Philosopher", "Simone de Beauvoir": "Philosopher",
        "Friedrich Nietzsche": "Philosopher", "Søren Kierkegaard": "Philosopher", "Aristotle": "Philosopher", "G. E. Moore": "Philosopher",
        "John Stuart Mill": "Philosopher", "Immanuel Kant": "Philosopher", "Georg Wilhelm Friedrich Hegel": "Philosopher", "Jean-Jacques Rousseau": "Philosopher"
        # Add more concept categories here as you analyze your maps and want to categorize more nodes
    }

    node_colors = []
    for node in graph.nodes():
        category = concept_categories.get(node, "Other") # Default category "Other" if not found
        if category == "Philosopher":
            node_colors.append("lightblue")
        elif category == "Concept":
            node_colors.append("lightgreen")
        elif category == "Ethical Theory":
            node_colors.append("lightpurple")
        elif category == "Philosophical School":
            node_colors.append("navajowhite")
        elif category == "Religious Group":
            node_colors.append("lightcoral")
        elif category == "Philosophical Tradition":
            node_colors.append("lightseagreen")
        else:
            node_colors.append("skyblue") # Default color for "Other" or uncategorized

    # --- Layout with Relevance-Scaled Distances (same as before) ---
    pos = nx.spring_layout(graph, seed=42) # Initial layout

    central_node_pos = pos[input_concept]
    for node in graph.neighbors(input_concept):
        if 'weight' in graph[input_concept][node]: # Still check for weight
            weight = graph[input_concept][node]['weight']
            if weight > 0:
                distance_factor = 0.8 / math.log(1 + weight)
            else:
                distance_factor = 1.5
            original_vector = pos[node] - central_node_pos
            scaled_vector = original_vector * distance_factor
            new_pos = central_node_pos + scaled_vector
            pos[node] = new_pos

    # --- Drawing the Graph (mostly same as before, using filtered_weighted_edges for drawing weights) ---
    # --- Drawing the Graph (with check for empty edges list) ---
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

    filtered_weighted_edges = [edge for edge in filtered_weighted_edges if edge[2] > 0] # Ensure weights are positive

    if filtered_weighted_edges: # Check if filtered_weighted_edges is NOT empty before drawing edges
        edge_widths = [weight * 0.2 for _, _, weight in filtered_weighted_edges]
        nx.draw_networkx_edges(graph, pos, width=edge_widths, arrows=True, edge_color="gray", alpha=0.5,
                               connectionstyle='arc3,rad=0.2')
    else:
        print("No weighted edges to draw after relevance filtering with threshold =", relevance_threshold) # Informative message if no edges

    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title(f"Concept Map for '{input_concept}' (Relevance Filtered, Threshold={relevance_threshold})")
    plt.savefig(f"concept_map_{input_concept}_relevance_filtered_thresh{relevance_threshold}.png")
    plt.close()


# --- Main Script (minor change to pass relevance_threshold) ---
if __name__ == "__main__":
    input_concept = input("Enter a philosophical concept: ")
    relevance_threshold = 0.2 # <--- Set a default threshold here, or let user input it

    print(f"Generating concept map for: '{input_concept}' (Relevance Filtered, Threshold={relevance_threshold})...")

    wiki_text = get_wikipedia_content(input_concept)
    if not wiki_text:
        print(f"Error: Could not retrieve Wikipedia page for '{input_concept}'.")
    else:
        extracted_concepts_ner = extract_concepts_ner(wiki_text)
        relationship_counts = extract_relationships_cooccurrence(input_concept, wiki_text, extracted_concepts_ner)

        generate_concept_map(input_concept, extracted_concepts_ner, relationship_counts, relevance_threshold=relevance_threshold) # Pass relevance_threshold
        print(f"Concept map (relevance filtered, threshold={relevance_threshold}) generated and saved as concept_map_{input_concept}_relevance_filtered_thresh{relevance_threshold}.png")