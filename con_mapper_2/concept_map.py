"""
Real data concept map implementation.
Retrieves and processes actual Wikipedia content instead of using synthetic data.
"""
import os
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import traceback
from pathlib import Path
import time
from collections import Counter

# Import required packages - make sure these are installed
try:
    import wikipedia
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Successfully imported all required packages")
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please run: pip install wikipedia spacy scikit-learn")
    print("And: python -m spacy download en_core_web_sm")
    raise

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please run: python -m spacy download en_core_web_sm")
    raise

# List of core philosophical concepts to help with filtering
CORE_PHILOSOPHICAL_CONCEPTS = [
    "philosophy", "ethics", "aesthetics", "epistemology", "logic", "metaphysics",
    "moral", "knowledge", "truth", "reality", "existence", "being", "mind", 
    "consciousness", "identity", "justice", "rights", "freedom", "liberty",
    "determinism", "free will", "dualism", "monism", "empiricism", "rationalism",
    "skepticism", "relativism", "absolutism", "utilitarianism", "deontology",
    "virtue ethics", "nihilism", "existentialism", "phenomenology"
]

def is_philosophical_term(term, min_length=3):
    """
    Check if a term is likely to be a philosophical concept.
    
    Args:
        term: Term to check
        min_length: Minimum length for terms
        
    Returns:
        Boolean indicating if term is likely philosophical
    """
    # Process the term
    term = term.lower().strip()
    
    # Length check
    if len(term) < min_length:
        return False
    
    # Check against core concepts
    for concept in CORE_PHILOSOPHICAL_CONCEPTS:
        if (concept in term) or (term in concept):
            return True
    
    return False

def get_wikipedia_content(concept_name, cache_dir="wiki_cache"):
    """
    Get Wikipedia content for a concept with caching.
    
    Args:
        concept_name: The concept to retrieve
        cache_dir: Directory for caching Wikipedia content
        
    Returns:
        Wikipedia article content or error message
    """
    print(f"Getting Wikipedia content for '{concept_name}'")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a safe filename
    safe_name = "".join(c for c in concept_name if c.isalnum() or c in [' ', '_']).rstrip().replace(' ', '_')
    cache_file = os.path.join(cache_dir, f"{safe_name}.txt")
    
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading cached content for '{concept_name}'")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # If it starts with ERROR:, this was a failed lookup
                if content.startswith("ERROR:"):
                    print(f"Cached error for '{concept_name}': {content}")
                    return content
                
                print(f"Loaded {len(content)} characters from cache")
                return content
        except Exception as e:
            print(f"Error reading cache: {e}")
            # Continue to fetch from Wikipedia
    
    # Fetch from Wikipedia
    try:
        print(f"Fetching Wikipedia content for '{concept_name}'")
        
        # Try with auto_suggest=False first for exact match
        try:
            page = wikipedia.page(concept_name, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            # If disambiguation error, try with auto_suggest=True
            print(f"Disambiguation page found for '{concept_name}', trying auto-suggest")
            try:
                page = wikipedia.page(concept_name, auto_suggest=True)
            except Exception as inner_e:
                # If that also fails, try the first disambiguation option
                options = e.options[:5]  # Get first 5 options
                print(f"Options: {options}")
                
                for option in options:
                    try:
                        page = wikipedia.page(option, auto_suggest=False)
                        print(f"Using disambiguation option: {option}")
                        break
                    except:
                        continue
                else:
                    # If all options fail, raise the original error
                    raise e
        
        # Get content
        content = page.content
        print(f"Retrieved {len(content)} characters from Wikipedia page: {page.title}")
        
        # Cache the content
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error caching content: {e}")
        
        return content
        
    except wikipedia.exceptions.PageError:
        error_msg = f"ERROR: Wikipedia page not found for '{concept_name}'"
        print(error_msg)
        
        # Cache the error
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
        except Exception as e:
            print(f"Error caching error message: {e}")
        
        return error_msg
        
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]  # Get first 5 options
        error_msg = f"ERROR: Disambiguation error for '{concept_name}'. Options: {options}"
        print(error_msg)
        
        # Cache the error
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
        except Exception as e:
            print(f"Error caching error message: {e}")
        
        return error_msg
        
    except Exception as e:
        error_msg = f"ERROR: Failed to retrieve Wikipedia content for '{concept_name}': {e}"
        print(error_msg)
        
        # Cache the error
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(error_msg)
        except Exception as e:
            print(f"Error caching error message: {e}")
        
        return error_msg

def extract_all_concepts(wiki_text, primary_concept=None, max_length=20000):
    """
    Extract philosophical concepts from Wikipedia text.
    
    Args:
        wiki_text: The Wikipedia article text
        primary_concept: The main concept being analyzed
        max_length: Maximum text length to process
        
    Returns:
        List of dictionaries with concept information
    """
    # Check for error message
    if wiki_text.startswith("ERROR:"):
        print(f"Cannot extract concepts: {wiki_text}")
        return []
    
    print("Extracting concepts from Wikipedia text")
    
    try:
        # Process a limited chunk of text for performance
        limited_text = wiki_text[:max_length]
        
        # Process with spaCy
        doc = nlp(limited_text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "NORP", "GPE"]:
                entities.append(ent.text)
        
        # Extract noun chunks (potential concepts)
        noun_chunks = []
        for chunk in doc.noun_chunks:
            # Filter out chunks with stopwords only
            if not all(token.is_stop for token in chunk):
                # Remove determiners and keep the core concept
                filtered_chunk = " ".join([token.text for token in chunk 
                                        if token.pos_ != "DET"])
                if filtered_chunk:
                    noun_chunks.append(filtered_chunk)
        
        # Create a set of candidate concepts from entities and noun chunks
        candidates = set(entities + noun_chunks)
        
        # Filter candidates
        filtered_candidates = []
        for candidate in candidates:
            # Remove short or trivial candidates
            if len(candidate) < 3:
                continue
                
            # Remove candidates that are just numbers or punctuation
            if all(c.isdigit() or c in ".,;:!?()[]{}\"'" or c.isspace() for c in candidate):
                continue
                
            # Keep philosophical terms
            filtered_candidates.append(candidate)
        
        # Count occurrences (frequency) in the text
        concept_counts = Counter()
        for candidate in filtered_candidates:
            # Convert to lowercase for case-insensitive counting
            count = len(re.findall(r'\b' + re.escape(candidate) + r'\b', wiki_text, re.IGNORECASE))
            concept_counts[candidate] = count
        
        # Extract TF-IDF scores for relevance (if we had multiple documents)
        # For single document, we'll use a combination of frequency and philosophical relevance
        
        # Calculate a philosophical relevance score
        philosophical_score = {}
        for candidate in filtered_candidates:
            # Base score on whether it contains a core philosophical term
            score = 0.0
            for term in CORE_PHILOSOPHICAL_CONCEPTS:
                if term in candidate.lower() or candidate.lower() in term:
                    score = 0.7  # Strong match
                    break
            
            # If no match with core terms, use frequency to determine score
            if score == 0.0:
                norm_freq = min(concept_counts[candidate] / 10.0, 1.0)  # Normalize to max of 1.0
                score = 0.3 * norm_freq  # Lower base score for non-philosophical terms
            
            philosophical_score[candidate] = score
        
        # Combine scores
        final_concepts = []
        for candidate in filtered_candidates:
            # Calculate final relevance score
            frequency = concept_counts[candidate]
            phil_score = philosophical_score[candidate]
            
            # Final score is weighted combination
            relevance = 0.6 * phil_score + 0.4 * min(frequency / 10.0, 1.0)
            
            # Add to final list if relevant enough
            if frequency > 1 and relevance > 0.1:
                final_concepts.append({
                    "concept": candidate,
                    "relevance_score": relevance,
                    "frequency": frequency,
                    "philosophical_score": phil_score
                })
        
        # Sort by relevance score and take top concepts
        sorted_concepts = sorted(final_concepts, key=lambda x: x["relevance_score"], reverse=True)
        
        # Filter to a reasonable number and update relevance scores to spread between 0.3 and 1.0
        top_limit = 20
        result = sorted_concepts[:top_limit]
        
        # Add the primary concept with full relevance if it's not there
        if primary_concept and primary_concept not in [c["concept"] for c in result]:
            result.insert(0, {
                "concept": primary_concept,
                "relevance_score": 1.0,
                "frequency": max([c["frequency"] for c in result]) if result else 10,
                "philosophical_score": 1.0
            })
        
        print(f"Extracted {len(result)} concepts")
        return result
        
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        print(traceback.format_exc())
        return []

def extract_rich_relationships(input_concept, wiki_text, extracted_concepts):
    """
    Extract relationships between concepts based on co-occurrence.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        
    Returns:
        Dictionary of relationships between concept pairs
    """
    # Check for error message
    if wiki_text.startswith("ERROR:"):
        print(f"Cannot extract relationships: {wiki_text}")
        return {}
    
    if not extracted_concepts:
        print("No concepts to analyze relationships")
        return {}
    
    print(f"Extracting relationships for '{input_concept}'")
    
    try:
        # Process text with spaCy for sentence segmentation
        doc = nlp(wiki_text[:20000])  # Limit to first 20K characters for performance
        sentences = list(doc.sents)
        
        # Extract concept strings from dictionaries
        concept_strings = []
        concept_dict = {}
        for c in extracted_concepts:
            if isinstance(c, dict):
                concept_strings.append(c["concept"])
                concept_dict[c["concept"]] = c
            else:
                concept_strings.append(c)
                concept_dict[c] = {"relevance_score": 0.5}
        
        # Create a lowercase mapping for case-insensitive matching
        concept_lower = {c.lower(): c for c in concept_strings}
        
        # Initialize relationship data
        relationship_data = {}
        
        # Process each sentence
        for sentence in sentences:
            sentence_text = sentence.text.lower()
            
            # Find concepts in this sentence
            concepts_in_sentence = []
            for concept_lower_text, original_concept in concept_lower.items():
                if concept_lower_text in sentence_text:
                    concepts_in_sentence.append(original_concept)
            
            # Create relationships between concepts in the same sentence
            for i, concept1 in enumerate(concepts_in_sentence):
                for concept2 in concepts_in_sentence[i+1:]:
                    # Create a unique key for this concept pair
                    pair_key = frozenset({concept1, concept2})
                    
                    # Skip self-relationships
                    if len(pair_key) < 2:
                        continue
                    
                    # Initialize data structure if this is the first occurrence
                    if pair_key not in relationship_data:
                        relationship_data[pair_key] = {
                            "count": 0,
                            "sentences": [],
                            "relationship_type": "co-occurs_with",
                            "is_directed": False,
                            "direction": None,  # None for undirected
                            "verbs": Counter()
                        }
                    
                    # Update relationship data
                    relationship_data[pair_key]["count"] += 1
                    
                    # Store sentence examples (up to 5)
                    if len(relationship_data[pair_key]["sentences"]) < 5:
                        relationship_data[pair_key]["sentences"].append(sentence.text)
                    
                    # Try to extract relationship verbs
                    for token in sentence:
                        if token.pos_ == "VERB":
                            relationship_data[pair_key]["verbs"][token.lemma_] += 1
        
        # Update relationship types based on verbs
        for pair_key, data in relationship_data.items():
            verbs = data["verbs"]
            
            # Define verb-to-relationship mapping
            is_a_verbs = ["be", "constitute", "represent", "define"]
            has_verbs = ["have", "contain", "include", "consist"]
            causal_verbs = ["cause", "lead", "result", "create", "produce"]
            oppose_verbs = ["oppose", "contradict", "disagree", "reject"]
            
            # Check for specific relationship types
            if any(v in is_a_verbs for v in verbs):
                data["relationship_type"] = "is_a"
                data["is_directed"] = True
            elif any(v in has_verbs for v in verbs):
                data["relationship_type"] = "has_part"
                data["is_directed"] = True
            elif any(v in causal_verbs for v in verbs):
                data["relationship_type"] = "influences"
                data["is_directed"] = True
            elif any(v in oppose_verbs for v in verbs):
                data["relationship_type"] = "contrasts_with"
                data["is_directed"] = False
        
        print(f"Extracted {len(relationship_data)} relationships")
        return relationship_data
        
    except Exception as e:
        print(f"Error extracting relationships: {e}")
        print(traceback.format_exc())
        return {}

def generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data):
    """
    Generate a summary of the concept map.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        relationship_data: Dictionary of relationships
        
    Returns:
        Markdown-formatted summary
    """
    # Check for error message
    if wiki_text.startswith("ERROR:"):
        error_msg = wiki_text.split("ERROR: ")[1] if "ERROR: " in wiki_text else wiki_text
        return f"# Error: {error_msg}\n\nUnable to generate concept map for '{input_concept}'."
    
    if not extracted_concepts:
        return f"# Error: No concepts found\n\nUnable to generate concept map for '{input_concept}'."
    
    print(f"Generating summary for '{input_concept}'")
    
    try:
        # Sort concepts by relevance
        sorted_concepts = sorted(
            [c for c in extracted_concepts if isinstance(c, dict) and c["concept"] != input_concept],
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        
        # Start building the summary
        summary = [
            f"# Concept Map Summary for '{input_concept}'",
            f"\nThe concept map for '{input_concept}' contains {len(extracted_concepts)} related concepts and {len(relationship_data)} relationships."
        ]
        
        # Add excerpt from Wikipedia
        first_paragraph = wiki_text.split('\n\n')[0]
        if len(first_paragraph) > 300:
            first_paragraph = first_paragraph[:297] + "..."
        
        summary.append(f"\n## Overview\n{first_paragraph}")
        
        # Add top concepts by relevance
        summary.append("\n## Top Related Concepts by Relevance")
        
        for concept in sorted_concepts[:10]:  # Top 10
            name = concept["concept"]
            score = concept.get("relevance_score", 0)
            freq = concept.get("frequency", 0)
            summary.append(f"- **{name}** (relevance: {score:.2f}, frequency: {freq})")
        
        # Add relationship types
        summary.append("\n## Relationship Types")
        relationship_types = {}
        
        for rel_key, rel_data in relationship_data.items():
            rel_type = rel_data.get("relationship_type", "co-occurs_with")
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1
        
        for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"- **{rel_type}**: {count} relationships")
        
        # Add example relationships
        summary.append("\n## Example Relationships")
        
        # Find relationships with the most occurrences
        top_relationships = sorted(
            relationship_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]  # Top 5
        
        for rel_key, rel_data in top_relationships:
            concepts_in_rel = list(rel_key)
            c1, c2 = concepts_in_rel[0], concepts_in_rel[1]
            count = rel_data["count"]
            rel_type = rel_data.get("relationship_type", "co-occurs_with")
            
            summary.append(f"- **{c1}** {rel_type} **{c2}** (co-occurs {count} times)")
            
            # Add an example sentence
            if rel_data["sentences"]:
                example = rel_data["sentences"][0]
                if len(example) > 100:
                    example = example[:97] + "..."
                summary.append(f"  - Example: \"{example}\"")
        
        return "\n".join(summary)
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        print(traceback.format_exc())
        return f"# Error generating summary\n\nAn error occurred while generating the summary for '{input_concept}'."

def save_results(input_concept, wiki_text, extracted_concepts, relationship_data, output_dir="output"):
    """
    Save results to files.
    
    Args:
        input_concept: The primary philosophical concept
        wiki_text: The Wikipedia article text
        extracted_concepts: List of extracted concepts
        relationship_data: Dictionary of relationships
        output_dir: Directory to save results
        
    Returns:
        Path to the concept directory
    """
    print(f"Saving results for '{input_concept}' to {output_dir}")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        concept_dir = os.path.join(output_dir, input_concept.replace(" ", "_"))
        os.makedirs(concept_dir, exist_ok=True)
        
        # Save wiki text
        with open(os.path.join(concept_dir, "wiki_text.txt"), 'w', encoding='utf-8') as f:
            f.write(wiki_text)
        
        # Save extracted concepts
        with open(os.path.join(concept_dir, "extracted_concepts.json"), 'w', encoding='utf-8') as f:
            json.dump(extracted_concepts, f, indent=2)
        
        # Save relationship data (converting frozenset keys to strings)
        serializable_relationship_data = {}
        for key, value in relationship_data.items():
            # Convert frozenset to a string representation for JSON serialization
            if isinstance(key, frozenset):
                string_key = str(sorted(list(key)))
                serializable_relationship_data[string_key] = value
            else:
                serializable_relationship_data[str(key)] = value
        
        with open(os.path.join(concept_dir, "relationships.json"), 'w', encoding='utf-8') as f:
            json.dump(serializable_relationship_data, f, indent=2)
        
        # Generate and save summary
        summary = generate_summary(input_concept, wiki_text, extracted_concepts, relationship_data)
        with open(os.path.join(concept_dir, "summary.md"), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"All results saved to {concept_dir}")
        return concept_dir
        
    except Exception as e:
        print(f"Error saving results: {e}")
        print(traceback.format_exc())
        return output_dir

def create_visualization(concept, relationship_data, extracted_concepts, threshold=0.0):
    """
    Create a visualization of the concept map.
    
    Args:
        concept: The primary philosophical concept
        relationship_data: Dictionary of relationships
        extracted_concepts: List of extracted concepts
        threshold: Minimum relevance score for concepts to include
        
    Returns:
        matplotlib Figure object
    """
    print(f"Creating visualization for '{concept}' with threshold {threshold}")
    
    try:
        # Check for invalid inputs
        if not extracted_concepts or not relationship_data:
            fig, ax = plt.subplots(figsize=(10, 8))
            if not extracted_concepts:
                msg = f"No concepts found for '{concept}'"
            else:
                msg = f"No relationships found for '{concept}'"
            
            ax.text(0.5, 0.5, msg, 
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        # Create a directed graph
        G = nx.Graph()
        
        # Process concepts
        concept_dict = {}
        for c in extracted_concepts:
            if isinstance(c, dict):
                concept_dict[c["concept"]] = c
        
        # Add the main concept node
        G.add_node(concept, type="main")
        
        # Filter concepts by threshold
        filtered_concepts = []
        for c in extracted_concepts:
            if isinstance(c, dict):
                if c["concept"] != concept and c.get("relevance_score", 0) >= threshold:
                    filtered_concepts.append(c)
        
        # Sort by relevance and limit to reasonable number for display
        filtered_concepts = sorted(filtered_concepts, 
                                 key=lambda x: x.get("relevance_score", 0), 
                                 reverse=True)[:15]  # Limit to 15
        
        # Add related concept nodes
        for c in filtered_concepts:
            G.add_node(c["concept"], 
                     type="related", 
                     relevance=c.get("relevance_score", 0.5),
                     frequency=c.get("frequency", 1))
        
        # Add edges between concepts based on relationships
        for pair_key, data in relationship_data.items():
            # Check if this is a frozenset key (relationship between two concepts)
            if isinstance(pair_key, frozenset):
                c1, c2 = list(pair_key)
                
                # Only add edges where both concepts are in the graph
                if c1 in G and c2 in G:
                    # Add the edge with relationship data
                    G.add_edge(c1, c2, 
                             weight=data["count"],
                             type=data.get("relationship_type", "co-occurs_with"),
                             directed=data.get("is_directed", False))
        
        # If no nodes pass the threshold, show message
        if len(G.nodes()) <= 1:  # Only the main concept
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 
                  f"No concepts found for '{concept}' with threshold {threshold}.\nTry lowering the threshold.", 
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a layout with the main concept in the center
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        # Force the main concept to be centered
        if concept in pos:
            center_pos = pos[concept]
            offset_x = 0.5 - center_pos[0]
            offset_y = 0.5 - center_pos[1]
            
            # Adjust all positions
            for node in pos:
                pos[node] = (pos[node][0] + offset_x, pos[node][1] + offset_y)
        
        # Draw the main concept
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[concept],
                             node_color="skyblue",
                             node_size=3000,
                             alpha=0.8,
                             ax=ax)
        
        # Draw related concepts with color based on relevance
        related_nodes = [n for n in G.nodes() if n != concept]
        if related_nodes:
            node_colors = [G.nodes[n].get("relevance", 0.5) for n in related_nodes]
            node_sizes = [1000 + 1000 * G.nodes[n].get("relevance", 0.5) for n in related_nodes]
            
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=related_nodes,
                                 node_color=node_colors,
                                 cmap=plt.cm.Blues,
                                 node_size=node_sizes,
                                 alpha=0.7,
                                 ax=ax)
        
        # Draw edges with width based on weight
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + 4 * (w / max_weight) for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos,
                             width=edge_widths,
                             alpha=0.6,
                             edge_color="gray",
                             ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
        
        # Add legend for relationship types
        rel_types = set(nx.get_edge_attributes(G, "type").values())
        if rel_types:
            # Create a legend text
            legend_text = "Relationship Types:\n"
            for rt in rel_types:
                legend_text += f"- {rt}\n"
            
            # Add the legend to the plot
            plt.figtext(0.05, 0.05, legend_text, fontsize=10)
        
        # Set title
        plt.title(f"Concept Map for '{concept}' (Threshold: {threshold:.1f})")
        plt.axis("off")
        
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print(traceback.format_exc())
        
        # Return an error figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 
              f"Error creating visualization for '{concept}':\n{str(e)}", 
              horizontalalignment='center', verticalalignment='center',
              fontsize=12, color='red')
        ax.axis('off')
        return fig

def process_concept(concept, cache_dir=None, output_dir=None, threshold=None):
    """
    Process a philosophical concept through the entire pipeline.
    
    Args:
        concept: The philosophical concept to process
        cache_dir: Directory for caching Wikipedia content
        output_dir: Directory to save results
        threshold: Minimum relevance threshold for concepts
        
    Returns:
        Dictionary with processing results
    """
    # Set default directories if not specified
    if cache_dir is None:
        cache_dir = "wiki_cache"
    
    if output_dir is None:
        output_dir = "output"
    
    if threshold is None:
        threshold = 0.0
    
    print(f"Processing concept '{concept}' with threshold {threshold}")
    
    try:
        # Step 1: Get Wikipedia content
        wiki_text = get_wikipedia_content(concept, cache_dir=cache_dir)
        
        # Check for error
        if wiki_text.startswith("ERROR:"):
            return {
                "concept": concept,
                "error": wiki_text,
                "visualization": create_visualization(concept, {}, [], threshold)
            }
        
        # Step 2: Extract concepts
        extracted_concepts = extract_all_concepts(wiki_text, concept)
        
        # Step 3: Extract relationships
        relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
        
        # Step 4: Save results
        result_dir = save_results(concept, wiki_text, extracted_concepts, relationship_data, output_dir)
        
        # Step 5: Create visualization
        visualization = create_visualization(concept, relationship_data, extracted_concepts, threshold)
        
        return {
            "concept": concept,
            "wiki_text": wiki_text,
            "extracted_concepts": extracted_concepts,
            "relationships": relationship_data,
            "result_directory": result_dir,
            "visualization": visualization
        }
        
    except Exception as e:
        print(f"Error processing concept '{concept}': {e}")
        print(traceback.format_exc())
        
        # Return error state
        return {
            "concept": concept,
            "error": f"Error processing concept: {e}",
            "visualization": create_visualization(concept, {}, [], threshold)
        }

def main():
    """Test the implementation with a few concepts."""
    test_concepts = ["Ethics", "Justice", "Metaphysics", "Epistemology"]
    
    for concept in test_concepts:
        print(f"\nTesting concept: {concept}")
        result = process_concept(concept, threshold=0.2)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success: Found {len(result['extracted_concepts'])} concepts and {len(result['relationships'])} relationships")
            
            # Save the visualization
            plt.savefig(f"{concept}_concept_map.png")
            print(f"Visualization saved to {concept}_concept_map.png")
    
    print("\nTesting complete.")
    return 0

if __name__ == "__main__":
    main()