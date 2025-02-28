import wikipedia
import os
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    exit(1)

# Philosophical terms to detect
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

def main():
    # Test the full pipeline with a simple concept
    concept = "Ethics"
    print(f"Testing concept map generation for '{concept}'")
    
    # Get Wikipedia content
    wiki_text = get_wikipedia_content(concept)
    if not wiki_text:
        print(f"Failed to retrieve Wikipedia content for '{concept}'")
        return
    
    # Extract concepts
    extracted_concepts = extract_all_concepts(wiki_text)
    print(f"Found {len(extracted_concepts)} related concepts")
    print("Top concepts:", extracted_concepts[:10])
    
    # Extract relationships
    relationship_data = extract_rich_relationships(concept, wiki_text, extracted_concepts)
    print(f"Found {len(relationship_data)} relationships")
    
    # Print some relationships
    print("\nTop relationships:")
    for i, (pair_key, data) in enumerate(sorted(relationship_data.items(), key=lambda x: x[1]["count"], reverse=True)):
        if i >= 5:  # Only show top 5
            break
        pair_list = list(pair_key)
        other_concept = pair_list[0] if pair_list[1] == concept else pair_list[1]
        print(f"- {concept} and {other_concept}: mentioned together {data['count']} times")
        if data["sentences"]:
            sample_sentence = data["sentences"][0]
            if len(sample_sentence) > 100:
                sample_sentence = sample_sentence[:97] + "..."
            print(f"  Example: \"{sample_sentence}\"")

if __name__ == "__main__":
    main()