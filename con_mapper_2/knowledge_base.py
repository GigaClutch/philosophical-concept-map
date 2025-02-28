import json
import os
import pickle
from datetime import datetime

class PhilosophicalKnowledgeBase:
    """
    A knowledge base to store and retrieve philosophical concepts and their relationships.
    """
    def __init__(self, kb_directory="philosophical_kb"):
        self.kb_directory = kb_directory
        os.makedirs(kb_directory, exist_ok=True)
        self.concepts_dir = os.path.join(kb_directory, "concepts")
        os.makedirs(self.concepts_dir, exist_ok=True)
        self.relationships_file = os.path.join(kb_directory, "relationships.pkl")
        self.concept_metadata_file = os.path.join(kb_directory, "concept_metadata.json")
        
        # Initialize or load existing data
        self.relationships = self._load_relationships()
        self.concept_metadata = self._load_concept_metadata()
    
    def _load_relationships(self):
        """Load relationships from pickle file or initialize new dictionary."""
        if os.path.exists(self.relationships_file):
            with open(self.relationships_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _load_concept_metadata(self):
        """Load concept metadata from JSON file or initialize new dictionary."""
        if os.path.exists(self.concept_metadata_file):
            with open(self.concept_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_concept(self, concept_name, wiki_text, extracted_concepts, relationship_data):
        """Save a concept and its associated data to the knowledge base."""
        # Create concept directory if it doesn't exist
        concept_dir = os.path.join(self.concepts_dir, self._safe_filename(concept_name))
        os.makedirs(concept_dir, exist_ok=True)
        
        # Save wiki text
        with open(os.path.join(concept_dir, "wiki_text.txt"), 'w', encoding='utf-8') as f:
            f.write(wiki_text)
        
        # Save extracted concepts
        with open(os.path.join(concept_dir, "extracted_concepts.json"), 'w', encoding='utf-8') as f:
            json.dump(extracted_concepts, f, indent=2)
        
        # Update relationships
        for concept_pair, data in relationship_data.items():
            # Convert frozenset to string for serialization
            pair_key = str(sorted(list(concept_pair)))
            self.relationships[pair_key] = data
        
        # Save relationships
        with open(self.relationships_file, 'wb') as f:
            pickle.dump(self.relationships, f)
        
        # Update concept metadata
        self.concept_metadata[concept_name] = {
            "last_updated": datetime.now().isoformat(),
            "related_concepts_count": len(extracted_concepts),
            "relationships_count": len(relationship_data)
        }
        
        # Save concept metadata
        with open(self.concept_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.concept_metadata, f, indent=2)
    
    def get_concept(self, concept_name):
        """Retrieve concept data from the knowledge base."""
        concept_dir = os.path.join(self.concepts_dir, self._safe_filename(concept_name))
        
        if not os.path.exists(concept_dir):
            return None
        
        # Load wiki text
        with open(os.path.join(concept_dir, "wiki_text.txt"), 'r', encoding='utf-8') as f:
            wiki_text = f.read()
        
        # Load extracted concepts
        with open(os.path.join(concept_dir, "extracted_concepts.json"), 'r', encoding='utf-8') as f:
            extracted_concepts = json.load(f)
        
        # Find related relationships
        concept_relationships = {}
        for pair_key, data in self.relationships.items():
            # Convert string representation back to list
            pair_list = eval(pair_key)
            if concept_name in pair_list:
                concept_relationships[frozenset(pair_list)] = data
        
        return {
            "wiki_text": wiki_text,
            "extracted_concepts": extracted_concepts,
            "relationships": concept_relationships,
            "metadata": self.concept_metadata.get(concept_name, {})
        }
    
    def get_all_concepts(self):
        """Get a list of all concepts in the knowledge base."""
        return list(self.concept_metadata.keys())
    
    def _safe_filename(self, filename):
        """Convert a string to a safe filename."""
        return "".join(c for c in filename if c.isalnum() or c in [' ', '_']).rstrip().replace(' ', '_')