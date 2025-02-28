# test_comparator.py
from concept_comparator import ConceptComparator

print("Testing ConceptComparator...")
comparator = ConceptComparator()
print("Successfully created ConceptComparator instance!")

# Create dummy data
concept_data = {
    "Ethics": {
        "wiki_text": "Ethics is about morality and values.",
        "extracted_concepts": ["Morality", "Values", "Good", "Bad"]
    },
    "Justice": {
        "wiki_text": "Justice is about fairness and rights.",
        "extracted_concepts": ["Fairness", "Rights", "Good", "Law"]
    }
}

# Test comparison
print("Attempting comparison...")
results = comparator.compare_concepts(concept_data)
print("Comparison results:", results)