# Philosophical Concept Map Generator

A tool for generating and analyzing visual concept maps from philosophical texts.

## Overview

This application creates interactive visual maps of philosophical concepts and their relationships, using data from Wikipedia. It helps philosophers, students, and researchers understand the connections between different philosophical ideas, schools of thought, and thinkers.

## Features

- **Data Extraction**: Fetches philosophical concept data from Wikipedia
- **Concept Recognition**: Identifies philosophical concepts using both Named Entity Recognition and a curated list of philosophical terms
- **Relationship Analysis**: Analyzes how concepts relate to each other based on co-occurrence and semantic analysis
- **Interactive Visualization**: Creates interactive, filterable concept maps with adjustable relevance thresholds
- **Comparative Analysis**: Compares multiple philosophical concepts to find similarities and differences
- **Knowledge Base**: Stores and retrieves concept data for future reference
- **GUI Interface**: User-friendly interface for analyzing concepts without code
- **Export Options**: Save concept maps and analysis in various formats

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/philosophical-concept-map.git
cd philosophical-concept-map
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Command Line Interface

```bash
# Generate a concept map for "Ethics"
python concept_map.py --concept "Ethics"

# Change the relevance threshold
python concept_map.py --concept "Justice" --threshold 2.0

# Create an interactive visualization
python concept_map.py --concept "Utilitarianism" --interactive

# Set custom output directory
python concept_map.py --concept "Existentialism" --output "my_results"
```

### Graphical User Interface

```bash
python concept_map.py --gui
```

The GUI allows you to:
- Enter a philosophical concept to analyze
- Adjust the relevance threshold using a slider
- View the concept map visualization
- Save the resulting map and analysis
- Explore concepts in the knowledge base

### Python API

```python
# Import the necessary modules
from concept_map import get_wikipedia_content, extract_all_concepts
from concept_map import extract_rich_relationships, generate_concept_map

# Get content for a philosophical concept
wiki_text = get_wikipedia_content("Ethics")

# Extract concepts
concepts = extract_all_concepts(wiki_text)

# Analyze relationships
relationships = extract_rich_relationships("Ethics", wiki_text, concepts)

# Generate and display a concept map
generate_concept_map("Ethics", concepts, relationships, relevance_threshold=1.5)
```

## Customization

### Adding Philosophical Terms

Edit the `philosophical_terms` list in `concept_map.py`:

```python
philosophical_terms = [
    "Ethics", "Metaphysics", "Epistemology",
    # Add your additional terms here
    "Phenomenology", "Hermeneutics", "Pragmatism"
]
```

### Concept Categories

Edit the `concept_categories` dictionary:

```python
concept_categories = {
    "Justice": "Concept",
    "Plato": "Philosopher",
    # Add your categorizations here
    "Hegel": "Philosopher",
    "Modernism": "Philosophical Movement"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
