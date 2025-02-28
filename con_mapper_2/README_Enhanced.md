# Philosophical Concept Map Generator (Enhanced)

A powerful tool for visualizing and analyzing philosophical concepts and their relationships. This enhanced version includes advanced NLP and analysis capabilities for deeper insight into philosophical concepts.

## New Features in Version 2.0

### 1. Enhanced Concept Extraction
- Contextual relevance scoring for concepts
- Advanced NLP-based extraction using noun phrases and named entities
- Filtering mechanisms to reduce false positives
- Domain-specific concept identification

### 2. Semantic Relationship Analysis
- Extraction of relationship types (e.g., "is_a", "opposes", "influences")
- Detection of relationship directionality
- Pattern-based relationship identification
- Rich relationship metadata

### 3. Concept Comparison
- Compare multiple philosophical concepts
- Identify common and unique concepts
- Calculate similarity scores between concepts
- Visualize concept relationships

### 4. Advanced Visualization
- Color-coded nodes and edges based on relevance and relationship types
- Sized nodes based on relevance score
- More informative visual representation
- Interactive graph features

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/philosophical-concept-map.git
   cd philosophical-concept-map
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download spaCy language model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### GUI Mode

To start the application with a graphical user interface:

```bash
python main_enhanced.py
```

### Command Line Mode

To process a concept with enhanced features:

```bash
python main_enhanced.py --cli --enhanced --concept "Ethics" --threshold 0.3
```

### Concept Comparison

To compare multiple philosophical concepts:

```bash
python main_enhanced.py --compare "Ethics" "Justice" "Existentialism"
```

## Command Line Options

- `--gui`: Run in GUI mode (default)
- `--cli`: Run in command line mode
- `--enhanced`: Use enhanced NLP and analysis capabilities
- `--compare CONCEPT [CONCEPT ...]`: Compare multiple concepts
- `--concept CONCEPT`: Specify the philosophical concept to analyze
- `--threshold THRESHOLD`: Set the relevance threshold for filtering concepts (0.0-1.0)
- `--output DIRECTORY`: Specify the output directory for results

## Example Workflow

1. Generate concept maps for individual concepts:
   ```bash
   python main_enhanced.py --cli --enhanced --concept "Ethics" --output "./results"
   python main_enhanced.py --cli --enhanced --concept "Justice" --output "./results"
   ```

2. Compare the concepts:
   ```bash
   python main_enhanced.py --compare "Ethics" "Justice"
   ```

3. Review the comparison results in the output directory:
   - `comparison_results.json`: Detailed comparison data
   - `comparison_visualization.png`: Visual representation of the comparison
   - `comparison_summary.md`: Textual summary of the comparison

## Enhanced Analysis Outputs

The enhanced processing generates more detailed outputs:

1. **Concept Information**:
   - Extracted concepts with relevance scores
   - Frequency and domain information
   - TF-IDF scores and co-occurrence data

2. **Relationship Data**:
   - Relationship types and directionality
   - Example sentences demonstrating relationships
   - Most common verbs connecting concepts
   - Pattern-based relationship detection

3. **Visualization**:
   - Color-coded nodes based on relevance
   - Edge colors representing different relationship types
   - Node sizing based on importance
   - Clearer visual representation of complex relationships

4. **Comparison Results**:
   - Similarity matrix between concepts
   - Common and unique concepts
   - Shared relationships across concepts
   - Key themes across all compared concepts

## Project Structure

```
philosophical_concept_map/
├── concept_extraction.py         # Enhanced concept extraction
├── relationship_extraction.py    # Enhanced relationship extraction
├── concept_comparator.py         # Concept comparison functionality
├── concept_map_integration.py    # Integration of enhanced features
├── main_enhanced.py              # Enhanced main entry point
├── config.py                     # Configuration settings
├── error_handling.py             # Error handling utilities
├── logging_utils.py              # Logging system
├── performance_utils.py          # Performance optimization tools
├── data/                         # Data files for philosophical concepts
├── logs/                         # Log files
├── output/                       # Output directory for results
└── wiki_cache/                   # Cache for Wikipedia content
```

## Advanced Features

### Customizing Relationship Types

The system recognizes various relationship types between concepts:

- **Definitional**: is_a, part_of, defined_as
- **Causal**: causes, leads_to, influences
- **Oppositional**: opposes, contradicts, contrasts_with
- **Similarity**: similar_to, agrees_with, related_to
- **Temporal**: precedes, follows, contemporary_with

You can customize the detection patterns in `relationship_extraction.py`.

### Adjusting Relevance Thresholds

The relevance scoring system takes into account:

- Frequency of mention
- TF-IDF score
- Domain relevance
- Co-occurrence with main concept

Adjust threshold values in the configuration to control which concepts are included.

### Building a Philosophical Dictionary

The system can build and use a philosophical dictionary for better domain-specific concept detection:

```bash
python concept_extraction.py --build-dictionary
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wikipedia for providing the content
- spaCy for natural language processing capabilities
- NetworkX for graph visualization
- scikit-learn for TF-IDF and similarity calculations