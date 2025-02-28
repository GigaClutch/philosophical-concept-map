# Philosophical Concept Map Generator - Deployment and Execution Guide

This guide provides instructions for deploying and running the Philosophical Concept Map Generator application.

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 4GB RAM or more recommended
- 500MB disk space for application and cache

### Required Python Packages

The following packages are required to run the application:

```
numpy>=1.26.0
matplotlib>=3.8.0
networkx>=3.2.0
spacy>=3.7.0
wikipedia>=1.4.0
scikit-learn>=1.3.0
pillow>=10.0.0
pyvis>=0.3.2
seaborn>=0.13.0
nltk>=3.8.1
```

## Installation

### 1. Clone the Repository (if applicable)

```bash
git clone https://github.com/yourusername/philosophical-concept-map.git
cd philosophical-concept-map
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## Application Structure

After installation, your directory structure should look like this:

```
philosophical-concept-map/
├── config.py
├── concept_map_refactored.py
├── error_handling.py
├── gui_refactored.py
├── logging_utils.py
├── performance_utils.py
├── test_concept_map.py
├── test_framework.py
├── style_guide.md
├── requirements.txt
├── data/
├── logs/
├── output/
└── wiki_cache/
```

## Running the Application

### GUI Mode

To start the application with the graphical user interface:

```bash
python gui_refactored.py
```

### Command Line Mode

To process a concept map from the command line:

```bash
python concept_map_refactored.py --concept "Ethics" --threshold 1.0 --output "./output"
```

Command line options:
- `--concept`: The philosophical concept to analyze
- `--threshold`: Relevance threshold for filtering concepts (default: 1.0)
- `--cache-dir`: Directory for caching Wikipedia content (default: "wiki_cache")
- `--output`: Directory to save results (default: "output")

## Running Tests

To run all tests:

```bash
python test_framework.py
```

To run specific tests:

```bash
python test_concept_map.py
```

## Configuration

The application configuration is stored in `config.py`. User-specific settings are saved in `user_config.json` in the application directory.

### Important Configuration Settings

- `DEFAULT_THRESHOLD`: Default relevance threshold for concepts
- `MAX_TEXT_LENGTH`: Maximum text length to process
- `MAX_CONCEPTS_DISPLAY`: Maximum number of concepts to display
- `CACHE_DIR`: Directory for Wikipedia content cache
- `OUTPUT_DIR`: Directory for saving results

## Troubleshooting

### Common Issues

1. **SpaCy Model Not Found**

   Error: `OSError: [E050] Can't find model 'en_core_web_sm'`

   Solution: Install the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Wikipedia API Errors**

   Error: `Wikipedia page not found` or `Disambiguation error`

   Solution: Try a more specific concept name or check your internet connection.

3. **GUI Display Issues**

   Problem: Visualization not showing or appearing truncated

   Solution: Resize the window or adjust the threshold slider to show fewer concepts.

### Logging

The application logs are stored in the `logs` directory. Check these files for detailed error information:

- `concept_mapper_TIMESTAMP.log`: Main application log
- `gui_TIMESTAMP.log`: GUI-specific log

## Performance Optimization

For large concepts with many relationships, consider:

1. Increasing the threshold value to display fewer concepts
2. Enabling disk caching for Wikipedia content
3. Running the application on a machine with more memory

## Data Management

The application stores data in several locations:

- `wiki_cache/`: Cached Wikipedia content to reduce API calls
- `output/`: Saved concept maps and related data
- `logs/`: Application logs for debugging

You can safely delete the cache directory to free up disk space, but this will require re-downloading content from Wikipedia.

## Advanced Usage

### Customizing the Visualization

You can modify visualization parameters in `config.py`:

- `FIGURE_SIZE`: Size of the figure (width, height)
- `NODE_SIZE`: Size of nodes in the graph
- `NODE_COLOR`: Color of nodes
- `EDGE_COLOR`: Color of edges

### Adding Custom Philosophical Terms

To add custom philosophical terms to detect, modify the `PHILOSOPHICAL_TERMS` list in `config.py`.

## Next Steps

After successful deployment, consider:

1. Exploring different philosophical concepts
2. Comparing related concepts by generating multiple maps
3. Adjusting the threshold to find the optimal visualization density
4. Saving and sharing your concept maps

For more information, refer to the project documentation or contact the project maintainers.