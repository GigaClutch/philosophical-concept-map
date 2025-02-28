# Philosophical Concept Map Generator - Project Structure

## Overview

This document describes the structure of the Philosophical Concept Map Generator project after the code structure enhancements. The project has been refactored to improve error handling, add a unified logging system, standardize naming conventions, and implement proper test cases.

## Directory Structure

```
philosophical_concept_map/
├── config.py                   # Centralized configuration settings
├── concept_map_refactored.py   # Refactored core functionality
├── error_handling.py           # Error handling utilities
├── gui_refactored.py           # Refactored GUI application
├── logging_utils.py            # Logging utilities
├── test_concept_map.py         # Unit tests for concept map functionality
├── test_framework.py           # Testing framework and utilities
├── data/                       # Directory for application data
├── logs/                       # Directory for log files
├── output/                     # Directory for saved results
├── wiki_cache/                 # Cache directory for Wikipedia content
├── tests/                      # Test directory
├── docs/                       # Documentation
└── style_guide.md              # Code style guidelines
```

## Module Descriptions

### 1. config.py

Centralizes all configuration settings for the application, including:
- File paths and directories
- Processing parameters
- Visualization settings
- User preferences

Provides a `Config` class for managing and persisting user settings.

### 2. concept_map_refactored.py

Contains the core functionality for the concept map generation pipeline:
- Fetching Wikipedia content
- Extracting philosophical concepts
- Analyzing relationships between concepts
- Generating summaries and visualizations
- Saving results to files

This module has been refactored to use proper error handling and logging.

### 3. error_handling.py

Implements a comprehensive error handling system:
- Custom exception classes for different error types
- Centralized error handling function
- Error severity levels
- Logging integration

### 4. logging_utils.py

Provides a unified logging system:
- Centralized log manager
- File and console logging
- Logging decorators for function execution
- UI logging integration

### 5. gui_refactored.py

Implements the graphical user interface:
- Input controls for concept entry and settings
- Visualization display
- Log viewer
- Menu system with file operations
- Theme support (light/dark)
- Recent concepts tracking

### 6. test_framework.py

Provides a testing framework for the application:
- Base test class with common setup and teardown
- Mock data and utilities
- Test cases for core functionality

### 7. test_concept_map.py

Implements specific unit tests for the concept map functionality:
- Wikipedia content fetching
- Concept extraction
- Relationship analysis
- Data saving
- Visualization creation

## Key Improvements

1. **Error Handling**
   - Custom exception classes for specific error types
   - Centralized error handling function
   - Proper error propagation and logging
   - User-friendly error messages

2. **Logging System**
   - Consistent logging across all modules
   - Log files with timestamps
   - Log levels (INFO, WARNING, ERROR, etc.)
   - UI integration for log display

3. **Configuration Management**
   - Centralized settings
   - User preference persistence
   - Environment-specific configuration

4. **Code Organization**
   - Consistent naming conventions
   - Proper documentation with docstrings
   - Logical module structure
   - Separation of concerns

5. **Testing Framework**
   - Unit tests for core functionality
   - Mock objects for external dependencies
   - Test utilities and helpers

6. **User Experience**
   - Improved error messages
   - Theme support
   - Recent concepts tracking
   - Help and about information

## Usage

### Running the Application

```python
# Start the GUI application
python gui_refactored.py

# Process a concept from the command line
python concept_map_refactored.py --concept "Ethics" --threshold 1.0
```

### Running Tests

```python
# Run all tests
python test_framework.py

# Run specific test cases
python test_concept_map.py
```

## Next Steps

1. Complete the implementation of all tests
2. Add additional visualization options
3. Implement comparison functionality between concepts
4. Integrate with additional data sources beyond Wikipedia
5. Add export options for different formats (PDF, SVG, JSON)