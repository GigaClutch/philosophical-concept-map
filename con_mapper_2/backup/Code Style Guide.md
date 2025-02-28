# Philosophical Concept Map Generator - Code Style Guide

## General Principles

- Write clean, readable, and maintainable code
- Follow PEP 8 guidelines for Python code
- Document code thoroughly with docstrings and comments
- Use consistent naming and formatting across the codebase

## Python Standards

### Naming Conventions

- **Modules**: Use lowercase with underscores (snake_case)
  - Example: `concept_map.py`, `error_handling.py`

- **Classes**: Use CamelCase (CapWords)
  - Example: `ConceptMapper`, `WikipediaClient`

- **Functions and Methods**: Use lowercase with underscores (snake_case)
  - Example: `extract_concepts()`, `get_wikipedia_content()`

- **Variables**: Use lowercase with underscores (snake_case)
  - Example: `concept_data`, `relationship_map`

- **Constants**: Use uppercase with underscores
  - Example: `MAX_CONCEPTS`, `DEFAULT_THRESHOLD`

- **Private Members**: Prefix with a single underscore
  - Example: `_private_method()`, `_internal_data`

### File Organization

- Each Python file should follow this structure:
  1. Module docstring
  2. Imports (standard library, third-party, local)
  3. Constants
  4. Exception classes
  5. Classes
  6. Functions
  7. Main execution section (if applicable)

- Organize imports in the following order:
  ```python
  # Standard library imports
  import os
  import sys
  
  # Third-party imports
  import numpy as np
  import spacy
  
  # Local imports
  from .error_handling import handle_error
  from .logging_utils import get_logger
  ```

### Code Formatting

- Use 4 spaces for indentation (not tabs)
- Maximum line length: 88 characters (as per Black formatter)
- Use vertical whitespace (blank lines) to separate logical sections
- Use consistent spacing around operators and after commas

### Documentation

- All modules, classes, and public methods must have docstrings
- Use Google-style docstring format:
  ```python
  def example_function(param1, param2):
      """
      Brief description of function.
      
      Extended description of function if needed.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: When and why this exception is raised
      """
  ```

- Include type hints where appropriate:
  ```python
  def process_data(data: dict) -> list:
      """Process the input data."""
      # Function body
  ```

## Testing Standards

- All new features should include unit tests
- Test file names should match the module they test with `test_` prefix
- Use descriptive test method names that explain the expected behavior
- Each test should verify one specific aspect of functionality

## Git Workflow

- Commit messages should be clear and descriptive
- Use present tense in commit messages ("Add feature" not "Added feature")
- Reference issue numbers in commit messages when applicable

## Error Handling

- Use the centralized error handling system
- Avoid bare `except:` clauses
- Be specific about which exceptions to catch
- Always include meaningful error messages

## Logging

- Use the centralized logging system
- Log appropriate information at each level:
  - DEBUG: Detailed debugging information
  - INFO: Confirmation that things are working as expected
  - WARNING: Something unexpected happened but the application can continue
  - ERROR: Something failed but the application can still run
  - CRITICAL: Application failure requiring immediate attention

## UI Design

- Keep UI code separate from business logic
- Use consistent naming for UI elements
- Follow Model-View-Controller (MVC) pattern where appropriate

## Performance Considerations

- Be mindful of resource usage, especially with large datasets
- Cache results where appropriate
- Use efficient data structures for expected operations
- Consider threading for long-running operations