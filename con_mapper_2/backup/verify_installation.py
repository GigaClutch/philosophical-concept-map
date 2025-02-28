"""
Verification script for Philosophical Concept Map Generator.

This script checks the installation and file structure to help troubleshoot issues.
"""
import os
import sys
import importlib.util
import platform

def print_separator():
    """Print a separator line."""
    print("-" * 80)

def print_header(title):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()

def check_file_exists(file_path):
    """Check if a file exists and return its info."""
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        modified = os.path.getmtime(file_path)
        from datetime import datetime
        modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
        return f"EXISTS - Size: {size} bytes, Modified: {modified_str}"
    else:
        return "NOT FOUND"

def check_directory_exists(dir_path):
    """Check if a directory exists and return the number of files in it."""
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        return f"EXISTS - Contains {len(files)} files/directories"
    else:
        return "NOT FOUND"

def check_module_can_import(module_name):
    """Try to import a module and return the result."""
    try:
        importlib.import_module(module_name)
        return "IMPORTABLE"
    except ImportError as e:
        return f"NOT IMPORTABLE - {str(e)}"
    except Exception as e:
        return f"ERROR - {str(e)}"

def print_system_info():
    """Print system information."""
    print_header("System Information")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path}")

def print_file_structure():
    """Print information about the file structure."""
    print_header("File Structure")
    
    # Core files
    core_files = [
        "concept_map.py",
        "concept_map_refactored.py",
        "concept_map_integration.py",
        "gui.py",
        "gui_refactored.py",
        "main.py",
        "main_enhanced.py",
        "__init__.py"
    ]
    
    print("Core Files:")
    for file in core_files:
        status = check_file_exists(file)
        print(f"  - {file}: {status}")
    
    # Enhanced modules
    enhanced_modules = [
        "concept_extraction.py",
        "relationship_extraction.py",
        "concept_comparator.py",
        "config.py",
        "error_handling.py",
        "logging_utils.py",
        "performance_utils.py"
    ]
    
    print("\nEnhanced Modules:")
    for file in enhanced_modules:
        status = check_file_exists(file)
        print(f"  - {file}: {status}")
    
    # Directories
    directories = [
        "data",
        "logs",
        "output",
        "wiki_cache"
    ]
    
    print("\nDirectories:")
    for directory in directories:
        status = check_directory_exists(directory)
        print(f"  - {directory}: {status}")

def check_imports():
    """Check if modules can be imported."""
    print_header("Import Checks")
    
    # Core modules
    core_modules = [
        "concept_map",
        "concept_map_refactored",
        "concept_map_integration",
        "gui",
        "gui_refactored"
    ]
    
    print("Core Modules:")
    for module in core_modules:
        status = check_module_can_import(module)
        print(f"  - {module}: {status}")
    
    # Enhanced modules
    enhanced_modules = [
        "concept_extraction",
        "relationship_extraction",
        "concept_comparator",
        "config",
        "error_handling",
        "logging_utils",
        "performance_utils"
    ]
    
    print("\nEnhanced Modules:")
    for module in enhanced_modules:
        status = check_module_can_import(module)
        print(f"  - {module}: {status}")
    
    # Required packages
    required_packages = [
        "spacy",
        "networkx",
        "matplotlib",
        "numpy",
        "sklearn",
        "wikipedia"
    ]
    
    print("\nRequired Packages:")
    for package in required_packages:
        status = check_module_can_import(package)
        print(f"  - {package}: {status}")

def check_spacy_model():
    """Check if the spaCy model is installed."""
    print_header("spaCy Model Check")
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' is installed and loaded successfully.")
        except OSError:
            print("spaCy model 'en_core_web_sm' is NOT installed.")
            print("Run 'python -m spacy download en_core_web_sm' to install it.")
    except ImportError:
        print("spaCy is not installed. Cannot check for model.")

def suggest_fixes():
    """Suggest fixes for common issues."""
    print_header("Suggestions for Common Issues")
    
    print("1. Module Import Issues:")
    print("   - Ensure all files are in the same directory")
    print("   - Create an empty __init__.py file in the directory")
    print("   - Add the directory to PYTHONPATH:")
    print("     export PYTHONPATH=$PYTHONPATH:/path/to/concept_mapper  # Linux/Mac")
    print("     set PYTHONPATH=%PYTHONPATH%;C:\\path\\to\\concept_mapper  # Windows")
    
    print("\n2. Missing Dependencies:")
    print("   - Install required packages:")
    print("     pip install numpy matplotlib networkx spacy wikipedia scikit-learn")
    print("   - Install spaCy model:")
    print("     python -m spacy download en_core_web_sm")
    
    print("\n3. File Not Found Errors:")
    print("   - Ensure all required directories exist (data, logs, output, wiki_cache)")
    print("   - Check file paths in config.py if using custom directories")
    
    print("\n4. GUI Issues:")
    print("   - Ensure tkinter is installed:")
    print("     apt-get install python3-tk  # Linux")
    print("     brew install python-tk  # Mac")
    print("     (Comes with standard Python installation on Windows)")

def main():
    """Run all verification checks."""
    print_header("Philosophical Concept Map Generator - Installation Verification")
    
    print_system_info()
    print_file_structure()
    check_imports()
    check_spacy_model()
    suggest_fixes()
    
    print_header("Verification Complete")
    print("If you continue to have issues, please report them with the above information.")

if __name__ == "__main__":
    main()