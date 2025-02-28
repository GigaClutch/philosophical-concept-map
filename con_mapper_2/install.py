"""
Installation script for the Philosophical Concept Map Generator.

This script sets up the necessary directory structure and copies files to the right places.
"""
import os
import sys
import shutil
import subprocess
import platform

# Files that should be in the installation directory
CORE_FILES = [
    "__init__.py",
    "main.py",
    "main_enhanced.py",
    "main_enhanced_fixed.py",
    "run_app.py",
    "verify_installation.py",
    "install.py"
]

# Enhanced modules
ENHANCED_MODULES = [
    "concept_extraction.py",
    "relationship_extraction.py",
    "concept_comparator.py",
    "concept_map_integration.py",
    "config.py",
    "error_handling.py",
    "logging_utils.py",
    "performance_utils.py"
]

# Original modules
ORIGINAL_MODULES = [
    "concept_map.py",
    "concept_map_refactored.py",
    "gui.py",
    "gui_refactored.py"
]

# Directories to create
DIRECTORIES = [
    "data",
    "logs",
    "output",
    "wiki_cache"
]

def print_header(title):
    """Print a section header."""
    print("=" * 80)
    print(title)
    print("=" * 80)

def create_directories():
    """Create the necessary directories."""
    print_header("Creating Directories")
    
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def check_files():
    """Check which files are present in the current directory."""
    print_header("Checking Files")
    
    all_files = CORE_FILES + ENHANCED_MODULES + ORIGINAL_MODULES
    missing_files = []
    
    for file in all_files:
        if os.path.exists(file):
            print(f"Found: {file}")
        else:
            print(f"Missing: {file}")
            missing_files.append(file)
    
    return missing_files

def create_init_file():
    """Create the __init__.py file if it doesn't exist."""
    print_header("Creating __init__.py")
    
    init_file = "__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Philosophical Concept Map Generator package."""\n\n')
            f.write('__version__ = "2.0.0"\n')
        print(f"Created file: {init_file}")
    else:
        print(f"File already exists: {init_file}")

def check_python_path():
    """Check if the current directory is in PYTHONPATH."""
    print_header("Checking PYTHONPATH")
    
    current_dir = os.path.abspath(os.getcwd())
    python_path = os.environ.get('PYTHONPATH', '')
    
    path_sep = ';' if platform.system() == 'Windows' else ':'
    python_path_dirs = python_path.split(path_sep)
    
    if current_dir in python_path_dirs:
        print(f"Current directory is in PYTHONPATH: {current_dir}")
        return True
    else:
        print(f"Current directory is NOT in PYTHONPATH: {current_dir}")
        
        # Suggest command to add to PYTHONPATH
        if platform.system() == 'Windows':
            print("\nTo add to PYTHONPATH, run:")
            print(f'set PYTHONPATH=%PYTHONPATH%;{current_dir}')
        else:
            print("\nTo add to PYTHONPATH, run:")
            print(f'export PYTHONPATH=$PYTHONPATH:{current_dir}')
        
        return False

def install_dependencies():
    """Install required Python packages."""
    print_header("Installing Dependencies")
    
    dependencies = [
        "numpy",
        "matplotlib",
        "networkx",
        "spacy",
        "wikipedia",
        "scikit-learn"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("Successfully installed dependencies.")
        
        # Install spaCy model
        print("\nInstalling spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully installed spaCy model.")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def verify_installation():
    """Run the verification script if it exists."""
    print_header("Verifying Installation")
    
    if os.path.exists("verify_installation.py"):
        try:
            subprocess.check_call([sys.executable, "verify_installation.py"])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running verification script: {e}")
            return False
    else:
        print("Verification script not found.")
        return False

def main():
    """Main installation function."""
    print_header("Installing Philosophical Concept Map Generator")
    
    # Create directories
    create_directories()
    
    # Check PYTHONPATH
    check_python_path()
    
    # Offer to install dependencies
    print_header("Dependencies")
    install = input("Would you like to install required dependencies? (y/n): ")
    if install.lower() == 'y':
        install_dependencies()
    
    # Verify installation
    print_header("Verification")
    verify = input("Would you like to verify the installation? (y/n): ")
    if verify.lower() == 'y':
        verify_installation()
    
    # Final instructions
    print_header("Installation Complete")
    print("To run the application:")
    print("1. With GUI:")
    print("   python run_app.py")
    print("2. Process a concept:")
    print("   python run_app.py Ethics")
    print("3. For enhanced features (if available):")
    print("   python main_enhanced_fixed.py --concept Ethics --enhanced")
    print("4. For concept comparison (if available):")
    print("   python main_enhanced_fixed.py --compare Ethics Justice")
    
    print("\nIf you encounter any issues:")
    print("   python verify_installation.py")

if __name__ == "__main__":
    main()
    missing_files = check_files()
    
    # Create __init__.py
    create_init_file()
    
    # Check