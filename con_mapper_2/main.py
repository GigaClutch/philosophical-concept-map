import sys
import os
import importlib.util
import subprocess

def module_exists(module_name):
    """
    Check if a module exists by attempting to import it.
    
    Args:
        module_name (str): Name of the module to check.
    
    Returns:
        bool: True if module exists, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """
    Install required dependencies using pip.
    """
    dependencies = [
        'numpy', 
        'matplotlib', 
        'networkx', 
        'spacy', 
        'wikipedia', 
        'scikit-learn', 
        'pillow',
        'nltk',
        'pyvis',
        'seaborn'
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}")

def main():
    """
    Main entry point for the Philosophical Concept Mapper.
    """
    # Add the current directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # First, check and install dependencies
    install_dependencies()
    
    # Try to download spaCy language model
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
    
    # Import and run the primary application
    try:
        from concept_map import main as concept_map_main
        return concept_map_main()
    except ImportError:
        print("Could not import concept_map module. Checking alternatives...")
        
        # List alternative modules to try
        alternative_modules = [
            'simple_gui',
            'gui',
            'concept_map'
        ]
        
        for module_name in alternative_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'main'):
                    return module.main()
            except ImportError:
                print(f"Could not import {module_name}")
        
        print("No suitable main module found. Please check your project setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())