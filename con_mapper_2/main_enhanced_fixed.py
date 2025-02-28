"""
Enhanced main entry point for the Philosophical Concept Map Generator application.

This file serves as the primary entry point for running the application with the
enhanced NLP and analysis capabilities from Phase 2 of development.
"""
import os
import sys
import traceback
import argparse

def setup_environment():
    """Set up the environment for the application."""
    # Ensure required directories exist
    directories = ["logs", "output", "wiki_cache", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Add the current directory to the path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    # Also add the parent directory if needed
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

def check_module_exists(module_name):
    """Check if a module exists without importing it."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_packages = []
    
    try:
        import spacy
    except ImportError:
        missing_packages.append("spacy")
    
    try:
        import networkx
    except ImportError:
        missing_packages.append("networkx")
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import sklearn
    except ImportError:
        missing_packages.append("scikit-learn")
    
    try:
        import wikipedia
    except ImportError:
        missing_packages.append("wikipedia")
    
    # Check if spaCy model is installed
    if 'spacy' not in missing_packages:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' is not installed.")
            print("Run 'python -m spacy download en_core_web_sm' to install it.")
    
    if missing_packages:
        print("Warning: The following packages are missing and should be installed:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

def start_gui():
    """Start the GUI application."""
    print("Starting GUI application...")
    try:
        # First try the original GUI
        try:
            from gui import ConceptMapApp
            import tkinter as tk
            print("Using original GUI...")
            root = tk.Tk()
            app = ConceptMapApp(root)
            root.mainloop()
            return
        except ImportError as e:
            print(f"Could not import original GUI: {e}")
            pass
            
        # Then try the refactored GUI
        try:
            from gui_refactored import start_gui
            print("Using refactored GUI...")
            start_gui()
            return
        except ImportError as e:
            print(f"Could not import refactored GUI: {e}")
            pass
    except Exception as e:
        print(f"Error starting application: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def check_available_modules():
    """Check which modules are available in the system."""
    available_modules = {}
    
    # Check original modules
    available_modules["concept_map"] = check_module_exists("concept_map")
    available_modules["gui"] = check_module_exists("gui")
    
    # Check refactored modules
    available_modules["concept_map_refactored"] = check_module_exists("concept_map_refactored")
    available_modules["gui_refactored"] = check_module_exists("gui_refactored")
    
    # Check enhanced modules
    available_modules["concept_extraction"] = check_module_exists("concept_extraction")
    available_modules["relationship_extraction"] = check_module_exists("relationship_extraction")
    available_modules["concept_comparator"] = check_module_exists("concept_comparator")
    available_modules["concept_map_integration"] = check_module_exists("concept_map_integration")
    
    # Print available modules
    print("Available modules:")
    for module, available in available_modules.items():
        status = "Available" if available else "Not available"
        print(f"  - {module}: {status}")
    
    return available_modules

def process_single_concept(concept, threshold=None, output_dir=None, use_enhanced=True):
    """Process a single concept using the appropriate module."""
    print(f"Processing concept: {concept}")
    
    # Determine which module to use
    if use_enhanced and check_module_exists("concept_map_integration"):
        try:
            from concept_map_integration import process_concept_enhanced
            return process_concept_enhanced(concept, threshold=threshold, output_dir=output_dir)
        except ImportError as e:
            print(f"Error importing concept_map_integration: {e}")
            print("Falling back to basic processing...")
    
    # Fall back to basic processing
    if check_module_exists("concept_map_refactored"):
        try:
            from concept_map_refactored import process_concept
            return process_concept(concept, threshold=threshold, output_dir=output_dir)
        except ImportError as e:
            print(f"Error importing concept_map_refactored: {e}")
    
    if check_module_exists("concept_map"):
        try:
            from concept_map import process_concept
            return process_concept(concept, threshold=threshold, output_dir=output_dir)
        except (ImportError, AttributeError) as e:
            print(f"Error importing or using concept_map: {e}")
    
    # If we reach here, no modules are available
    print("Error: No processing modules available.")
    return {"error": "No processing modules available"}

def compare_concepts(concepts, output_dir=None):
    """Compare multiple concepts."""
    print(f"Comparing concepts: {', '.join(concepts)}")
    
    # Check if comparison module is available
    if not check_module_exists("concept_comparator") and not check_module_exists("concept_map_integration"):
        print("Error: Concept comparison requires enhanced modules that are not available.")
        return 1
    
    try:
        # Try to use concept_map_integration first
        if check_module_exists("concept_map_integration"):
            from concept_map_integration import compare_concepts as compare_func
            result = compare_func(concepts, output_dir=output_dir)
        else:
            # Fall back to concept_comparator
            from concept_comparator import compare_concepts_from_files
            result = compare_concepts_from_files(concepts, output_dir=output_dir)
        
        if "error" in result:
            print(f"Error comparing concepts: {result['error']}")
            return 1
            
        print(f"Comparison complete. Results saved to {result.get('result_directory', 'output')}")
        return 0
    except ImportError as e:
        print(f"Error importing comparison module: {e}")
        print("Concept comparison requires enhanced modules.")
        return 1
    except Exception as e:
        print(f"Error in concept comparison: {e}")
        print(traceback.format_exc())
        return 1

def main():
    """Main entry point."""
    # Set up the environment
    setup_environment()
    
    # Check dependencies
    check_dependencies()
    
    # Check available modules
    available_modules = check_available_modules()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Philosophical Concept Map Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the GUI
  python main_enhanced_fixed.py
  
  # Process a single concept
  python main_enhanced_fixed.py --concept "Ethics"
  
  # Use enhanced processing
  python main_enhanced_fixed.py --concept "Ethics" --enhanced
  
  # Compare multiple concepts
  python main_enhanced_fixed.py --compare "Ethics" "Justice" "Metaphysics"
  
  # Process with specific options
  python main_enhanced_fixed.py --concept "Ethics" --threshold 0.5 --output "./results"
"""
    )
    
    # Concept processing arguments
    parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
    parser.add_argument("--threshold", type=float, help="Relevance threshold for filtering concepts")
    parser.add_argument("--output", type=str, help="Output directory")
    
    # Mode arguments
    parser.add_argument("--gui", action="store_true", help="Run in GUI mode (default if no other options)")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced NLP and analysis")
    parser.add_argument("--compare", nargs='+', metavar="CONCEPT", help="Compare multiple concepts")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        return compare_concepts(args.compare, output_dir=args.output)
    
    # Handle single concept processing
    if args.concept:
        result = process_single_concept(
            args.concept, 
            threshold=args.threshold, 
            output_dir=args.output,
            use_enhanced=args.enhanced
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print(f"Processing complete. Results saved to {result.get('result_directory', 'output')}")
        return 0
    
    # Default to GUI mode if no specific command
    if not (args.concept or args.compare):
        start_gui()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())