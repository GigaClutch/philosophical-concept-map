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
    try:
        # First try the enhanced GUI (if available)
        try:
            from gui_enhanced import start_gui
            start_gui()
            return
        except ImportError:
            pass
        
        # Then try the refactored GUI
        try:
            from gui_refactored import start_gui
            start_gui()
            return
        except ImportError:
            pass
        
        # Fall back to the original GUI
        from gui import ConceptMapApp
        import tkinter as tk
        root = tk.Tk()
        app = ConceptMapApp(root)
        root.mainloop()
    except ImportError as e:
        print(f"Error starting GUI: {e}")
        print("Unable to start any GUI version. Please check your installation.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def start_cli_basic():
    """Start the basic command line interface."""
    try:
        # Try the refactored version first
        try:
            from concept_map_refactored import main as concept_map_main
            concept_map_main()
        except ImportError:
            # Fall back to the original version
            from concept_map import main as concept_map_main
            concept_map_main()
    except ImportError as e:
        print(f"Error starting CLI: {e}")
        print("Unable to start the CLI. Please check your installation.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in CLI execution: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def start_cli_enhanced():
    """Start the enhanced command line interface."""
    try:
        from concept_map_integration import main as integration_main
        integration_main()
    except ImportError as e:
        print(f"Error starting enhanced CLI: {e}")
        print("Enhanced modules not available. Falling back to basic CLI.")
        start_cli_basic()
    except Exception as e:
        print(f"Error in enhanced CLI execution: {e}")
        print(traceback.format_exc())
        sys.exit(1)

def compare_concepts(concepts):
    """Compare multiple concepts."""
    try:
        from concept_map_integration import compare_concepts
        result = compare_concepts(concepts)
        
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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Philosophical Concept Map Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the GUI
  python main.py
  
  # Process a single concept
  python main.py --cli --concept "Ethics"
  
  # Use enhanced processing
  python main.py --cli --enhanced --concept "Ethics"
  
  # Compare multiple concepts
  python main.py --compare "Ethics" "Justice" "Metaphysics"
  
  # Process with specific options
  python main.py --cli --concept "Ethics" --threshold 0.5 --output "./results"
"""
    )
    
    # Basic mode arguments
    parser.add_argument("--gui", action="store_true", help="Run in GUI mode (default)")
    parser.add_argument("--cli", action="store_true", help="Run in command line mode")
    
    # Enhanced mode arguments
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced NLP and analysis")
    parser.add_argument("--compare", nargs='+', metavar="CONCEPT", help="Compare multiple concepts")
    
    # Concept processing arguments
    parser.add_argument("--concept", type=str, help="Philosophical concept to analyze")
    parser.add_argument("--threshold", type=float, help="Relevance threshold for filtering concepts")
    parser.add_argument("--output", type=str, help="Output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        return compare_concepts(args.compare)
    
    # Handle CLI mode
    if args.cli:
        # Modify the remaining arguments for the CLI
        cli_args = []
        if args.concept:
            cli_args.extend(["--concept", args.concept])
        if args.threshold is not None:
            cli_args.extend(["--threshold", str(args.threshold)])
        if args.output:
            cli_args.extend(["--output", args.output])
        
        # Set the arguments for the CLI
        sys.argv = [sys.argv[0]] + cli_args
        
        # Start the appropriate CLI
        if args.enhanced:
            start_cli_enhanced()
        else:
            start_cli_basic()
        return 0
    
    # Default to GUI mode
    start_gui()
    return 0

if __name__ == "__main__":
    sys.exit(main())