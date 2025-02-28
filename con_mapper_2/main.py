"""
Main entry point for the Philosophical Concept Map Generator.
"""
import os
import sys
import importlib

def setup_environment():
    """Set up the environment for the application."""
    # Ensure required directories exist
    directories = ["logs", "output", "wiki_cache", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Add the current directory to the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Check if __init__.py exists
    init_file = os.path.join(current_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Philosophical Concept Map Generator package."""\n')

def module_exists(module_name):
    """Check if a module exists without importing it."""
    return importlib.util.find_spec(module_name) is not None

def main():
    """Main entry point."""
    # Set up the environment
    setup_environment()
    
    # Check if enhanced modules are available
    if module_exists("main_enhanced_fixed"):
        # Use the enhanced main script
        print("Using enhanced features...")
        from main_enhanced_fixed import main as enhanced_main
        return enhanced_main()
    elif module_exists("concept_map_refactored"):
        # Use the refactored concept map
        print("Using refactored features...")
        from concept_map_refactored import main as refactored_main
        return refactored_main()
    elif module_exists("concept_map"):
        # Use the original concept map
        print("Using basic features...")
        from concept_map import main as original_main
        return original_main()
    else:
        # No modules available, use run_app as fallback
        if module_exists("run_app"):
            print("Using fallback app...")
            from run_app import main as run_app_main
            return run_app_main()
        else:
            print("Error: No valid modules found. Please run install.py first.")
            return 1

if __name__ == "__main__":
    sys.exit(main())