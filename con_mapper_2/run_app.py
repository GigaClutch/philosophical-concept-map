"""
Simple startup script for the Philosophical Concept Map Generator.

This script is designed to work with any version of the application,
whether the original, refactored, or enhanced modules are available.
"""
import os
import sys
import importlib
import traceback

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
        
    # Also add the parent directory if needed
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(current_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Philosophical Concept Map Generator package."""\n')

def module_exists(module_name):
    """Check if a module exists without importing it."""
    return importlib.util.find_spec(module_name) is not None

def start_gui():
    """Start the GUI application using the best available module."""
    print("Starting the Philosophical Concept Map Generator GUI...")
    
    # Try each GUI module in order of preference
    modules_to_try = [
        # GUI modules in order of preference
        ("gui_enhanced", "start_gui"),
        ("gui_refactored", "start_gui"),
        ("gui", "ConceptMapApp")
    ]
    
    for module_name, attr_name in modules_to_try:
        if module_exists(module_name):
            try:
                print(f"Using {module_name}...")
                module = importlib.import_module(module_name)
                
                if attr_name == "ConceptMapApp":
                    # Original GUI needs special handling
                    import tkinter as tk
                    root = tk.Tk()
                    app = getattr(module, attr_name)(root)
                    root.mainloop()
                else:
                    # Refactored or enhanced GUI
                    func = getattr(module, attr_name)
                    func()
                return True
            except Exception as e:
                print(f"Error using {module_name}: {e}")
                print(traceback.format_exc())
    
    print("Error: Could not start any GUI module.")
    print("Please run python verify_installation.py to check your installation.")
    return False

def process_concept(concept_name):
    """Process a single concept using the best available module."""
    print(f"Processing concept: {concept_name}")
    
    # Try each processing module in order of preference
    modules_to_try = [
        # Processing modules in order of preference
        ("concept_map_integration", "process_concept_enhanced"),
        ("concept_map_refactored", "process_concept"),
        ("concept_map", "process_concept")
    ]
    
    for module_name, func_name in modules_to_try:
        if module_exists(module_name):
            try:
                print(f"Using {module_name}...")
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                result = func(concept_name)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                
                print(f"Processing complete. Results saved to {result.get('result_directory', 'output')}")
                return True
            except AttributeError as e:
                print(f"Function {func_name} not found in {module_name}: {e}")
            except Exception as e:
                print(f"Error using {module_name}: {e}")
                print(traceback.format_exc())
    
    print("Error: Could not process concept with any available module.")
    print("Please run python verify_installation.py to check your installation.")
    return False

def main():
    """Main entry point for the application."""
    # Set up the environment
    setup_environment()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        concept_name = sys.argv[1]
        return process_concept(concept_name)
    else:
        # No arguments, start GUI
        return start_gui()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)