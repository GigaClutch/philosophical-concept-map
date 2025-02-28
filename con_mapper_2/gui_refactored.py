"""
Graphical User Interface for the Philosophical Concept Map Generator.

This module provides a user-friendly interface for generating, visualizing,
and managing philosophical concept maps.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import os
import time
import json
import math
import queue
import traceback
from pathlib import Path

# Import local modules
try:
    from config import config
except ImportError:
    # Create a simple config object if the module doesn't exist
    class SimpleConfig:
        def get(self, key, default=None):
            return default
            
        def get_recent_concepts(self):
            return []
            
        def update_user_setting(self, key, value):
            pass
            
        def clear_recent_concepts(self):
            pass
            
        def save_user_config(self):
            pass
    
    config = SimpleConfig()
    print("Warning: config.py not found. Using a simple configuration object.")

try:
    from error_handling import handle_error
except ImportError:
    # Simple error handler if the module doesn't exist
    def handle_error(error, raise_exception=False, log_traceback=True, ui_callback=None):
        print(f"Error: {error}")
        if log_traceback and isinstance(error, Exception):
            traceback.print_exc()
    
    print("Warning: error_handling.py not found. Using a simple error handler.")

try:
    from logging_utils import get_logger, log_to_ui
except ImportError:
    # Simple logger if the module doesn't exist
    def get_logger(name):
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def critical(self, msg): print(f"CRITICAL: {msg}")
        return SimpleLogger()
        
    def log_to_ui(message, level=None):
        print(f"UI LOG: {message}")
    
    print("Warning: logging_utils.py not found. Using a simple logger.")

# Initialize logger
logger = get_logger("gui")

# Import concept_map functions
concept_map_module = None
try:
    # First try the refactored module
    from concept_map_refactored import (
        get_wikipedia_content,
        extract_all_concepts,
        extract_rich_relationships,
        generate_summary,
        save_results,
        create_visualization
    )
    concept_map_module = "concept_map_refactored"
    logger.info("Using concept_map_refactored.py module")
except ImportError as e:
    logger.warning(f"Error importing from concept_map_refactored.py: {e}")
    
    # If import fails, try the original module
    try:
        from concept_map import (
            get_wikipedia_content,
            extract_all_concepts,
            extract_rich_relationships,
            generate_summary,
            save_results,
            create_visualization
        )
        concept_map_module = "concept_map"
        logger.info("Using original concept_map.py module")
    except ImportError as e:
        error_msg = f"Failed to import any concept_map module: {e}"
        logger.critical(error_msg)
        
        # Define stub functions to prevent crashes
        def get_wikipedia_content(concept, cache_dir=None):
            print(f"Stub function: get_wikipedia_content({concept})")
            return "Placeholder content - concept_map module not found"
            
        def extract_all_concepts(wiki_text):
            print("Stub function: extract_all_concepts")
            return ["Concept1", "Concept2", "Concept3"]
            
        def extract_rich_relationships(concept, wiki_text, extracted_concepts):
            print("Stub function: extract_rich_relationships")
            return {}
            
        def generate_summary(concept, wiki_text, extracted_concepts, relationship_data):
            print("Stub function: generate_summary")
            return f"Summary for {concept}"
            
        def save_results(concept, wiki_text, extracted_concepts, relationship_data, output_dir):
            print("Stub function: save_results")
            return output_dir
            
        def create_visualization(concept, relationship_data, extracted_concepts, threshold=1.0):
            print("Stub function: create_visualization")
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.text(0.5, 0.5, f"Placeholder: {concept}", 
                   horizontalalignment='center', verticalalignment='center')
            return fig


class ConceptMapApp:
    """Main application class for the Philosophical Concept Map Generator GUI."""
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title(f"Philosophical Concept Map Generator v{config.get('VERSION')}")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up theme
        self.style = ttk.Style()
        self.apply_theme(config.get("theme", "light"))
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input section
        self.setup_input_section()
        
        # Create output section
        self.setup_output_section()
        
        # Set up status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize variables
        self.concept_graph = None
        self.current_threshold = config.get("default_threshold", 1.0)
        self.extracted_concepts = []
        self.relationship_data = {}
        self.wiki_text = ""
        
        # Load recently used concepts
        self.recent_concepts = config.get_recent_concepts()
        
        # Create a queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Set up polling mechanism to handle events from background threads
        self.root.after(100, self.process_queue)
        
        # Set up cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Update the recent concepts menu
        self.update_recent_concepts_menu()
        
        # Show help on startup if configured
        if config.get("show_help_on_startup", True):
            self.show_help()
    
    def apply_theme(self, theme_name):
        """
        Apply the specified theme to the application.
        
        Args:
            theme_name: Name of the theme to apply ('light' or 'dark')
        """
        if theme_name == "dark":
            # Configure dark theme
            self.style.configure("TFrame", background="#333333")
            self.style.configure("TLabel", background="#333333", foreground="#ffffff")
            self.style.configure("TButton", background="#555555", foreground="#ffffff")
            self.style.configure("TLabelframe", background="#333333", foreground="#ffffff")
            self.style.configure("TLabelframe.Label", background="#333333", foreground="#ffffff")
            self.style.configure("TEntry", fieldbackground="#555555", foreground="#ffffff")
            self.style.configure("TScale", background="#333333", troughcolor="#555555")
            
            # Configure the root window
            self.root.configure(background="#333333")
        else:
            # Use default light theme
            self.style.theme_use('clam')
    
    def setup_input_section(self):
        """Set up the input section of the GUI."""
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        # Concept input
        input_area = ttk.Frame(input_frame)
        input_area.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_area, text="Enter Philosophical Concept:").pack(side=tk.LEFT, padx=5)
        
        self.concept_var = tk.StringVar()
        self.concept_var.set(config.get("last_concept", "Ethics"))
        
        self.concept_entry = ttk.Combobox(input_area, textvariable=self.concept_var, width=25)
        self.concept_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Set up recent concepts dropdown
        if hasattr(self, 'recent_concepts') and self.recent_concepts:
            self.concept_entry['values'] = self.recent_concepts
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Add generate button
        self.generate_btn = ttk.Button(button_frame, text="Generate Concept Map", 
                                command=self.generate_concept_map)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Add save button
        self.save_btn = ttk.Button(button_frame, text="Save Results", 
                                command=self.save_results)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.config(state=tk.DISABLED)
        
        # Threshold slider
        threshold_frame = ttk.LabelFrame(input_frame, text="Visualization Settings")
        threshold_frame.pack(fill=tk.X, pady=10, padx=5)
        
        slider_frame = ttk.Frame(threshold_frame)
        slider_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(slider_frame, text="Relevance Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=config.get("default_threshold", 1.0))
        threshold_slider = ttk.Scale(slider_frame, from_=0.0, to=10.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add a threshold value label
        self.threshold_label = ttk.Label(slider_frame, text=f"{self.threshold_var.get():.1f}")
        self.threshold_label.pack(side=tk.RIGHT, padx=5)
        
        # Update the threshold label when the slider changes
        self.threshold_var.trace_add("write", self.update_threshold_label)
        
        # Apply button for threshold changes
        apply_btn = ttk.Button(threshold_frame, text="Apply Threshold", 
                             command=self.apply_threshold)
        apply_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Log area
        log_frame = ttk.LabelFrame(input_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status indicators
        status_frame = ttk.Frame(input_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Create a menu
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Concept", command=self.open_concept)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        
        # Create recent concepts submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Concepts", menu=self.recent_menu)
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        
        # View menu
        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        
        self.theme_var = tk.StringVar()
        self.theme_var.set(config.get("theme", "light"))
        theme_menu.add_radiobutton(label="Light", variable=self.theme_var, value="light", 
                                  command=lambda: self.change_theme("light"))
        theme_menu.add_radiobutton(label="Dark", variable=self.theme_var, value="dark",
                                  command=lambda: self.change_theme("dark"))
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Help", command=self.show_help)
    
    def setup_output_section(self):
        """Set up the output section of the GUI."""
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Concept Map Visualization", padding="10")
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        self.canvas_frame = ttk.Frame(self.output_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.placeholder_label = ttk.Label(self.canvas_frame, 
                                        text="Generate a concept map to see visualization here")
        self.placeholder_label.pack(expand=True)
    
    def update_recent_concepts_menu(self):
        """Update the recent concepts menu."""
        # Check if the menu has been initialized
        if not hasattr(self, 'recent_menu'):
            return  # Menu not initialized yet, will be updated when it's created
            
        # Clear existing menu items
        self.recent_menu.delete(0, tk.END)
        
        if self.recent_concepts:
            for concept in self.recent_concepts:
                self.recent_menu.add_command(
                    label=concept, 
                    command=lambda c=concept: self.load_recent_concept(c)
                )
            
            self.recent_menu.add_separator()
            self.recent_menu.add_command(label="Clear Recent Concepts", 
                                       command=self.clear_recent_concepts)
        else:
            self.recent_menu.add_command(label="No Recent Concepts", state=tk.DISABLED)
    
    def load_recent_concept(self, concept):
        """
        Load a concept from the recent concepts list.
        
        Args:
            concept: The concept to load
        """
        self.concept_var.set(concept)
        self.generate_concept_map()
    
    def clear_recent_concepts(self):
        """Clear the recent concepts list."""
        config.clear_recent_concepts()
        self.recent_concepts = []
        self.update_recent_concepts_menu()
    
    def change_theme(self, theme_name):
        """
        Change the application theme.
        
        Args:
            theme_name: Name of the theme to apply
        """
        config.update_user_setting("theme", theme_name)
        self.apply_theme(theme_name)
    
    def show_about(self):
        """Show the about dialog."""
        about_message = f"""Philosophical Concept Map Generator v{config.get('VERSION')}

A tool for visualizing relationships between philosophical concepts.

This application helps explore philosophical ideas by generating
interactive concept maps based on Wikipedia content.

© 2023 - All rights reserved
"""
        messagebox.showinfo("About", about_message)
    
    def show_help(self):
        """Show the help dialog."""
        help_message = """Philosophical Concept Map Generator - Help

Getting Started:
1. Enter a philosophical concept in the text field (e.g., "Ethics", "Existentialism")
2. Click "Generate Concept Map" to create a visualization
3. Adjust the relevance threshold slider to control which concepts are displayed
4. Click "Apply Threshold" to update the visualization
5. Use "Save Results" to export the concept map and related data

Tips:
- Hover over nodes in the concept map to see more information
- The thickness of lines represents the strength of relationships
- Use the recent concepts menu to quickly access previous searches
- Try different thresholds to find the optimal visualization

For more information, visit the project documentation.
"""
        messagebox.showinfo("Help", help_message)
    
    def open_concept(self):
        """Open a concept from a saved file."""
        # Implementation to be added
        messagebox.showinfo("Not Implemented", "This feature is not yet implemented.")
    
    def update_threshold_label(self, *args):
        """Update the threshold label when the slider changes."""
        self.threshold_label.config(text=f"{self.threshold_var.get():.1f}")
    
    def apply_threshold(self):
        """Apply the threshold change and update the visualization."""
        threshold = self.threshold_var.get()
        
        if not hasattr(self, 'relationship_data') or not self.relationship_data:
            messagebox.showinfo("Information", "Generate a concept map first.")
            return
        
        # Update config
        config.update_user_setting("default_threshold", threshold)
        
        # Queue the update
        self.queue.put(("update_viz", threshold))
    
    def log(self, message):
        """
        Add a message to the log.
        
        Args:
            message: The message to log
        """
        # Use the queue to log from background threads
        self.queue.put(("log", message))
    
    def _log_message(self, message):
        """
        Internal method to actually update the log from the main thread.
        
        Args:
            message: The message to log
        """
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, timestamp + message + "\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
    
    def process_queue(self):
        """Process the event queue to handle threading."""
        try:
            while True:
                action, data = self.queue.get_nowait()
                
                if action == "log":
                    self._log_message(data)
                elif action == "data_ready":
                    self.create_visualization(self.concept_var.get().strip())
                    self.save_btn.config(state=tk.NORMAL)
                    self.generate_btn.config(state=tk.NORMAL)
                elif action == "update_viz":
                    self.create_visualization(self.concept_var.get().strip(), data)
                elif action == "error":
                    messagebox.showerror("Error", data)
                    self.generate_btn.config(state=tk.NORMAL)
                
                self.queue.task_done()
        except queue.Empty:
            # If queue is empty, schedule to check again
            self.root.after(100, self.process_queue)
    
    def generate_concept_map(self):
        """Generate a concept map for the entered concept."""
        concept = self.concept_var.get().strip()
        if not concept:
            messagebox.showerror("Error", "Please enter a philosophical concept")
            return
        
        # Add to recent concepts
        if concept not in self.recent_concepts:
            self.recent_concepts.insert(0, concept)
            if len(self.recent_concepts) > 10:
                self.recent_concepts = self.recent_concepts[:10]
            
            config.update_user_setting("recent_concepts", self.recent_concepts)
            config.update_user_setting("last_concept", concept)
            self.update_recent_concepts_menu()
        
        self.log(f"Starting to generate concept map for: '{concept}'")
        self.generate_btn.config(state=tk.DISABLED)
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=self.process_concept_map, args=(concept,), daemon=True).start()
    
    def process_concept_map(self, concept):
        """
        Process the concept map generation in a background thread.
        
        Args:
            concept: The concept to process
        """
        try:
            # Get Wikipedia content
            self.log("Getting Wikipedia content...")
            self.wiki_text = get_wikipedia_content(concept)
            
            if not self.wiki_text:
                self.queue.put(("error", f"Could not retrieve Wikipedia page for '{concept}'."))
                return
            
            # Extract concepts
            self.log("Extracting concepts...")
            self.extracted_concepts = extract_all_concepts(self.wiki_text)
            self.log(f"Found {len(self.extracted_concepts)} related concepts")
            
            # Extract relationships
            self.log("Analyzing relationships...")
            self.relationship_data = extract_rich_relationships(concept, self.wiki_text, self.extracted_concepts)
            self.log(f"Found {len(self.relationship_data)} relationships")
            
            # Signal the main thread that data is ready
            self.queue.put(("log", "Data processing complete. Creating visualization..."))
            self.queue.put(("data_ready", None))
            
        except Exception as e:
            error_text = traceback.format_exc()
            self.log(f"Error: {str(e)}")
            self.log(error_text)
            self.queue.put(("error", f"An error occurred: {str(e)}"))
            
            # Log the error
            logger.error(f"Error processing concept '{concept}': {e}")
            logger.error(error_text)
    
    def create_visualization(self, concept, threshold=None):
        """
        Create visualization in the main thread.
        
        Args:
            concept: The concept to visualize
            threshold: Optional threshold value (uses slider value if None)
        """
        # Remove previous visualization
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Get the threshold from the slider if not provided
        if threshold is None:
            threshold = self.threshold_var.get()
        
        try:
            # Create visualization
            fig = create_visualization(concept, self.relationship_data, 
                                      self.extracted_concepts, threshold)
            
            # Store the graph for saving
            self.concept_graph = fig
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self._log_message(f"Visualization created for '{concept}' with threshold {threshold:.1f}")
            
        except Exception as e:
            error_text = traceback.format_exc()
            self._log_message(f"Error creating visualization: {str(e)}")
            self._log_message(error_text)
            
            # Create an error message in the visualization area
            error_label = ttk.Label(self.canvas_frame, 
                                  text=f"Error creating visualization:\n{str(e)}")
            error_label.pack(expand=True)
            
            # Log the error
            logger.error(f"Error creating visualization for '{concept}': {e}")
            logger.error(error_text)
    
    def save_results(self):
        """Save all results to a directory."""
        if not hasattr(self, 'wiki_text') or not self.wiki_text:
            messagebox.showerror("Error", "No data to save. Generate a concept map first.")
            return
        
        concept = self.concept_var.get().strip()
        output_dir = filedialog.askdirectory(title="Select directory to save results")
        
        if not output_dir:
            return  # User cancelled
        
        try:
            self._log_message(f"Saving results for '{concept}' to {output_dir}...")
            
            # Save data
            save_path = save_results(concept, self.wiki_text, self.extracted_concepts, 
                                    self.relationship_data, output_dir)
            
            # Save the visualization if available
            if hasattr(self, 'concept_graph') and self.concept_graph:
                fig_path = os.path.join(output_dir, f"{concept.replace(' ', '_')}_concept_map.png")
                self.concept_graph.savefig(fig_path)
                self._log_message(f"Visualization saved to {fig_path}")
            
            self._log_message(f"All results saved successfully to {save_path}")
            messagebox.showinfo("Success", f"Results saved to {save_path}")
            
        except Exception as e:
            error_text = traceback.format_exc()
            self._log_message(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Error saving results: {e}")
            
            # Log the error
            logger.error(f"Error saving results for '{concept}': {e}")
            logger.error(error_text)
    
    def on_close(self):
        """Clean up and close the application."""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Save configuration
            if 'config' in globals():
                config.save_user_config()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        # Destroy the root window
        self.root.destroy()


def start_gui():
    """Start the GUI application."""
    try:
        root = tk.Tk()
        app = ConceptMapApp(root)
        root.mainloop()
    except Exception as e:
        logger.critical(f"Error starting GUI: {e}")
        logger.critical(traceback.format_exc())
        
        # Show error in a simple window if possible
        try:
            messagebox.showerror("Critical Error", 
                               f"The application encountered a critical error and cannot start:\n\n{e}")
        except:
            print(f"CRITICAL ERROR: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    start_gui()