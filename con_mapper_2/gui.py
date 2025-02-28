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

# Import functions from concept_map.py
try:
    from concept_map import (
        get_wikipedia_content,
        extract_all_concepts,
        extract_rich_relationships,
        generate_summary,
        save_results
    )
except ImportError as e:
    print(f"Error importing from concept_map.py: {e}")
    # Define placeholder functions for testing
    def get_wikipedia_content(concept):
        return f"Placeholder text for {concept}"
    
    def extract_all_concepts(wiki_text):
        return ["Concept1", "Concept2", "Concept3"]
    
    def extract_rich_relationships(concept, wiki_text, extracted_concepts):
        return {}
    
    def generate_summary(concept, wiki_text, extracted_concepts, relationship_data):
        return f"Summary for {concept}"
    
    def save_results(concept, wiki_text, extracted_concepts, relationship_data, output_dir):
        return output_dir

class ConceptMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Philosophical Concept Map Generator")
        self.root.geometry("1200x800")
        
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
        self.current_threshold = 1
        self.extracted_concepts = []
        self.relationship_data = {}
        self.wiki_text = ""
        
        # Create a queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Set up polling mechanism to handle events from background threads
        self.root.after(100, self.process_queue)
    
    def setup_input_section(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        # Concept input
        ttk.Label(input_frame, text="Enter Philosophical Concept:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.concept_entry = ttk.Entry(input_frame, width=30)
        self.concept_entry.grid(row=0, column=1, pady=5, padx=5)
        self.concept_entry.insert(0, "Ethics")
        
        # Add generate button
        self.generate_btn = ttk.Button(input_frame, text="Generate Concept Map", 
                                command=self.generate_concept_map)
        self.generate_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Threshold slider
        threshold_frame = ttk.Frame(input_frame)
        threshold_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        ttk.Label(threshold_frame, text="Relevance Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=1.0)
        threshold_slider = ttk.Scale(threshold_frame, from_=0.0, to=10.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add a threshold value label
        self.threshold_label = ttk.Label(threshold_frame, text="1.0")
        self.threshold_label.pack(side=tk.RIGHT, padx=5)
        
        # Update the threshold label when the slider changes
        self.threshold_var.trace_add("write", self.update_threshold_label)
        
        # Log area
        log_frame = ttk.LabelFrame(input_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Save button
        self.save_btn = ttk.Button(input_frame, text="Save Results", 
                                 command=self.save_results)
        self.save_btn.grid(row=4, column=0, columnspan=2, pady=10)
        self.save_btn.config(state=tk.DISABLED)
    
    def setup_output_section(self):
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Concept Map Visualization", padding="10")
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        self.canvas_frame = ttk.Frame(self.output_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.placeholder_label = ttk.Label(self.canvas_frame, 
                                        text="Generate a concept map to see visualization here")
        self.placeholder_label.pack(expand=True)
    
    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{self.threshold_var.get():.1f}")
        
        # If we have data already, update the visualization
        if hasattr(self, 'relationship_data') and self.relationship_data:
            # Queue the update instead of doing it directly
            self.queue.put(("update_viz", None))
    
    def log(self, message):
        # Use the queue to log from background threads
        self.queue.put(("log", message))
    
    def _log_message(self, message):
        """Internal method to actually update the log from the main thread"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
    
    def process_queue(self):
        """Process the event queue to handle threading"""
        try:
            while True:
                action, data = self.queue.get_nowait()
                
                if action == "log":
                    self._log_message(data)
                elif action == "data_ready":
                    self.create_visualization(self.concept_entry.get().strip())
                    self.save_btn.config(state=tk.NORMAL)
                    self.generate_btn.config(state=tk.NORMAL)
                elif action == "update_viz":
                    self.create_visualization(self.concept_entry.get().strip())
                elif action == "error":
                    messagebox.showerror("Error", data)
                    self.generate_btn.config(state=tk.NORMAL)
                
                self.queue.task_done()
        except queue.Empty:
            # If queue is empty, schedule to check again
            self.root.after(100, self.process_queue)
    
    def generate_concept_map(self):
        concept = self.concept_entry.get().strip()
        if not concept:
            messagebox.showerror("Error", "Please enter a philosophical concept")
            return
        
        self.log(f"Starting to generate concept map for: '{concept}'")
        self.generate_btn.config(state=tk.DISABLED)
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=self.process_concept_map, args=(concept,), daemon=True).start()
    
    def process_concept_map(self, concept):
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
            import traceback
            error_text = traceback.format_exc()
            self.log(f"Error: {str(e)}")
            self.log(error_text)
            self.queue.put(("error", f"An error occurred: {str(e)}"))
    
    def create_visualization(self, concept):
        """Create visualization in the main thread"""
        # Remove previous visualization
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Get the threshold from the slider
        threshold = self.threshold_var.get()
        
        try:
            # Create a figure for the concept map
            fig = plt.figure(figsize=(8, 6))
            
            # Create graph
            G = nx.Graph()
            G.add_node(concept)
            
            # Calculate relevance scores for concepts
            concept_relevance = {}
            for pair_key, data in self.relationship_data.items():
                pair_list = list(pair_key)
                if pair_list[0] == concept:
                    concept_relevance[pair_list[1]] = data["count"]
                elif pair_list[1] == concept:
                    concept_relevance[pair_list[0]] = data["count"]
            
            # Filter concepts by threshold and limit to top 15 for clarity
            filtered_concepts = [(c, score) for c, score in concept_relevance.items() if score >= threshold]
            top_concepts = sorted(filtered_concepts, key=lambda x: x[1], reverse=True)[:15]
            
            # Calculate positions in a circular layout
            positions = {concept: (0.5, 0.5)}  # Center the main concept
            
            for i, (related_concept, score) in enumerate(top_concepts):
                angle = 2 * math.pi * i / len(top_concepts) if top_concepts else 0
                x = 0.5 + 0.4 * math.cos(angle)
                y = 0.5 + 0.4 * math.sin(angle)
                positions[related_concept] = (x, y)
                G.add_node(related_concept)
                G.add_edge(concept, related_concept, weight=score)
            
            # Draw the graph
            nx.draw_networkx_nodes(G, positions, 
                                  node_color="skyblue", 
                                  node_size=3000, 
                                  alpha=0.8)
            
            # Draw edges with width based on weight
            if G.edges():
                edge_weights = [G[u][v]['weight']/2 for u, v in G.edges()]
                nx.draw_networkx_edges(G, positions, 
                                     width=edge_weights, 
                                     alpha=0.5, 
                                     edge_color="gray")
            
            # Draw labels
            nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold')
            
            plt.title(f"Concept Map for '{concept}'")
            plt.axis('off')
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store the graph for saving
            self.concept_graph = fig
            
            self._log_message(f"Visualization created for '{concept}' with threshold {threshold:.1f}")
            
        except Exception as e:
            import traceback
            self._log_message(f"Error creating visualization: {str(e)}")
            self._log_message(traceback.format_exc())
    
    def save_results(self):
        """Save all results to a directory."""
        if not hasattr(self, 'wiki_text') or not self.wiki_text:
            messagebox.showerror("Error", "No data to save. Generate a concept map first.")
            return
        
        concept = self.concept_entry.get().strip()
        output_dir = filedialog.askdirectory(title="Select directory to save results")
        
        if not output_dir:
            return  # User cancelled
        
        try:
            self._log_message(f"Saving results for '{concept}' to {output_dir}...")
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
            self._log_message(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Error saving results: {e}")

# For testing
if __name__ == "__main__":
    root = tk.Tk()
    app = ConceptMapApp(root)
    root.mainloop()