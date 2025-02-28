import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import os
import math
import json

# Import the necessary functions from concept_map.py
from concept_map import (
    get_wikipedia_content,
    extract_all_concepts,
    extract_rich_relationships,
    save_results
)

class SimpleConceptMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Philosophical Concept Map Generator")
        self.root.geometry("1200x800")
        
        # Set up cleanup on closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Data variables
        self.wiki_text = None
        self.extracted_concepts = None
        self.relationship_data = None
        self.concept_graph = None
        
        # Create the main layout
        self.create_widgets()

    def on_close(self):
        """Handle window close event by cleaning up matplotlib resources"""
        plt.close('all')  # Close all matplotlib figures
        self.root.destroy()  # Then destroy the tkinter window
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Concept input
        ttk.Label(left_panel, text="Concept:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.concept_entry = ttk.Entry(left_panel, width=20)
        self.concept_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        self.concept_entry.insert(0, "Ethics")
        
        # Threshold
        ttk.Label(left_panel, text="Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=1.0)
        threshold_scale = ttk.Scale(left_panel, from_=0, to=10, variable=self.threshold_var,
                                  orient=tk.HORIZONTAL, length=150)
        threshold_scale.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Map", 
                                     command=self.generate_map)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Results", 
                                  command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Log
        ttk.Label(left_panel, text="Log:").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.log_text = scrolledtext.ScrolledText(left_panel, height=20, width=36)
        self.log_text.grid(row=4, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        
        # Right panel - Visualization
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial message
        self.placeholder = ttk.Label(self.right_panel, 
                                   text="Click 'Generate Map' to create a concept map")
        self.placeholder.pack(expand=True)
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def generate_map(self):
        # Get input
        concept = self.concept_entry.get().strip()
        if not concept:
            messagebox.showerror("Error", "Please enter a concept name")
            return
        
        # Disable button while processing
        self.generate_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.log(f"Generating concept map for '{concept}'...")
        
        try:
            # Get Wikipedia content
            self.log("Fetching Wikipedia content...")
            self.wiki_text = get_wikipedia_content(concept)
            
            if not self.wiki_text:
                self.log("Error: Could not retrieve Wikipedia page")
                messagebox.showerror("Error", "Could not retrieve Wikipedia page")
                self.generate_btn.config(state=tk.NORMAL)
                return
            
            # Extract concepts
            self.log("Extracting concepts...")
            self.extracted_concepts = extract_all_concepts(self.wiki_text)
            self.log(f"Found {len(self.extracted_concepts)} related concepts")
            
            # Extract relationships
            self.log("Analyzing relationships...")
            self.relationship_data = extract_rich_relationships(concept, self.wiki_text, self.extracted_concepts)
            self.log(f"Found {len(self.relationship_data)} relationships")
            
            # Create visualization
            self.log("Creating visualization...")
            self.create_visualization(concept)
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        
        # Re-enable button
        self.generate_btn.config(state=tk.NORMAL)
    
    def create_visualization(self, concept):
        # Clear previous visualization
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Get threshold
        threshold = self.threshold_var.get()
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(10, 8))
        
        # Create graph
        G = nx.Graph()
        G.add_node(concept)
        
        # Calculate relevance scores
        concept_relevance = {}
        for pair_key, data in self.relationship_data.items():
            pair_list = list(pair_key)
            if pair_list[0] == concept:
                concept_relevance[pair_list[1]] = data["count"]
            elif pair_list[1] == concept:
                concept_relevance[pair_list[0]] = data["count"]
        
        # Filter concepts by threshold
        filtered_concepts = [(c, score) for c, score in concept_relevance.items() if score >= threshold]
        top_concepts = sorted(filtered_concepts, key=lambda x: x[1], reverse=True)[:15]
        
        # Create layout
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
        
        # Draw edges
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
        canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store the graph for saving
        self.concept_graph = fig
    
    def save_results(self):
        if not self.wiki_text or not self.extracted_concepts or not self.relationship_data:
            messagebox.showerror("Error", "No data to save")
            return
        
        concept = self.concept_entry.get().strip()
        output_dir = filedialog.askdirectory(title="Select output directory")
        
        if not output_dir:
            return  # User cancelled
        
        try:
            self.log(f"Saving results to {output_dir}...")
            
            # Save data using the save_results function
            result_path = save_results(concept, self.wiki_text, self.extracted_concepts, 
                                      self.relationship_data, output_dir)
            
            # Save visualization
            if self.concept_graph:
                viz_path = os.path.join(output_dir, f"{concept.replace(' ', '_')}_concept_map.png")
                self.concept_graph.savefig(viz_path)
                self.log(f"Visualization saved to {viz_path}")
            
            self.log("All results saved successfully")
            messagebox.showinfo("Success", f"Results saved to {result_path}")
            
        except Exception as e:
            self.log(f"Error saving results: {str(e)}")
            messagebox.showerror("Error", f"Error saving: {str(e)}")

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleConceptMapApp(root)
    root.mainloop()