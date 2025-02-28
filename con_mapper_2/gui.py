import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import os
from PIL import Image, ImageTk

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
    
    def setup_input_section(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        # Concept input
        ttk.Label(input_frame, text="Enter Philosophical Concept:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.concept_entry = ttk.Entry(input_frame, width=30)
        self.concept_entry.grid(row=0, column=1, pady=5, padx=5)
        self.concept_entry.insert(0, "Ethics")
        
        # Advanced options
        options_frame = ttk.LabelFrame(input_frame, text="Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(options_frame, text="Relevance Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=1.0)
        self.threshold_slider = ttk.Scale(options_frame, from_=0.0, to=10.0, 
                                        variable=self.threshold_var, orient=tk.HORIZONTAL)
        self.threshold_slider.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.threshold_label = ttk.Label(options_frame, text="1.0")
        self.threshold_label.grid(row=0, column=2)
        
        self.threshold_var.trace_add("write", self.update_threshold_label)
        
        # Concept categories frame
        categories_frame = ttk.LabelFrame(input_frame, text="Concept Categories", padding="10")
        categories_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        self.categories_text = scrolledtext.ScrolledText(categories_frame, width=30, height=10)
        self.categories_text.pack(fill=tk.BOTH, expand=True)
        self.categories_text.insert(tk.END, "Justice: Concept\nEthics: Concept\nPlato: Philosopher\n" +
                                   "Kant: Philosopher\nUtilitarianism: Ethical Theory")
        
        # Action buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Concept Map", 
                                     command=self.generate_concept_map)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Concept Map", 
                                  command=self.save_concept_map)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.config(state=tk.DISABLED)
        
        # Log frame
        log_frame = ttk.LabelFrame(input_frame, text="Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=30, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
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
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def generate_concept_map(self):
        concept = self.concept_entry.get().strip()
        if not concept:
            self.log("Please enter a philosophical concept")
            return
        
        self.log(f"Generating concept map for: '{concept}'...")
        self.generate_btn.config(state=tk.DISABLED)
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=self.process_concept_map, args=(concept,), daemon=True).start()
    
    def process_concept_map(self, concept):
        try:
            # Clear previous visualization
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Here you would call your existing functions
            # For example:
            # wiki_text = get_wikipedia_content(concept)
            # self.extracted_concepts = extract_concepts_ner(wiki_text)
            # self.relationship_data = extract_relationships_cooccurrence(concept, wiki_text, self.extracted_concepts)
            
            # Placeholder for demonstration
            self.log("Getting Wikipedia content...")
            self.log("Extracting concepts...")
            self.log("Analyzing relationships...")
            
            # Create a sample visualization (replace with your real visualization code)
            self.create_sample_visualization(concept)
            
            self.log(f"Concept map for '{concept}' generated successfully")
            self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state=tk.NORMAL)
    
    def create_sample_visualization(self, concept):
        # Create a figure for the concept map (replace with your visualization)
        fig = plt.figure(figsize=(8, 6))
        
        # Create sample graph
        G = nx.DiGraph()
        G.add_node(concept, pos=(0.5, 0.5))
        sample_concepts = ["Morality", "Justice", "Virtue", "Consequentialism", "Rights"]
        positions = [(0.2, 0.8), (0.8, 0.8), (0.8, 0.2), (0.2, 0.2), (0.5, 0.1)]
        
        for i, sample_concept in enumerate(sample_concepts):
            G.add_node(sample_concept, pos=positions[i])
            G.add_edge(concept, sample_concept, weight=i+1)
        
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", 
                font_weight='bold', arrows=True)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store the graph for saving
        self.concept_graph = fig
    
    def save_concept_map(self):
        if self.concept_graph:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"concept_map_{self.concept_entry.get()}.png"
            )
            if file_path:
                self.concept_graph.savefig(file_path)
                self.log(f"Concept map saved to {file_path}")
        else:
            self.log("No concept map to save")

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ConceptMapApp(root)
    root.mainloop()