"""
Generate sample data for concept mapping.
"""
from simple_visualization import create_sample_data, create_visualization
import matplotlib.pyplot as plt

# Generate data for common philosophical concepts
concepts = ["Justice", "Ethics", "Metaphysics", "Epistemology", "Aesthetics"]

for concept in concepts:
    print(f"Generating sample data for {concept}...")
    create_sample_data(concept)

print("All sample data generated successfully!")

# Test visualization
fig = create_visualization("Justice", {}, [])
plt.savefig("test_visualization.png")
print("Test visualization saved to test_visualization.png")