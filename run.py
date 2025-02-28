"""
Simple launcher for the Philosophical Concept Map Generator.
"""
import os
import sys

# Add the project directory to the path
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "con_mapper_2")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Make sure required directories exist
os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "wiki_cache"), exist_ok=True)

# Run the application
try:
    from con_mapper_2.main import main
    sys.exit(main())
except ImportError as e:
    print(f"Error importing application: {e}")
    print("Make sure all required packages are installed.")
    sys.exit(1)