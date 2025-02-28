"""
Configuration settings for Philosophical Concept Map Generator.
"""
import os
import json
import logging
from pathlib import Path

# Application version
VERSION = "0.2.0"

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "wiki_cache"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOG_DIR / "concept_mapper.log"

# Text processing
MAX_TEXT_LENGTH = 20000  # Limit for text processing
MIN_SENTENCE_LENGTH = 10  # Minimum sentence length to consider
MAX_SENTENCES_PER_CONCEPT = 5  # Maximum number of example sentences to save

# Visualization
MAX_CONCEPTS_DISPLAY = 15  # Maximum number of concepts to display
DEFAULT_THRESHOLD = 1.0  # Default relevance threshold
FIGURE_SIZE = (12, 10)  # Default figure size for visualizations
NODE_SIZE = 3000  # Size of nodes in graph
NODE_COLOR = "skyblue"  # Color of nodes in graph
EDGE_COLOR = "gray"  # Color of edges in graph

# NLP configuration
RELEVANT_ENTITY_TYPES = ["ORG", "PERSON", "NORP", "MISC", "GPE"]

# Core philosophical terms
PHILOSOPHICAL_TERMS = [
    "Ethics", "Metaphysics", "Epistemology", "Logic", "Aesthetics", 
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology", 
    "Determinism", "Free Will", "Consciousness", "Virtue Ethics", 
    "Deontology", "Utilitarianism", "Moral Realism", "Relativism",
    "Ontology", "Dualism", "Materialism", "Idealism", "Pragmatism",
    "Positivism", "Skepticism", "Nihilism", "Subjectivism", "Objectivism"
]

# User configuration
USER_CONFIG_FILE = BASE_DIR / "user_config.json"

# Default user configuration
DEFAULT_USER_CONFIG = {
    "theme": "light",
    "cache_enabled": True,
    "auto_save": True,
    "default_threshold": 1.0,
    "max_concepts": 15,
    "show_help_on_startup": True,
    "recent_concepts": []
}


class Config:
    """Configuration manager for application settings"""
    
    def __init__(self):
        """Initialize configuration with default and user settings"""
        self.settings = {}
        
        # Load default settings from this module
        for key, value in globals().items():
            if key.isupper() and not key.startswith('_'):
                self.settings[key.lower()] = value
        
        # Load user config if exists
        self.user_config = self._load_user_config()
        
        # Merge user config into settings
        for key, value in self.user_config.items():
            if key.lower() in self.settings:
                self.settings[key.lower()] = value
    
    def _load_user_config(self):
        """Load user configuration from file"""
        if USER_CONFIG_FILE.exists():
            try:
                with open(USER_CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return DEFAULT_USER_CONFIG.copy()
        return DEFAULT_USER_CONFIG.copy()
    
    def save_user_config(self):
        """Save user configuration to file"""
        with open(USER_CONFIG_FILE, 'w') as f:
            json.dump(self.user_config, f, indent=2)
    
    def get(self, key, default=None):
        """Get a configuration value"""
        return self.settings.get(key.lower(), default)
    
    def set(self, key, value):
        """Set a configuration value"""
        key = key.lower()
        self.settings[key] = value
        
        # If this is a user configurable setting, also update user_config
        if key in self.user_config:
            self.user_config[key] = value
            self.save_user_config()
    
    def update_user_setting(self, key, value):
        """Update a user-specific setting"""
        self.user_config[key] = value
        self.settings[key.lower()] = value
        self.save_user_config()
    
    def add_recent_concept(self, concept):
        """Add a concept to the recent concepts list"""
        recent = self.user_config.get("recent_concepts", [])
        
        # Remove if already exists to avoid duplicates
        if concept in recent:
            recent.remove(concept)
        
        # Add to beginning of list
        recent.insert(0, concept)
        
        # Limit list size
        self.user_config["recent_concepts"] = recent[:10]
        self.save_user_config()
    
    def get_recent_concepts(self):
        """Get the list of recent concepts"""
        return self.user_config.get("recent_concepts", [])
    
    def clear_recent_concepts(self):
        """Clear the recent concepts list"""
        self.user_config["recent_concepts"] = []
        self.save_user_config()
    
    def reset_to_defaults(self):
        """Reset all user settings to defaults"""
        self.user_config = DEFAULT_USER_CONFIG.copy()
        self.save_user_config()
        
        # Reload settings
        for key, value in self.user_config.items():
            self.settings[key.lower()] = value


# Create global configuration instance
config = Config()