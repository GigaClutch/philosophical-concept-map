"""
Unit tests for the concept_map.py module.
"""
import unittest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import concept_map
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from test_framework import BaseConceptMapTest


class TestGetWikipediaContent(BaseConceptMapTest):
    """Test the get_wikipedia_content function"""
    
    @patch('wikipedia.page')
    def test_successful_retrieval(self, mock_page):
        """Test successfully retrieving content from Wikipedia"""
        from concept_map import get_wikipedia_content
        
        # Set up mock
        mock_page_obj = MagicMock()
        mock_page_obj.content = "Test content"
        mock_page_obj.title = "Test Title"
        mock_page.return_value = mock_page_obj
        
        # Call function
        result = get_wikipedia_content("Test", cache_dir=self.cache_dir)
        
        # Verify result
        self.assertEqual(result, "Test content")
        mock_page.assert_called_once_with("Test", auto_suggest=False)
        
        # Check that content was cached
        cache_file = os.path.join(self.cache_dir, "Test.txt")
        self.assertTrue(os.path.exists(cache_file))
        with open(cache_file, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Test content")
    
    def test_cache_retrieval(self):
        """Test retrieving content from cache"""
        from concept_map import get_wikipedia_content
        
        # Create cache file
        cache_file = os.path.join(self.cache_dir, "Test.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write("Cached content")
        
        # Call function with Wikipedia mock to ensure it's not called
        with patch('wikipedia.page') as mock_page:
            result = get_wikipedia_content("Test", cache_dir=self.cache_dir)
            
            # Verify result comes from cache
            self.assertEqual(result, "Cached content")
            mock_page.assert_not_called()
    
    @patch('wikipedia.page')
    def test_wikipedia_page_error(self, mock_page):
        """Test handling WikipediaError"""
        from concept_map import get_wikipedia_content
        import wikipedia
        
        # Set up mock to raise exception
        mock_page.side_effect = wikipedia.exceptions.PageError("Test")
        
        # Call function
        result = get_wikipedia_content("Test", cache_dir=self.cache_dir)
        
        # Verify result
        self.assertIsNone(result)
    
    @patch('wikipedia.page')
    def test_wikipedia_disambiguation_error(self, mock_page):
        """Test handling DisambiguationError"""
        from concept_map import get_wikipedia_content
        import wikipedia
        
        # Set up mock to raise exception
        mock_page.side_effect = wikipedia.exceptions.DisambiguationError("Test", ["Option1", "Option2"])
        
        # Call function
        result = get_wikipedia_content("Test", cache_dir=self.cache_dir)
        
        # Verify result
        self.assertIsNone(result)


class TestExtractAllConcepts(BaseConceptMapTest):
    """Test the extract_all_concepts function"""
    
    def test_empty_text(self):
        """Test extracting concepts from empty text"""
        from concept_map import extract_all_concepts
        
        result = extract_all_concepts("")
        self.assertEqual(result, [])
    
    @patch('concept_map.nlp')
    def test_extract_concepts(self, mock_nlp):
        """Test extracting concepts from text"""
        from concept_map import extract_all_concepts
        
        # Create mock document and entities
        mock_doc = MagicMock()
        
        mock_entity1 = MagicMock()
        mock_entity1.text = "Kant"
        mock_entity1.label_ = "PERSON"
        
        mock_entity2 = MagicMock()
        mock_entity2.text = "Utilitarianism"
        mock_entity2.label_ = "MISC"
        
        mock_doc.ents = [mock_entity1, mock_entity2]
        mock_nlp.return_value = mock_doc
        
        # Set up philosophical terms
        with patch('concept_map.philosophical_terms', ["Ethics", "Utilitarianism"]):
            result = extract_all_concepts("Test text about Ethics and Kant")
            
            # Should find both NER entities and terms
            self.assertIn("Kant", result)
            self.assertIn("Utilitarianism", result)
            self.assertIn("Ethics", result)


class TestExtractRichRelationships(BaseConceptMapTest):
    """Test the extract_rich_relationships function"""
    
    def test_empty_text(self):
        """Test with empty text"""
        from concept_map import extract_rich_relationships
        
        result = extract_rich_relationships("Ethics", "", ["Utilitarianism"])
        self.assertEqual(result, {})
    
    @patch('concept_map.nlp')
    def test_extract_relationships(self, mock_nlp):
        """Test extracting relationships between concepts"""
        from concept_map import extract_rich_relationships
        
        # Create mock sentences
        mock_sent1 = MagicMock()
        mock_sent1.text = "Ethics is related to morality."
        
        mock_sent2 = MagicMock()
        mock_sent2.text = "Utilitarianism is an ethical theory."
        
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_nlp.return_value = mock_doc
        
        # Test extraction
        result = extract_rich_relationships(
            "Ethics", 
            "Test text", 
            ["Ethics", "Utilitarianism", "Morality"]
        )
        
        # Check relationships
        self.assertIn(frozenset({"Ethics", "Morality"}), result)
        self.assertIn(frozenset({"Ethics", "Utilitarianism"}), result)
        
        # Check counts and sentences
        self.assertEqual(result[frozenset({"Ethics", "Morality"})]["count"], 1)
        self.assertEqual(
            result[frozenset({"Ethics", "Morality"})]["sentences"][0],
            "Ethics is related to morality."
        )


class TestGenerateSummary(BaseConceptMapTest):
    """Test the generate_summary function"""
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        from concept_map import generate_summary
        
        result = generate_summary("Ethics", "", [], {})
        self.assertIn("Insufficient data", result)
    
    def test_generate_summary(self):
        """Test generating a summary"""
        from concept_map import generate_summary
        
        # Test with sample data
        result = generate_summary(
            "Ethics",
            self.sample_wiki_text,
            self.sample_concepts,
            self.sample_relationships
        )
        
        # Check the summary content
        self.assertIn("Ethics", result)
        self.assertIn("Concept Map Summary", result)
        self.assertIn("Most Closely Related Concepts", result)
        
        # Should mention all related concepts
        for concept in ["Normative Ethics", "Applied Ethics", "Metaethics"]:
            self.assertIn(concept, result)


class TestSaveResults(BaseConceptMapTest):
    """Test the save_results function"""
    
    def test_save_results(self):
        """Test saving results to files"""
        from concept_map import save_results
        
        # Call the function
        result_dir = save_results(
            "Ethics",
            self.sample_wiki_text,
            self.sample_concepts,
            self.sample_relationships,
            self.output_dir
        )
        
        # Verify output directory
        expected_dir = os.path.join(self.output_dir, "Ethics")
        self.assertEqual(result_dir, expected_dir)
        self.assertTrue(os.path.exists(expected_dir))
        
        # Verify files
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "wiki_text.txt")))
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "extracted_concepts.json")))
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "relationships.json")))
        
        # Verify content
        with open(os.path.join(expected_dir, "wiki_text.txt"), 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), self.sample_wiki_text)
        
        with open(os.path.join(expected_dir, "extracted_concepts.json"), 'r', encoding='utf-8') as f:
            concepts = json.load(f)
            self.assertEqual(concepts, self.sample_concepts)


class TestCreateVisualization(BaseConceptMapTest):
    """Test the create_visualization function"""
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('networkx.spring_layout')
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_labels')
    def test_create_visualization(self, mock_labels, mock_edges, mock_nodes, 
                                mock_layout, mock_show, mock_savefig, mock_figure):
        """Test creating a visualization"""
        from concept_map import create_visualization
        
        # Mock networkx layout
        mock_layout.return_value = {
            "Ethics": (0.5, 0.5),
            "Normative Ethics": (0.2, 0.3),
            "Applied Ethics": (0.7, 0.3)
        }
        
        # Call the function
        create_visualization(
            "Ethics",
            self.sample_relationships,
            self.sample_concepts,
            threshold=1.0
        )
        
        # Verify the calls
        mock_figure.assert_called_once()
        mock_layout.assert_called_once()
        mock_nodes.assert_called_once()
        mock_savefig.assert_called_once()
        
        # Verify show was called
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()