"""
Test framework for Philosophical Concept Map Generator.
"""
import unittest
import os
import shutil
import tempfile
import json
from unittest.mock import patch, MagicMock


class BaseConceptMapTest(unittest.TestCase):
    """Base class for all concept map test cases"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, 'cache')
        self.output_dir = os.path.join(self.test_dir, 'output')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample test data
        self.sample_wiki_text = """
        Ethics is the philosophical study of moral phenomena. Also called moral philosophy, 
        it investigates normative questions about what people ought to do or which behavior 
        is morally right. Its main branches include normative ethics, applied ethics, and metaethics.
        
        Normative ethics aims to find general principles that govern how people should act. 
        Applied ethics examines concrete ethical problems in real-life situations, such as abortion, 
        treatment of animals, and business practices. Metaethics explores the underlying assumptions 
        and concepts of ethics. It asks whether there are objective moral facts, how moral knowledge 
        is possible, and how moral judgments motivate people.
        """
        
        self.sample_concepts = [
            "Ethics", "Normative Ethics", "Applied Ethics", "Metaethics",
            "Moral Philosophy", "Moral Facts", "Knowledge"
        ]
        
        self.sample_relationships = {
            frozenset({"Ethics", "Normative Ethics"}): {
                "count": 2,
                "sentences": ["Normative ethics aims to find general principles that govern how people should act."]
            },
            frozenset({"Ethics", "Applied Ethics"}): {
                "count": 1,
                "sentences": ["Applied ethics examines concrete ethical problems in real-life situations."]
            },
            frozenset({"Ethics", "Metaethics"}): {
                "count": 1,
                "sentences": ["Metaethics explores the underlying assumptions and concepts of ethics."]
            }
        }
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_mock_wikipedia_page(self, title, content):
        """Create a mock Wikipedia page file in the cache directory"""
        filename = os.path.join(self.cache_dir, f"{title.replace(' ', '_')}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return filename


class WikipediaTestCase(BaseConceptMapTest):
    """Test cases for Wikipedia functionality"""
    
    @patch('wikipedia.page')
    def test_get_wikipedia_content_api(self, mock_page):
        """Test fetching content from Wikipedia API"""
        from concept_map import get_wikipedia_content
        
        # Set up mock return value
        mock_page_instance = MagicMock()
        mock_page_instance.content = self.sample_wiki_text
        mock_page_instance.title = "Ethics"
        mock_page.return_value = mock_page_instance
        
        # Test the function
        result = get_wikipedia_content("Ethics", cache_dir=self.cache_dir)
        
        # Verify the results
        self.assertEqual(result, self.sample_wiki_text)
        mock_page.assert_called_once_with("Ethics", auto_suggest=False)
        
        # Verify that the content was cached
        cache_file = os.path.join(self.cache_dir, "Ethics.txt")
        self.assertTrue(os.path.exists(cache_file))
        
        # Test cache retrieval on second call
        mock_page.reset_mock()
        result2 = get_wikipedia_content("Ethics", cache_dir=self.cache_dir)
        self.assertEqual(result2, self.sample_wiki_text)
        mock_page.assert_not_called()  # Should use cached version


class ConceptExtractionTestCase(BaseConceptMapTest):
    """Test cases for concept extraction functionality"""
    
    def test_extract_all_concepts(self):
        """Test extracting concepts from text"""
        from concept_map import extract_all_concepts
        
        # Create a mock spaCy model and entities
        with patch('concept_map.nlp') as mock_nlp:
            # Mock document with entities
            mock_doc = MagicMock()
            mock_ents = []
            
            # Create mock entities
            mock_entity1 = MagicMock()
            mock_entity1.text = "Ethics"
            mock_entity1.label_ = "MISC"
            
            mock_entity2 = MagicMock()
            mock_entity2.text = "Normative Ethics"
            mock_entity2.label_ = "ORG"
            
            mock_ents.extend([mock_entity1, mock_entity2])
            mock_doc.ents = mock_ents
            
            mock_nlp.return_value = mock_doc
            
            # Test with philosophical_terms mocked
            with patch('concept_map.philosophical_terms', ["Ethics", "Metaethics"]):
                concepts = extract_all_concepts(self.sample_wiki_text)
                
                # Should find both NER entities and philosophical terms
                self.assertIn("Ethics", concepts)
                self.assertIn("Normative Ethics", concepts)
                self.assertIn("Metaethics", concepts)


class RelationshipExtractionTestCase(BaseConceptMapTest):
    """Test cases for relationship extraction functionality"""
    
    def test_extract_rich_relationships(self):
        """Test extracting relationships between concepts"""
        from concept_map import extract_rich_relationships
        
        # Mock spaCy sentence parsing
        with patch('concept_map.nlp') as mock_nlp:
            # Create mock sentences
            mock_sent1 = MagicMock()
            mock_sent1.text = "Ethics is the philosophical study of moral phenomena."
            
            mock_sent2 = MagicMock()
            mock_sent2.text = "Normative ethics aims to find general principles that govern how people should act."
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent1, mock_sent2]
            mock_nlp.return_value = mock_doc
            
            # Test the function
            relationships = extract_rich_relationships(
                "Ethics", 
                self.sample_wiki_text, 
                ["Ethics", "Normative Ethics"]
            )
            
            # Verify results
            self.assertIn(frozenset({"Ethics", "Normative Ethics"}), relationships)
            self.assertEqual(relationships[frozenset({"Ethics", "Normative Ethics"})]["count"], 1)


class DataSavingTestCase(BaseConceptMapTest):
    """Test cases for data saving functionality"""
    
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
        
        # Verify output directory structure
        expected_dir = os.path.join(self.output_dir, "Ethics")
        self.assertEqual(result_dir, expected_dir)
        self.assertTrue(os.path.exists(expected_dir))
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "wiki_text.txt")))
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "extracted_concepts.json")))
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "relationships.json")))
        self.assertTrue(os.path.exists(os.path.join(expected_dir, "summary.md")))
        
        # Verify content of saved files
        with open(os.path.join(expected_dir, "wiki_text.txt"), 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), self.sample_wiki_text)
            
        with open(os.path.join(expected_dir, "extracted_concepts.json"), 'r', encoding='utf-8') as f:
            saved_concepts = json.load(f)
            self.assertEqual(saved_concepts, self.sample_concepts)


def run_tests():
    """Run all tests"""
    unittest.main()


if __name__ == "__main__":
    run_tests()