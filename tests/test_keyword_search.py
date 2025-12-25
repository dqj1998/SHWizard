import unittest
from unittest.mock import Mock, patch, MagicMock
from shwizard.storage.database import Database
from shwizard.storage.history import HistoryManager
from shwizard.core.ai_service import AIService


class TestKeywordSearch(unittest.TestCase):
    """Test keyword-based history search functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db = Database(db_path=":memory:")
        self.history_manager = HistoryManager(database=self.db)
        
        # Add some test data
        context = {"os": "linux", "cwd": "/home/test", "shell": "bash"}
        
        # Add test commands
        cmd1_id = self.db.add_command_history(
            user_query="find all python files",
            generated_command="find . -name '*.py'",
            executed=True,
            platform="linux",
            working_directory="/home/test",
            context_data=context
        )
        self.db.update_command_execution(cmd1_id, True, "success", 1)
        
        cmd2_id = self.db.add_command_history(
            user_query="list python files in directory",
            generated_command="ls -la *.py",
            executed=True,
            platform="linux",
            working_directory="/home/test",
            context_data=context
        )
        self.db.update_command_execution(cmd2_id, True, "success", 1)
        
        cmd3_id = self.db.add_command_history(
            user_query="compress all images",
            generated_command="tar -czf images.tar.gz *.jpg *.png",
            executed=True,
            platform="linux",
            working_directory="/home/test",
            context_data=context
        )
        self.db.update_command_execution(cmd3_id, True, "success", 1)
    
    def test_search_by_keywords_basic(self):
        """Test basic keyword search functionality"""
        keywords = ["python", "files"]
        results = self.db.search_by_keywords(keywords, limit=10, executed_only=True)
        
        # Should find 2 commands matching "python" and "files"
        self.assertGreater(len(results), 0)
        
        # Check that results are sorted by keyword match count
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i]["keyword_match_count"],
                    results[i + 1]["keyword_match_count"]
                )
    
    def test_search_by_keywords_ranking(self):
        """Test that results are properly ranked by keyword match count"""
        keywords = ["python", "files", "find"]
        results = self.db.search_by_keywords(keywords, limit=10, executed_only=True)
        
        # The first result should have the highest keyword match count
        if len(results) > 0:
            first_result = results[0]
            self.assertIn("keyword_match_count", first_result)
            self.assertGreater(first_result["keyword_match_count"], 0)
    
    def test_search_by_keywords_no_match(self):
        """Test search with keywords that don't match anything"""
        keywords = ["nonexistent", "keyword"]
        results = self.db.search_by_keywords(keywords, limit=10, executed_only=True)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
    
    def test_search_by_keywords_single_keyword(self):
        """Test search with single keyword"""
        keywords = ["python"]
        results = self.db.search_by_keywords(keywords, limit=10, executed_only=True)
        
        # Should find commands containing "python"
        self.assertGreater(len(results), 0)
        for result in results:
            query_lower = result["user_query"].lower()
            cmd_lower = result["generated_command"].lower()
            self.assertTrue("python" in query_lower or "python" in cmd_lower)
    
    def test_history_manager_search_by_keywords(self):
        """Test HistoryManager wrapper for keyword search"""
        keywords = ["python", "files"]
        results = self.history_manager.search_by_keywords(
            keywords=keywords,
            limit=10
        )
        
        # Should return results
        self.assertGreater(len(results), 0)
        
        # Each result should have keyword_match_count
        for result in results:
            self.assertIn("keyword_match_count", result)
    
    @patch('shwizard.core.ai_service.requests.post')
    def test_extract_keywords(self, mock_post):
        """Test keyword extraction from AI service"""
        # Mock the Ollama API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "find, python, files, directory, search, list"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create AI service with mocked Ollama manager
        mock_ollama_manager = MagicMock()
        mock_ollama_manager.is_server_running.return_value = True
        mock_ollama_manager.ensure_model_available.return_value = True
        
        ai_service = AIService(
            ollama_manager=mock_ollama_manager,
            model="test-model",
            base_url="http://localhost:11435"
        )
        
        # Initialize
        self.assertTrue(ai_service.initialize())
        
        # Extract keywords
        keywords = ai_service.extract_keywords("find all python files in the current directory")
        
        # Should return list of keywords
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        self.assertLessEqual(len(keywords), 6)
    
    def test_keyword_match_count_calculation(self):
        """Test that keyword match count is correctly calculated"""
        keywords = ["python", "files"]
        results = self.db.search_by_keywords(keywords, limit=10, executed_only=True)
        
        for result in results:
            # Manually count matches
            user_query_lower = result["user_query"].lower()
            cmd_lower = result["generated_command"].lower()
            expected_count = sum(
                1 for kw in keywords
                if kw.lower() in user_query_lower or kw.lower() in cmd_lower
            )
            
            self.assertEqual(result["keyword_match_count"], expected_count)


if __name__ == '__main__':
    unittest.main()
