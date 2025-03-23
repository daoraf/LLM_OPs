import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch, MagicMock
import os

# Importer les modules nécessaires de votre application
from app import app, Chatbot

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test case."""
        self.app = app.test_client()
        self.app.testing = True

        # Mock the environment variable for OpenAI API key
        os.environ["OPENAI_API_KEY"] = "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"

        # Initialize the chatbot with a mock vector database path
        self.chatbot = Chatbot("/app/vectorstore")

    def test_home_route(self):
        """Test the home route."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Historique de la conversation', response.data)

    @patch('app.Chatbot.ask')
    def test_ask_route(self, mock_ask):
        """Test the ask route with a mocked chatbot response."""
        mock_ask.return_value = "Mocked response"
        response = self.app.post('/ask', data={'question': 'Test question'})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Mocked response", response.json['response'])

    @patch('app.FAISS.load_local')
    def test_create_retriever(self, mock_load_local):
        """Test the create_retriever function with a mocked FAISS database."""
        mock_load_local.return_value = MagicMock()
        retriever = self.chatbot.create_retriever("/app/vectorstore")
        self.assertIsNotNone(retriever)

    @patch('app.ChatOpenAI')
    @patch('app.RetrievalQA.from_chain_type')
    def test_create_chatbot(self, mock_chain, mock_llm):
        """Test the create_chatbot function with mocked dependencies."""
        mock_llm.return_value = MagicMock()
        mock_chain.return_value = MagicMock()
        qa, llm = self.chatbot.create_chatbot("/app/vectorstore")
        self.assertIsNotNone(qa)
        self.assertIsNotNone(llm)

    def test_ask_method(self):
        """Test the ask method with a mocked response."""
        with patch.object(self.chatbot.qa, 'run', return_value="Mocked QA response"):
            response = self.chatbot.ask("Test question")
            self.assertEqual(response, "Mocked QA response")

if __name__ == '__main__':
    unittest.main()
