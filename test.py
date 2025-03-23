import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch, MagicMock
import os

# Importer les modules nécessaires de votre application
from app import app, Chatbot
from dotenv import load_dotenv


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test case."""
        self.app = app.test_client()
        self.app.testing = True

        # Mock the environment variable for OpenAI API key0000
        # 🔐 Charger la clé API OpenAI depuis les variables d’environnement
        load_dotenv()  # Charge les variables d'environnement du fichier .env

        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("Clé API OpenAI manquante ! Définissez OPENAI_API_KEY dans vos variables d’environnement.")

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
