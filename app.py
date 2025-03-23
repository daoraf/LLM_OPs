from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Charger la clé API OpenAI depuis les variables d’environnement
load_dotenv()  # Charge les variables d'environnement du fichier .env

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Clé API OpenAI manquante ! Définissez OPENAI_API_KEY dans vos variables d’environnement.")

# Charger la base de données vectorielle FAISS
def create_retriever(vector_db_path):
    embeddings = OpenAIEmbeddings()
    try:
        faiss_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        return faiss_db.as_retriever()
    except ValueError as e:
        print(f"Erreur lors du chargement de la base FAISS : {e}")
        return None

# Création du chatbot
def create_chatbot(vector_db_path):
    retriever = create_retriever(vector_db_path)
    if retriever is None:
        return None, None
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return retrieval_chain, llm

# Classe chatbot
class Chatbot:
    def __init__(self, vector_db_path):
        self.qa, self.llm = create_chatbot(vector_db_path)

    def ask(self, question):
        if not self.qa:
            return "Erreur de chargement du modèle."
        specific_response = self.qa.run(question)
        if not specific_response.strip():
            return self.llm.predict(question)
        return specific_response.strip()

# Interface Flask
vector_db_path = "/app/vectorstore"  # Adaptez ce chemin si nécessaire
chatbot = Chatbot(vector_db_path)

chat_history = []

@app.route('/')
def home():
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    question = request.form['question']
    response = chatbot.ask(question)
    chat_history.append((question, response))
    return jsonify({"response": response, "chat_history": chat_history})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8085, debug=True)
