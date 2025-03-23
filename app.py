from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Charger la clé API OpenAI depuis les variables d’environnement
load_dotenv()  # Charge les variables d'environnement du fichier .env

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "Clé API OpenAI manquante ! Définissez OPENAI_API_KEY dans vos variables d’environnement."
    )

# Charger la base de données vectorielle FAISS
def create_retriever(vector_db_path):
    embeddings = OpenAIEmbeddings()
    try:
        faiss_db = FAISS.load_local(
            vector_db_path, embeddings, allow_dangerous_deserialization=True
        )
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

    system_prompt = (
        "Utilisez le contexte fourni pour répondre à la question. "
        "Si vous ne connaissez pas la réponse, dites que vous ne savez pas. "
        "Utilisez un maximum de trois phrases et soyez concis. "
        "Contexte : {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    return retrieval_chain, llm

# Classe chatbot
class Chatbot:
    def __init__(self, vector_db_path):
        self.qa, self.llm = create_chatbot(vector_db_path)
        self.chat_history = []

    def ask(self, question):
        if not self.qa:
            return "Erreur de chargement du modèle."
        response = self.qa.invoke({"input": question})
        answer = response.get("answer", "Je ne sais pas.").strip()
        self.chat_history.append((question, answer))
        return answer

    def get_chat_history(self):
        return self.chat_history

# Interface Flask
vector_db_path = "/app/vectorstore"  # Adaptez ce chemin si nécessaire
chatbot = Chatbot(vector_db_path)

@app.route("/")
def home():
    return render_template("index.html", chat_history=chatbot.get_chat_history())

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    response = chatbot.ask(question)
    return jsonify({"response": response, "chat_history": chatbot.get_chat_history()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085, debug=True)
