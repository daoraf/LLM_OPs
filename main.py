import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 🔐 Charger la clé API OpenAI depuis les variables d’environnement
openai_api_key = "clé1"
if not openai_api_key:
    st.error("🔑 Clé API OpenAI manquante ! Définissez OPENAI_API_KEY dans vos variables d’environnement.")
    st.stop()

# 📥 Charger la base de données vectorielle FAISS
def create_retriever(vector_db_path):
    embeddings = OpenAIEmbeddings()
    try:
        faiss_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        return faiss_db.as_retriever()
    except ValueError as e:
        st.error(f"❌ Erreur lors du chargement de la base FAISS : {e}")
        return None

# 🤖 Création du chatbot
def create_chatbot(vector_db_path):
    retriever = create_retriever(vector_db_path)
    if retriever is None:
        return None, None
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return retrieval_chain, llm

# 🎭 Classe chatbot
class Chatbot:
    def __init__(self, vector_db_path):
        self.qa, self.llm = create_chatbot(vector_db_path)

    def ask(self, question):
        if not self.qa:
            return "⚠️ Erreur de chargement du modèle."

        # 🔎 Recherche dans FAISS
        specific_response = self.qa.run(question)

        # 📌 Si la réponse est vide ou peu pertinente, utiliser GPT-3.5
        if not specific_response.strip():
            return self.llm.predict(question)

        return specific_response.strip()

# 🎨 Interface Streamlit
st.title("🤖 Chatbot sur la Naturalisation Française 🇫🇷")

vector_db_path = "C:\\Users\\Sysai\\PycharmProjects\\LLM_OPs\\vectorstore"  # Adapte ce chemin
chatbot = Chatbot(vector_db_path)

# 📌 Système de mémoire pour sauvegarder l'historique des échanges
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔎 Champ de saisie utilisateur
question = st.text_input("✏️ Posez votre question :")

if st.button("🗣️ Envoyer"):
    if question:
        response = chatbot.ask(question)
        st.session_state.chat_history.append((question, response))  # Ajout au chat

# 📜 Affichage de l'historique
st.subheader("💬 Historique de la conversation")
for q, r in st.session_state.chat_history:
    st.write(f"**🧑‍💼 Vous :** {q}")
    st.write(f"**🤖 Chatbot :** {r}")
    st.write("---")
