import os
import chainlit as cl
from dotenv import load_dotenv
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Charger la clé API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Charger la base FAISS existante
def load_vectorstore(path: str):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_qa_chain(vectorstore_path: str):
    db = load_vectorstore(vectorstore_path)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = create_qa_chain("vectorstore")

# Historique de conversation pour GPT
conversation_history = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant expert en naturalisation française. "
            "Tes réponses sont claires, bienveillantes et basées sur la loi. "
            "Tu réponds dans la langue de l'utilisateur (français, anglais, espagnol, italien ou allemand). "
            "Si tu ne comprends pas la langue, utilise le français par défaut."
        )
    }
]

translations = {
    # 🇫🇷 Français
    "fr": {
        "welcome": "🎉 **Bienvenue dans l'Assistant de Naturalisation Française 🇫🇷**\n\nJe suis là pour vous guider à chaque étape du processus...",
        "guide_title": "🧭 **Guide étape par étape de la naturalisation**",
        "steps": [
            "📍 **Étape 1 :** Avez-vous plus de 18 ans...",
            "📍 **Étape 2 :** Votre casier judiciaire est-il compatible...",
            "📍 **Étape 3 :** Avez-vous une preuve de niveau de langue B1...",
            "📍 **Étape 4 :** Disposez-vous des documents administratifs nécessaires...",
            "📍 **Étape 5 :** Savez-vous comment et où déposer votre demande..."
        ],
        "checklist_prompt": "✅ Si vous avez validé toutes ces étapes, tapez `checklist`...",
        "checklist": """📋 **Checklist des documents à fournir :**\n\n1. 🛂 Titre de séjour...\n2. 📄 Acte de naissance...\n...""",
        "depot": """📬 **Dépôt de votre demande :**\n\nVous pouvez déposer votre demande en ligne ou en préfecture..."""
    },

    # 🇬🇧 English
    "en": {
        "welcome": "🎉 **Welcome to the French Naturalization Assistant 🇫🇷**\n\nI'm here to guide you...",
        "guide_title": "🧭 **Step-by-step naturalization guide**",
        "steps": [
            "📍 **Step 1:** Are you over 18...",
            "📍 **Step 2:** Is your criminal record clean?",
            "📍 **Step 3:** Do you have a B1 French certificate?",
            "📍 **Step 4:** Do you have required administrative documents?",
            "📍 **Step 5:** Do you know how and where to submit?"
        ],
        "checklist_prompt": "✅ If you meet all the steps, type `checklist`...",
        "checklist": """📋 **Required documents checklist:**\n\n1. 🛂 Valid residence permit\n2. 📄 Birth certificate...\n...""",
        "depot": """📬 **Submitting your application:**\n\nYou can submit it online or at the prefecture..."""
    },

    # 🇪🇸 Español
    "es": {
        "welcome": "🎉 **Bienvenido al Asistente de Naturalización Francesa 🇫🇷**\n\nEstoy aquí para guiarte...",
        "guide_title": "🧭 **Guía paso a paso para la naturalización**",
        "steps": [
            "📍 **Paso 1:** ¿Tienes más de 18 años y vives en Francia desde hace 5 años?",
            "📍 **Paso 2:** ¿Tu historial judicial es compatible?",
            "📍 **Paso 3:** ¿Tienes un certificado de nivel B1 en francés?",
            "📍 **Paso 4:** ¿Tienes los documentos requeridos?",
            "📍 **Paso 5:** ¿Sabes dónde y cómo presentar tu solicitud?"
        ],
        "checklist_prompt": "✅ Si cumples con todos los pasos, escribe `checklist`...",
        "checklist": """📋 **Lista de documentos requeridos:**\n\n1. 🛂 Permiso de residencia válido\n2. 📄 Acta de nacimiento...\n...""",
        "depot": """📬 **Presentación de tu solicitud:**\n\nPuedes hacerlo en línea o en la prefectura..."""
    },

    # 🇮🇹 Italiano
    "it": {
        "welcome": "🎉 **Benvenuto nell'Assistente per la Naturalizzazione Francese 🇫🇷**\n\nSono qui per aiutarti...",
        "guide_title": "🧭 **Guida passo dopo passo alla naturalizzazione**",
        "steps": [
            "📍 **Passaggio 1:** Hai più di 18 anni e vivi in Francia da almeno 5 anni?",
            "📍 **Passaggio 2:** Il tuo casellario giudiziale è compatibile?",
            "📍 **Passaggio 3:** Hai una certificazione di livello B1 in francese?",
            "📍 **Passaggio 4:** Hai tutti i documenti richiesti?",
            "📍 **Passaggio 5:** Sai dove e come presentare la domanda?"
        ],
        "checklist_prompt": "✅ Se hai completato tutti i passaggi, digita `checklist`...",
        "checklist": """📋 **Elenco dei documenti richiesti:**\n\n1. 🛂 Permesso di soggiorno valido\n2. 📄 Certificato di nascita...\n...""",
        "depot": """📬 **Presentazione della domanda:**\n\nPuoi farlo online o in prefettura..."""
    },

    # 🇩🇪 Deutsch
    "de": {
        "welcome": "🎉 **Willkommen beim Assistenten für die französische Einbürgerung 🇫🇷**\n\nIch begleite Sie durch den gesamten Prozess...",
        "guide_title": "🧭 **Schritt-für-Schritt-Anleitung zur Einbürgerung**",
        "steps": [
            "📍 **Schritt 1:** Sind Sie über 18 Jahre alt und leben seit mindestens 5 Jahren in Frankreich?",
            "📍 **Schritt 2:** Ist Ihr Strafregister mit der Einbürgerung vereinbar?",
            "📍 **Schritt 3:** Haben Sie ein B1-Sprachnachweis in Französisch?",
            "📍 **Schritt 4:** Haben Sie alle erforderlichen Unterlagen?",
            "📍 **Schritt 5:** Wissen Sie, wie und wo Sie den Antrag einreichen?"
        ],
        "checklist_prompt": "✅ Wenn Sie alle Schritte erfüllt haben, geben Sie `checklist` ein...",
        "checklist": """📋 **Checkliste der erforderlichen Unterlagen:**\n\n1. 🛂 Gültiger Aufenthaltstitel\n2. 📄 Geburtsurkunde...\n...""",
        "depot": """📬 **Einreichung Ihres Antrags:**\n\nSie können den Antrag online oder in der Präfektur einreichen..."""
    }
}

def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in translations else "fr"
    except:
        return "fr"

def t(lang, key):
    return translations.get(lang, translations["fr"]).get(key, "")

@cl.on_chat_start
async def start():
    msg = translations["fr"]["welcome"] + "\n\n💡 Tapez `guide`, `checklist`, ou `dépôt` à tout moment pour être guidé."
    await cl.Message(content=msg).send()

@cl.on_message
async def handle_message(message: cl.Message):
    global conversation_history
    user_input = message.content.strip()
    lang = detect_language(user_input)

    if user_input.lower() == "guide":
        await launch_step_by_step_guide(lang)
        return
    elif user_input.lower() == "checklist":
        await send_checklist(lang)
        return
    elif user_input.lower() == "dépôt":
        await send_depot_info(lang)
        return
    elif user_input.lower() in ["/reset", "reset"]:
        conversation_history.clear()
        await cl.Message(content="♻️ Conversation réinitialisée. Posez votre question !").send()
        return

    try:
        result = qa_chain.invoke({"query": user_input})
        await cl.Message(content=result["result"]).send()

    except Exception as e:
        await cl.Message(content=f"❌ Erreur : {e}").send()

async def launch_step_by_step_guide(lang="fr"):
    await cl.Message(content=t(lang, "guide_title")).send()
    for step in t(lang, "steps"):
        await cl.Message(content=step).send()
    await cl.Message(content=t(lang, "checklist_prompt")).send()

async def send_checklist(lang="fr"):
    await cl.Message(content=t(lang, "checklist")).send()

async def send_depot_info(lang="fr"):
    await cl.Message(content=t(lang, "depot")).send()