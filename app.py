import os
import chainlit as cl
from dotenv import load_dotenv
import openai  # ✅ nouvelle API

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ instanciation correcte

conversation_history = [
    {
        "role": "system",
        "content": (
            "Tu es un expert en naturalisation française. "
            "Tes réponses sont claires, bienveillantes et basées sur la loi."
        )
    }
]

@cl.on_chat_start
async def start():
    await cl.Message(content="""
🎉 **Bienvenue dans l'Assistant de Naturalisation Française 🇫🇷**

Je suis là pour vous guider à chaque étape du processus :
- Vérification des conditions
- Constitution du dossier
- Dépôt de la demande

Tapez `guide` pour démarrer un parcours étape par étape, ou posez une question librement !
""").send()

@cl.on_message
async def handle_message(message: cl.Message):
    global conversation_history

    user_input = message.content.strip().lower()

    if user_input == "guide":
        await launch_step_by_step_guide()
        return

    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation_history,
            temperature=0.4,
            max_tokens=500
        )
        answer = response.choices[0].message["content"]  # ✅ bonne syntaxe
        conversation_history.append({"role": "assistant", "content": answer})
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"❌ Erreur : {e}").send()

async def launch_step_by_step_guide():
    steps = [
        "📍 **Étape 1 :** Avez-vous plus de 18 ans et résidez-vous en France depuis au moins 5 ans ?",
        "📍 **Étape 2 :** Votre casier judiciaire est-il compatible avec la naturalisation ?",
        "📍 **Étape 3 :** Avez-vous une preuve de niveau de langue B1 (ex. TCF, DELF) ?",
        "📍 **Étape 4 :** Disposez-vous des documents administratifs nécessaires (titre de séjour, acte de naissance, justificatif de domicile...) ?",
        "📍 **Étape 5 :** Savez-vous comment et où déposer votre demande (en ligne ou en préfecture) ?"
    ]

    await cl.Message(content="🧭 **Guide étape par étape de la naturalisation**").send()

    for step in steps:
        await cl.Message(content=step).send()

    await cl.Message(content="✅ Si vous avez validé toutes ces étapes, tapez `checklist` pour voir les documents à préparer.").send()
