import chainlit as cl
import openai

openai.api_key = "VOTRE_CLE_API_OPENAI"

conversation_history = [
    {
        "role": "system",
        "content": (
            "Tu es un expert en naturalisation française. "
            "Tes réponses sont pédagogiques, fiables, à jour et reposent sur les textes officiels du gouvernement."
        )
    }
]

quick_questions = [
    "Quelles sont les conditions pour être naturalisé ?",
    "Quels documents dois-je fournir ?",
    "Combien de temps prend la procédure ?",
    "Puis-je faire une demande si je suis étudiant ?",
    "Comment se passe l'entretien de naturalisation ?"
]

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Bonjour et bienvenue !\n\nJe suis un assistant intelligent spécialisé dans la naturalisation française. Posez-moi une question, ou choisissez un sujet ci-dessous 👇",
        actions=[
            cl.Action(name=f"q{i}", value=q, label=q, payload={"question": q})
            for i, q in enumerate(quick_questions)
        ]
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    global conversation_history

    user_input = message.content
    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=conversation_history,
            temperature=0.4,
            max_tokens=500
        )

        answer = response["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": answer})

        await cl.Message(content=answer).send()

        await cl.Message(
            content="Souhaitez-vous explorer un autre sujet ?",
            actions=[
                cl.Action(name=f"q{i}", value=q, label=q, payload={"question": q})
                for i, q in enumerate(quick_questions)
            ]
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ Une erreur est survenue : {e}").send()
