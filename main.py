from fastapi import FastAPI
from pydantic import BaseModel
import requests

from database import engine, SessionLocal, Base
from models import Conversation, Message

from safety import (
    classify_message,
    get_safety_response,
    HIGH_RISK,
    DISTRESS
)


# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI Mental Health Companion",
    version="1.0.0"
)


# --------------------------------
# Ollama Configuration
# --------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"


# --------------------------------
# Request / Response Models
# --------------------------------

class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str


class ChatResponse(BaseModel):
    conversation_id: int
    response: str


# --------------------------------
# System Prompt
# --------------------------------

SYSTEM_PROMPT = """
You are an AI mental health companion.

Your role is to provide supportive, empathetic and non-judgmental
conversation.

You are not a doctor, therapist, or emergency service.

Do not diagnose mental health conditions.
Do not prescribe medication.
Do not claim certainty about a user's mental health.

Encourage professional help when appropriate.

Keep responses conversational, practical and concise.
"""


# --------------------------------
# Root Endpoint
# --------------------------------

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "AI Mental Health Companion"
    }


# --------------------------------
# Chat Endpoint
# --------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    db = SessionLocal()
    conversation = None

    try:

        # ============================================
        # 1. Create or Load Conversation
        # ============================================

        if request.conversation_id is None:

            conversation = Conversation()

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

        else:

            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id == request.conversation_id
                )
                .first()
            )

            if conversation is None:

                return ChatResponse(
                    conversation_id=0,
                    response="Conversation not found."
                )


        # ============================================
        # 2. Classify User Message
        # ============================================

        risk_level = classify_message(request.message)

        print(
            f"Safety classification: {risk_level}"
        )


        # ============================================
        # 3. Save User Message + Risk Level
        # ============================================

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            risk_level=risk_level
        )

        db.add(user_message)
        db.commit()


        # ============================================
        # 4. High-Risk Safety Flow
        # ============================================

        if risk_level == HIGH_RISK:

            safety_response = get_safety_response()

            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=safety_response,
                risk_level=HIGH_RISK
            )

            db.add(assistant_message)
            db.commit()

            return ChatResponse(
                conversation_id=conversation.id,
                response=safety_response
            )


        # ============================================
        # 5. Load Conversation History
        # ============================================

        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation.id
            )
            .order_by(Message.id)
            .all()
        )


        # ============================================
        # 6. Build Prompt
        # ============================================

        prompt = SYSTEM_PROMPT + "\n\n"


        # Additional guidance for distress
        if risk_level == DISTRESS:

            prompt += """
The user appears to be experiencing emotional distress.

Respond with extra empathy and patience.

Do not diagnose them.

Acknowledge their feelings without assuming their cause.

Encourage practical, low-risk coping steps when appropriate.

If the conversation suggests increasing safety concerns,
encourage the user to seek support from a trusted person
or qualified professional.

"""


        # Add conversation history

        for message in messages:

            if message.role == "user":

                prompt += (
                    f"User: {message.content}\n"
                )

            else:

                prompt += (
                    f"Assistant: {message.content}\n"
                )


        prompt += "\nAssistant:"


        # ============================================
        # 7. Send Request to Local Ollama
        # ============================================

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }


        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()


        data = response.json()


        assistant_response = data.get(
            "response",
            ""
        ).strip()


        # ============================================
        # 8. Save Assistant Response
        # ============================================

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_response
        )

        db.add(assistant_message)
        db.commit()


        # ============================================
        # 9. Return Response
        # ============================================

        return ChatResponse(
            conversation_id=conversation.id,
            response=assistant_response
        )


    # ================================================
    # Ollama Connection Error
    # ================================================

    except requests.exceptions.RequestException as e:

        return ChatResponse(
            conversation_id=conversation.id if conversation else 0,
            response=(
                f"Unable to connect to local AI model: {str(e)}"
            )
        )


    # ================================================
    # Close Database Connection
    # ================================================

    finally:

        db.close()