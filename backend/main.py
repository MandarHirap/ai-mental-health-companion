from fastapi import FastAPI
from pydantic import BaseModel

from database import engine, SessionLocal, Base
from models import Conversation, Message

from langgraph_workflow import graph


# ============================================
# Create database tables
# ============================================

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI Mental Health Companion",
    version="1.0.0"
)


# ============================================
# Request / Response Models
# ============================================

class ChatRequest(BaseModel):

    conversation_id: int | None = None

    message: str


class ChatResponse(BaseModel):

    conversation_id: int

    response: str


# ============================================
# Root Endpoint
# ============================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "AI Mental Health Companion"
    }


# ============================================
# Chat Endpoint
# ============================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    db = SessionLocal()

    conversation = None

    try:

        # ========================================
        # 1. Create or Load Conversation
        # ========================================

        if request.conversation_id is None:

            conversation = Conversation()

            db.add(conversation)

            db.commit()

            db.refresh(conversation)

        else:

            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id
                    == request.conversation_id
                )
                .first()
            )

            if conversation is None:

                return ChatResponse(
                    conversation_id=0,
                    response="Conversation not found."
                )


        # ========================================
        # 2. Save User Message
        # ========================================

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )

        db.add(user_message)

        db.commit()


        # ========================================
        # 3. Run LangGraph
        # ========================================

        result = graph.invoke({

            "conversation_id": conversation.id,

            "user_message": request.message,

            "history": ""

        })


        # ========================================
        # 4. Get LangGraph Response
        # ========================================

        assistant_response = result.get(
            "response",
            ""
        )


        # ========================================
        # 5. Return Response
        # ========================================

        return ChatResponse(

            conversation_id=conversation.id,

            response=assistant_response
        )


    except Exception as e:

        print(
            f"AI processing error: {str(e)}"
        )

        return ChatResponse(

            conversation_id=(
                conversation.id
                if conversation
                else 0
            ),
            response=(
                f"Unable to process your message: {str(e)}"
            )
        )


    finally:

        db.close()