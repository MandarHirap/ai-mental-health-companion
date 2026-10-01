from typing import TypedDict

from langgraph.graph import StateGraph, END

from database import SessionLocal
from models import Message

from safety import (
    classify_message,
    get_safety_response,
    HIGH_RISK,
    DISTRESS
)

from langchain_rag import run_rag


# ============================================
# LangGraph State
# ============================================

class MentalHealthState(TypedDict):

    conversation_id: int

    user_message: str

    risk_level: str

    risk_guidance: str

    history: str

    response: str

    retrieved_documents: list


# ============================================
# 1. Load Conversation Memory
# ============================================

def memory_node(state: MentalHealthState):

    db = SessionLocal()

    try:

        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id
                == state["conversation_id"]
            )
            .order_by(Message.id)
            .all()
        )

        history = ""

        for message in messages:

            if message.role == "user":

                history += (
                    f"User: {message.content}\n"
                )

            else:

                history += (
                    f"Assistant: {message.content}\n"
                )

        print(
            f"LangGraph memory: {len(messages)} messages loaded"
        )

        return {
            "history": history
        }

    finally:

        db.close()


# ============================================
# 2. Safety Node
# ============================================

def safety_node(state: MentalHealthState):

    user_message = state["user_message"]

    risk_level = classify_message(
        user_message
    )

    print(
        f"LangGraph safety classification: {risk_level}"
    )

    return {
        "risk_level": risk_level
    }


# ============================================
# 3. High-Risk Response Node
# ============================================

def high_risk_node(state: MentalHealthState):

    response = get_safety_response()

    return {
        "response": response
    }


# ============================================
# 4. Prepare Guidance Node
# ============================================

def guidance_node(state: MentalHealthState):

    risk_guidance = ""

    if state["risk_level"] == DISTRESS:

        risk_guidance = """
The user appears to be experiencing emotional distress.

Respond with extra empathy and patience.

Do not diagnose them.

Acknowledge their feelings without assuming their cause.

Encourage practical, low-risk coping steps when appropriate.

If the conversation suggests increasing safety concerns,
encourage the user to seek support from a trusted person
or qualified professional.
"""

    return {
        "risk_guidance": risk_guidance
    }


# ============================================
# 5. RAG + LLM Node
# ============================================

def response_node(state: MentalHealthState):

    response, documents = run_rag(
        question=state["user_message"],
        history=state.get("history", ""),
        risk_guidance=state.get(
            "risk_guidance",
            ""
        )
    )

    print(
        "LangGraph RAG documents:",
        [
            document.metadata.get("id")
            for document in documents
        ]
    )

    return {
        "response": response,
        "retrieved_documents": documents
    }


# ============================================
# 6. Save Assistant Response
# ============================================

def save_response_node(
    state: MentalHealthState
):

    db = SessionLocal()

    try:

        assistant_message = Message(
            conversation_id=state[
                "conversation_id"
            ],
            role="assistant",
            content=state["response"],
            risk_level=state["risk_level"]
        )

        db.add(assistant_message)

        db.commit()

        print(
            "LangGraph: assistant response saved"
        )

        return {}

    finally:

        db.close()


# ============================================
# 7. Routing Function
# ============================================

def route_after_safety(
    state: MentalHealthState
):

    if state["risk_level"] == HIGH_RISK:

        return "high_risk"

    return "normal"


# ============================================
# Build LangGraph
# ============================================

graph_builder = StateGraph(
    MentalHealthState
)


graph_builder.add_node(
    "memory",
    memory_node
)

graph_builder.add_node(
    "safety",
    safety_node
)

graph_builder.add_node(
    "high_risk_response",
    high_risk_node
)

graph_builder.add_node(
    "guidance",
    guidance_node
)

graph_builder.add_node(
    "response",
    response_node
)

graph_builder.add_node(
    "save_response",
    save_response_node
)


# ============================================
# Graph Edges
# ============================================

graph_builder.set_entry_point(
    "memory"
)


graph_builder.add_edge(
    "memory",
    "safety"
)


graph_builder.add_conditional_edges(
    "safety",
    route_after_safety,
    {
        "high_risk": "high_risk_response",
        "normal": "guidance"
    }
)


graph_builder.add_edge(
    "high_risk_response",
    "save_response"
)


graph_builder.add_edge(
    "guidance",
    "response"
)


graph_builder.add_edge(
    "response",
    "save_response"
)


graph_builder.add_edge(
    "save_response",
    END
)


# ============================================
# Compile Graph
# ============================================

graph = graph_builder.compile()