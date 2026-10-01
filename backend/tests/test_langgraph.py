from langgraph_workflow import graph

from database import SessionLocal
from models import Conversation, Message


# ============================================
# Create test conversation
# ============================================

db = SessionLocal()

conversation = Conversation()

db.add(conversation)
db.commit()
db.refresh(conversation)

conversation_id = conversation.id

db.close()

print(
    f"\nCreated test conversation: {conversation_id}"
)


# ============================================
# First message
# ============================================

print("\n===== FIRST MESSAGE =====")

result = graph.invoke({
    "conversation_id": conversation_id,
    "user_message": "I've been feeling stressed about work lately.",
    "history": ""
})

print("\nRisk level:")
print(result["risk_level"])

print("\nResponse:")
print(result["response"])


# ============================================
# Save first USER message
# ============================================

db = SessionLocal()

user_message = Message(
    conversation_id=conversation_id,
    role="user",
    content="I've been feeling stressed about work lately.",
    risk_level=result["risk_level"]
)

db.add(user_message)
db.commit()

db.close()


# ============================================
# Second message
# ============================================

print("\n===== SECOND MESSAGE =====")

result = graph.invoke({
    "conversation_id": conversation_id,
    "user_message": "What was I feeling earlier?",
    "history": ""
})

print("\nRisk level:")
print(result["risk_level"])

print("\nResponse:")
print(result["response"])


# ============================================
# High-risk test
# ============================================

print("\n===== HIGH-RISK MESSAGE =====")

result = graph.invoke({
    "conversation_id": conversation_id,
    "user_message": "I want to kill myself.",
    "history": ""
})

print("\nRisk level:")
print(result["risk_level"])

print("\nResponse:")
print(result["response"])