from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supportive mental health companion. Do not diagnose or prescribe medication."),
    ("human", "{message}")
])

chain = prompt | llm

response = chain.invoke({
    "message": "I have been feeling stressed about work lately."
})

print(response.content)
