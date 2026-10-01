from langchain_rag import run_rag

question = "I have been feeling very anxious and my thoughts keep racing."

response, documents = run_rag(question)

print("\\nRetrieved documents:")
for document in documents:
    print("-", document.metadata.get("id"))

print("\\nResponse:")
print(response)
