import chromadb

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from pydantic import Field

from vector_store import embedding_model


# ============================================
# ChromaDB Retriever
# ============================================

class ChromaRetriever(BaseRetriever):

    collection: object = Field(exclude=True)
    embedding_model: object = Field(exclude=True)
    top_k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ):

        query_embedding = self.embedding_model.encode(
            [query]
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        ids = results.get(
            "ids",
            [[]]
        )[0]

        return [
            Document(
                page_content=document,
                metadata={
                    "id": doc_id
                }
            )
            for document, doc_id
            in zip(documents, ids)
        ]


# ============================================
# ChromaDB Connection
# ============================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="mental_health_knowledge"
)


retriever = ChromaRetriever(
    collection=collection,
    embedding_model=embedding_model,
    top_k=3
)


# ============================================
# Ollama / LLM
# ============================================

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0
)


# ============================================
# Prompt
# ============================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI mental health companion.

Your role is to provide supportive, empathetic
and non-judgmental conversation.

You are not a doctor, therapist, or emergency service.

Do not diagnose mental health conditions.

Do not prescribe medication.

Do not claim certainty about a user's mental health.

Encourage professional help when appropriate.

Keep responses conversational, practical and concise.

Relevant educational information from the
knowledge base is provided below.

Use it as supporting context when relevant.

Do not blindly follow it if it does not fit
the user's situation.

Do not treat it as a diagnosis or personalized
medical assessment.

--- KNOWLEDGE BASE ---

{context}

--- END KNOWLEDGE BASE ---

{risk_guidance}

--- CONVERSATION HISTORY ---

{history}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ============================================
# RAG + LLM
# ============================================

def run_rag(
    question: str,
    history: str = "",
    risk_guidance: str = ""
):

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Convert documents into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create LangChain chain
    chain = prompt | llm

    # Run LLM
    response = chain.invoke(
        {
            "context": context,
            "risk_guidance": risk_guidance,
            "history": history,
            "question": question
        }
    )

    return response.content, documents