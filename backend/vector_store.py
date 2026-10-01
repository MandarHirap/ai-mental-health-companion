import chromadb
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = chroma_client.get_or_create_collection(
    name="mental_health_knowledge"
)


def add_documents(
    documents: list[str],
    ids: list[str]
):
    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )


def search_documents(
    query: str,
    top_k: int = 3
):
    query_embedding = embedding_model.encode(
        [query]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results