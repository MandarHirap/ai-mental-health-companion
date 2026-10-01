from pathlib import Path

from vector_store import add_documents, collection


KNOWLEDGE_DIR = Path("knowledge")


def main():
    documents = []
    ids = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(text)
        ids.append(file_path.stem)

    if not documents:
        print("No knowledge documents found.")
        return

    # Remove existing versions of these documents.
    existing = collection.get(ids=ids)

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print("Updated existing documents:", existing["ids"])

    # Add the current versions.
    add_documents(
        documents=documents,
        ids=ids
    )

    print("Knowledge ingestion complete.")
    print("Documents added/updated:", len(documents))
    print("Total documents:", collection.count())


if __name__ == "__main__":
    main()