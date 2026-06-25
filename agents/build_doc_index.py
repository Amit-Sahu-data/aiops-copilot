import os
import chromadb
from chromadb.utils import embedding_functions

RUNBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "runbooks")
DB_DIR = os.path.join(os.path.dirname(__file__), "doc_index")

def build_index():
    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recreate the collection fresh each time this script runs
    try:
        client.delete_collection("runbooks")
    except Exception:
        pass
    collection = client.create_collection("runbooks", embedding_function=embedding_fn)

    docs, ids, metadatas = [], [], []
    for filename in os.listdir(RUNBOOKS_DIR):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(RUNBOOKS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append(content)
        ids.append(filename)
        metadatas.append({"source": filename})

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"Indexed {len(docs)} documents: {ids}")

if __name__ == "__main__":
    build_index()