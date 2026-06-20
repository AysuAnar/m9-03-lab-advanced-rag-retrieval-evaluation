import os
import json
import chromadb
from dotenv import load_dotenv
from google import genai

def load_knowledge_base(filepath="knowledge_base.json"):
    """Loads knowledge base documents from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("Loading environment variables and initializing Gemini client...")
    load_dotenv()
    
    # Initialize the Google GenAI SDK client.
    # It automatically reads GEMINI_API_KEY from environment variables.
    client = genai.Client()

    print("Loading knowledge base...")
    kb = load_knowledge_base()
    print(f"Loaded {len(kb)} passages from knowledge base.")

    # Initialize Chroma persistent client
    print("Connecting to persistent Chroma database...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Recreate the collection to ensure clean state
    collection_name = "rag_knowledge_base"
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}' for re-indexing.")
    except Exception:
        # Collection didn't exist, ignore
        pass

    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created fresh collection '{collection_name}'.")

    # Generate embeddings and add to collection
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    print("Generating embeddings for each passage using 'gemini-embedding-2'...")
    for idx, item in enumerate(kb):
        doc_id = item["id"]
        source = item["source"]
        text = item["text"]
        
        print(f"  [{idx+1}/{len(kb)}] Embedding passage {doc_id} (Source: {source})...")
        
        # Embed the passage text
        res = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        embedding = res.embeddings[0].values
        
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({"source": source})
        embeddings.append(embedding)

    print("Adding passages and embeddings to Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    print(f"Indexing complete! Added {collection.count()} passages to the database.")

if __name__ == "__main__":
    main()
