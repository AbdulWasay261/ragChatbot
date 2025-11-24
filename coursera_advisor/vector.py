from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os


class VectorStoreManager:
    def __init__(self, csv_path="sample_500.csv"):
        self.csv_path = csv_path
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.db_location = "./chroma_langchain_db"
        self.BATCH_SIZE = 100

        self._initialize_vector_store()

    # Initialize vector store
    def _initialize_vector_store(self):
        add_documents = not os.path.exists(self.db_location)

        self.vector_store = Chroma(
            collection_name="coursera_db",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )

        if add_documents:
            print("No existing Chroma DB found. Creating new index...")
            self._load_and_index_documents()
        else:
            print("Chroma DB exists. Loading existing index...")

    # Load CSV → Chunk → Index in Chroma
    def _load_and_index_documents(self):
        print("Reading CSV...")
        df = pd.read_csv(self.csv_path)

        print("Starting document indexing...")
        print("Columns:", df.columns.tolist())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        documents = []

        for i, row in df.iterrows():
            full_text = f"""
Course Name: {row['name']}
Category: {row['category']}

Description:
{row['content']}

What You Will Learn:
{row['what_you_learn']}

Skills Covered:
{row['skills']}
"""
            chunks = text_splitter.split_text(full_text)

            for j, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "name": row["name"],
                            "category": row["category"],
                            "course_id": str(i),
                            "chunk_id": j
                        }
                    )
                )

            # Add in batches
            if len(documents) >= self.BATCH_SIZE:
                print(f"Adding batch of {len(documents)} documents...")
                self.vector_store.add_documents(documents)
                print(f"Current vector count: {self.vector_store._collection.count()}")
                documents = []

        # Add any remaining documents
        if documents:
            print(f"Adding final batch of {len(documents)} documents...")
            self.vector_store.add_documents(documents)

        print(f"Indexing complete. Final vector count: {self.vector_store._collection.count()}")

    # Return retriever for RAG
    def get_retriever(self, k=5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})



vector_manager = VectorStoreManager()

# Use this retriever everywhere in your app
retriever = vector_manager.get_retriever()
query = "global health"
results = retriever.invoke(query)

print("\nSearch Results:")
for i, doc in enumerate(results):
    print(f"\n{i+1}. {doc.metadata.get('name', 'Unknown')}")
    print(f"   {doc.page_content[:200]}...")
