from app.infrastructure.vectorstore import ChromaVectorStore


class CollectionManager:

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def get_default_collection(self):
        return self.vector_store.get_or_create_collection("atlas")

    def list(self):
        return self.vector_store.list_collections()
