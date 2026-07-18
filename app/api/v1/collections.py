"""
collections.py

Collection management API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import get_vector_store
from app.infrastructure.vectorstore.base import BaseVectorStore

router = APIRouter()


@router.get("")
def list_collections(
    store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> dict:
    """List all collections in the vector store."""
    return {"collections": [collection.name for collection in store.list_collections()]}
