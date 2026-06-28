from fastapi import APIRouter

from app.core.dependencies import get_vector_store

router = APIRouter()


@router.get("/collections")
def collections():

    store = get_vector_store()

    return {"collections": [collection.name for collection in store.list_collections()]}
