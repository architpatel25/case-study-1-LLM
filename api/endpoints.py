from fastapi import APIRouter, Depends
from core.data_service import IDataService, TxtaiDataService
from models.dto import Data
from txtai import Embeddings
from typing import List

router = APIRouter()

@router.post("/index")
def index_data(data: List[Data], service: IDataService = Depends(TxtaiDataService)):
    service.index_data(data)
    return {"message": "Data indexed successfully"}

@router.get("/search")
def search(query: str, service: IDataService = Depends(TxtaiDataService)):
    return service.search(query)

@router.put("/update/{idx}")
def update_data(idx: int, text: str, service: IDataService = Depends(TxtaiDataService)):
    service.update_data(idx, text)
    return {"message": f"Data at index {idx} updated successfully"}

@router.delete("/delete/{idx}")
def delete_data(idx: int, service: IDataService = Depends(TxtaiDataService)):
    service.delete_data(idx)
    return {"message": f"Data at index {idx} deleted successfully"}

@router.get("/rag")
def rag_generate(question: str, service: IDataService = Depends(TxtaiDataService)):
    return service.rag_generate(question)

