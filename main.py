from fastapi import FastAPI
from api.endpoints import router
from core.data_service import IDataService, TxtaiDataService, DataServiceFactory
from txtai import Embeddings
from typing import List
from models.dto import Data

app = FastAPI()

# Load test data
def load_test_data():
    # Define your test data here
    data = [
    "WHO warns of new Covid-19 surge in Europe",
    "Scientists discover new species of deep-sea creatures in the Pacific Ocean",
    "Unemployment rates reach record highs in several countries due to pandemic",
    "Amazon rainforest faces increased deforestation despite conservation efforts",
    "NASA launches new rover to explore Mars' surface for signs of past life",
    "Study finds link between air pollution and increased risk of respiratory diseases",
    "Social media platforms implement stricter policies to combat misinformation",
    "Renewable energy sources continue to gain popularity as fossil fuel emissions rise",
    "Global efforts underway to develop and distribute Covid-19 vaccines",
    "Bitcoin reaches new all-time high, surpassing $50,000 per coin"]

    # data = [
    # "US tops 5 million confirmed virus cases",
    #  "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
    #  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
    #  "The National Park Service warns against sacrificing slower friends in a bear attack",
    #  "Maine man wins $1M from $25 lottery ticket",
    #  "Make huge profits without work, earn up to $100,000 a day"
    #   ];

    return data

# Initialize embeddings instance
embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
llm_embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=True, autoid="uuid5")

# Dependency injection for data service
def get_data_service() -> IDataService:
    return TxtaiDataService(embeddings)

def get_llm_data_service() -> IDataService:
    return TxtaiDataService(llm_embeddings)

# Include routers from endpoints
app.include_router(router)

# Load test data at startup and index it
@app.on_event("startup")
async def startup_event():
    data_service = DataServiceFactory.create(embeddings)  # Create data service
    test_data = load_test_data()  # Load test data
    data_service.index_data(test_data)  # Index test data
    print("Data loaded and indexed successfully")

