## Architecture Diagram 
![LLM-SemanticSearch](https://github.com/architpatel25/case-study-1-LLM/assets/25317936/a34d4781-28de-411b-beba-91cc4cecb4c0)

## 1. Semantic Search ##
from txtai import Embeddings

# Sample data for indexing  
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
    "Bitcoin reaches new all-time high, surpassing $50,000 per coin"
]
embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
embeddings.index(data)

print("Semantic Search Results:")
for query in ["feel good story", "climate change"]:
    uid = embeddings.search(query, 1)[0][0]
    print(f"Query: {query}, Result: {data[uid]}")
  
## 2. Updates and Deletes ##

udata = data.copy()

uid = embeddings.search("feel good story", 1)[0][0]
print("\nBefore update:", data[uid])

# Update data
udata[0] = "See it: baby panda born"
embeddings.upsert([(0, udata[0], None)])

uid = embeddings.search("feel good story", 1)[0][0]
print("After update:", udata[uid])

# Delete record from index
embeddings.delete([0])

uid = embeddings.search("feel good story", 1)[0][0]
print("After delete:", udata[uid])

## 3. Persistence ##

print("\nPersistence Test:")
print("Before saving index")
uid = embeddings.search("climate change", 1)[0][0]
print("Result:", data[uid])

embeddings.save("index")
embeddings = Embeddings() # Resetting embeddings instance
embeddings.load("index")

print("After loading index")
uid = embeddings.search("climate change", 1)[0][0]
print("Result:", data[uid])

## 4. Keyword Search and Dense Vector index ##

# Create embeddings with subindexes
embeddings = Embeddings(
  content=True,
  defaults=False,
  indexes={
    "keyword": {
      "keyword": True
    },
    "dense": {
      "path": "sentence-transformers/nli-mpnet-base-v2"
    }
  }
)
embeddings.index(data)
print("Keyword & Dense Index Search Results:")
for query in ["NASA", "bitcoin"]:
  print(f"Query: {query}, Keyword Result: ")
  print(embeddings.search(query, limit=1, index="keyword"))
  print("Dense Index Result: ")
  print(embeddings.search(query, limit=1, index="dense"))
  
## 5. Hybrid Search (Sparse + Dense) ##
print("\nHybrid Search Results:")
hybrid_embeddings = Embeddings(hybrid=True, path="sentence-transformers/nli-mpnet-base-v2")
hybrid_embeddings.index(data)
for query in ["public health story", "war"]:
    uid = hybrid_embeddings.search(query, 1)[0][0]
    print(f"Query: {query}, Result: {data[uid]}")
    
## 6. Content Storage for large amount of data ##

print("\nContent Storage Test:")
content_embeddings = Embeddings(content=True, path="sentence-transformers/nli-mpnet-base-v2")
content_embeddings.index(data)
uid = int(content_embeddings.search("wildlife", 1)[0]["id"])
print("Result:", data[uid])

## 7. Create embeddings with a graph index ##
embeddings = Embeddings(
  path="sentence-transformers/nli-mpnet-base-v2",
  content=True,
  functions=[
    {"name": "graph", "function": "graph.attribute"},
  ],
  expressions=[
    {"name": "category", "expression": "graph(indexid, 'category')"},
    {"name": "topic", "expression": "graph(indexid, 'topic')"},
  ],
  graph={
    "topics": {
      "categories": ["health", "climate", "finance", "world politics"]
    }
  }
)

embeddings.index(data)
print("Graph Embeddings Result:")
print(embeddings.search("select topic, category, text from txtai"))

## 8. Using LLM  ##
import torch
from txtai.pipeline import LLM

llm = LLM("google/flan-t5-large", torch_dtype=torch.float32)
query = "Where is one place you'd go in Washington, DC?"
result = llm(query)
print("LLM Standalone Result:")
print("Query: ", query, result)

## 9. RAG (Retrieval-Augmented Generation) ##
from txtai.pipeline import Extractor

llm_embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=True, autoid="uuid5")
llm_embeddings.index(data)

extractor = Extractor(llm_embeddings, "google/flan-t5-large")

llm_query = "What country is having issues with climate change?"
context = lambda question: [{"query": question, "question": f"Answer the following question using the context below.\nQuestion: {question}\nContext:"}]
print("RAG Result:")
print(extractor(context(llm_query))[0])

## 10. Language Model Workflows ##
from txtai import Application

app = Application("embeddings.yml")

# Add data and index
app.add([{"id": idx, "text": text} for idx, text in enumerate(data)])
app.index()

# Execute the workflow
print("\nLanguage Model Workflows Result:")
print(list(app.workflow("search", ["select text from txtai where similar('feel good story') limit 1"])))
print(app.search("select translation(text, 'ta') text from txtai where similar('feel good story') limit 1"))

## OUTPUT::
![Screenshot from 2024-03-26 17-02-53](https://github.com/architpatel25/case-study-1-LLM/assets/25317936/2fa5a860-9403-4571-a611-80736b45b3de)
