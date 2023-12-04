from typing import Union
from uuid import uuid4
from fastapi import FastAPI
from scraping import get_abstracts
from pathlib import Path
import api_models
from topic_modeling import bertopic_model
from bertopic import BERTopic
from rag import semantic_search
from attrs import asdict
app = FastAPI()


with open('data_models/Falk_et_al_2023.txt', 'r') as f:
    text_input = f.read()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/abstracts", response_model =api_models.AbstractsModel, tags=['SCRAPPING'])
def abstracts(n_abstract:int=100, save=False):
    """Abstracts extraction."""
    item_id = uuid4()
    response = get_abstracts(n_abstract, save =False)
    dict_abstracts = [asdict(abstract) for abstract in response]
    return {"item_id": item_id.int, "abstracts": dict_abstracts}

@app.post("/topic", response_model =api_models.TopicModel, tags=['TOPIC_MODELING'])
def predict_topic(text_input: str= text_input):
    """Let an text input (abstract expected), return the words topic"""
    topic_model = BERTopic.load('data_models/bertopic_') if Path('data_models/bertopic_').is_file() else None
    if not topic_model:
        data = [paper['abstract'] for paper in abstracts(100)['abstracts']]
        topic_model = bertopic_model(data)
    inference = topic_model.transform(text_input)
    data = {}
    for topic_nb, prob in zip(inference[0], inference[1]):
        topic = [w[0] for w in topic_model.get_topic(topic_nb)]
        data[prob]=topic
    return {"input_": text_input, "response": data}

@app.post("/rag", response_model =api_models.QueryResponseModel, tags=['RAG'])
def run_semantic_search(query="What is skeleton gait used for?", search_type='similarity'):
    """Use a vector database and a local LLM to answer a query."""
    response = semantic_search(query, search_type)
    sources = [(source.metadata['title'], source.metadata['authors'],source.page_content )
               for source in response['source_documents']]
    return {"query": query, "response": response['result'], "source_documents": sources}
