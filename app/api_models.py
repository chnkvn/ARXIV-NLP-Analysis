from pydantic import BaseModel
from scraping import Paper_abstract
from typing import Union, List, Dict, Tuple

class AbstractsModel(BaseModel):
    item_id: int
    abstracts: list

class TopicModel(BaseModel):
    input_: str
    response: dict[float, list]

class QueryResponseModel(BaseModel):
    query: str
    response: str
    source_documents: List[Tuple]
