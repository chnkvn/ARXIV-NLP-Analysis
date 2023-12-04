from pathlib import Path

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=f'{Path.home()}/llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    temperature=0,
    max_tokens=2000,
    n_ctx = 1024,
    top_p=1,
    n_batch = 512,
    callback_manager=callback_manager,
    verbose=True,
    streaming=True,
    stop=['User:'],
    f16_kv=True,
)
embeddings_name = "BAAI/bge-small-en"
embeddings_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embeddings_name, model_kwargs=embeddings_kwargs,
    encode_kwargs=encode_kwargs
)

def metadata_func(record: dict, metadata: dict) -> dict:
    """Extract metadata: paper title and authors name"""
    metadata["title"] = record.get("title")
    metadata["authors"] = ', '. join(record.get("authors"))
    return metadata


def load_documents(path):
    """Load a jsonl file to get the abstracts and their metadata"""
    loader = JSONLoader(
        file_path=path,
        jq_schema=".",
        content_key="abstract",
        json_lines=True,
        metadata_func=metadata_func,
    )

    data = loader.load()
    return data

def build_vector_store(path='data_models/abstracts.jsonl',
                       persist_dir = 'db'):
    """Build the vector store
    - If db is a folder: load it
    - Else, chunk the documents, and create a db. Save it locally in the db folder"""
    if Path(persist_dir).is_dir():
        vectordb = Chroma(persist_directory=persist_dir,
                          embedding_function=embeddings)
    else:
        documents = load_documents(path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                       chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embeddings,
                                         persist_directory=persist_dir)
        vectordb.persist()

    return vectordb


db = build_vector_store()

def format_chat_prompt(message):
    """Format prompt"""
    instruction = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    """
    prompt = f"System:{instruction}"
    prompt = f"{prompt}\nUser: <s>[INST]{message}[/INST]\n"
    return prompt

def semantic_search(query, search_type="similarity"):
    """Send a request to the local LLM using the DB"""
    retriever = db.as_retriever(search=search_type, search_kwargs={"k": 7})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)
    prompt = format_chat_prompt(query)
    response = qa_chain(prompt)
    return response
