from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from transformers.pipelines import pipeline
# from more_itertools import collapse
from tqdm import tqdm
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


nlp = spacy.load('en_core_web_trf', exclude = ['ner'])

def bertopic_model(texts):
    """
    This function create a BERTopic model object with
        * embedding_model=embeddings (SentenceTransformer),
        * umap_model=umap_model,
        * hdbscan_model=hdbscan_model,
        * vectorizer_model=vectorizer_model,
        * ctfidf_model=ctfidf_model.
    The hyperparameters are also defined: top_n_words=10 and verbose=True.

    This function save the topic model as "bertopic_".
    It also create two html plots for topics and documents.
    """
    umap_model = UMAP(n_neighbors=20, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", max_features=800, ngram_range=(1, 4))
    ctfidf_model = ClassTfidfTransformer()#bm25_weighting=True, reduce_frequent_words=True)
    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embeddings,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,

        top_n_words=10,
        verbose=True)

    topics, probs = topic_model.fit_transform(texts)
    # save model
    topic_model.save("data_models/bertopic_", serialization="pickle")

    embed_docs = embeddings.encode(texts, show_progress_bar=True)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embed_docs)

    # topic visualization
    fig1 = topic_model.visualize_topics()
    fig1.write_html('data_models/topics.html')
    # document visualization
    fig2=topic_model.visualize_documents(texts, reduced_embeddings=reduced_embeddings)
    fig2.write_html("data_models/documents.html")
    # Return model
    return topic_model
