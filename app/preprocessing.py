import spacy
nlp = spacy.load('en_core_web_trf')

def sentencize(texts):
    """Sentencize a text using spacy"""
    if isinstance(texts, str):
        # Put the string into a list
        texts = [texts]
    # Sentencize texts
    sentences = [[sent.text.strip() for sent in doc.sents]
                 for doc in nlp.pipe(texts, disable=['ner', 'tagger', 'attribute_ruler', 'lemmatizer'])]
    return sentences
