import math
import utils
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def pre_processing(doc):
    """
    return tokens
    """
    words = fu.get_words(doc, lower_case=True, remove_stop_words=True)
    return words


def compute_tf(word, word_count):    
    return word_count[word]/sum(word_count.values())


def compute_idf(word, all_docs):
    df = 0
    for doc in all_docs:
        if word in doc:
            df += 1
    idf = math.log(len(all_docs) / df) + 1
    return idf


if __name__ == '__main__':
    fu = utils.FileUtil()

    corpus = [
        'Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages.',
        'Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.',
        'The history of natural language processing generally started in the 1950s, although work can be found from earlier periods.'
    ]

    ## use Counter
    tf_idf = defaultdict(lambda: defaultdict(float))
    processed_corpus = [pre_processing(doc) for doc in corpus]
    all_words = set([word for doc in processed_corpus for word in doc])

    for i, doc in enumerate(processed_corpus):
        for word in all_words:
            tf_idf_value = computeTF(word, Counter(doc)) * computeIDF(word, processed_corpus)
            tf_idf[word][i] = tf_idf_value

    result = pd.DataFrame.from_dict(tf_idf)
    print(result)

    ## Use TfidfVectorizer from scikit-learn
    vectorizer = TfidfVectorizer(smooth_idf=False)
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.fit_transform(corpus).toarray())

