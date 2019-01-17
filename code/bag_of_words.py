import utils
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

fu = utils.FileUtil()


def get_vector(sentence, vocabulary):
    sentence_words = fu.get_words(sentence, lower_case=True)
    vector = np.zeros(len(vocabulary))

    for word in sentence_words:
        for i, voc in enumerate(vocabulary):
            if voc == word:
                vector[i] += 1
    return np.array(vector)


def get_vector_sklearn(sentences, example_sentence, num_features):
    vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words = None, max_features = num_features) 
    train_data_features = vectorizer.fit_transform(sentences)
    vector = vectorizer.transform([example_sentence]).toarray()
    return vector


if __name__ == '__main__':
    sentences = [
        'Two roads diverged in a yellow wood,',
        'And sorry I could not travel both',
        'And be one traveler, long I stood',
        'And looked down one as far as I could',
        'To where it bent in the undergrowth;'
    ]

    words = []
    for sentence in sentences:
        words.extend(fu.get_words(sentence, lower_case=True))
    example_sentence = 'The wood is yellow.'

    ## use Counter
    word_counter = Counter(words)
    vocabulary = word_counter.most_common(30)
    vocabulary = [word for word, cnt in vocabulary]
    vector = get_vector(example_sentence, vocabulary)

    ## use sklearn CountVectorize
    vector_sklearn = get_vector_sklearn(sentences, example_sentence, 30)
