import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

stop_words = set(stopwords.words('english'))


class FileUtil:

    def read(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.strip(), lines))
            lines = list(filter(lambda x: x != '', lines))
        return lines

    def get_sentences(self, lines):
        sentences = []
        for line in lines:
            sentences.extend(sent_tokenize(line))
        return sentences

    def get_words(self, sentence, lower_case=False, remove_stop_words=False):
        words = sentence.split()
        words = list(map(lambda x: re.sub(r'\W+', '', x), words))
        if lower_case:
            words = list(map(lambda x: x.lower(), words))
        if remove_stop_words:
            words = list(filter(lambda x: x not in stop_words, words))
        return words