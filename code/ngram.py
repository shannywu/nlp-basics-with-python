import utils
from collections import defaultdict

fu = utils.FileUtil()


def get_ngrams(sentence, n, pad_left=False, pad_right=False):
    words = fu.get_words(sentence)
    for i in range(n-1):
        if pad_left:
            words.insert(0, None)
        if pad_right:
            words.append(None)
    result = [words[i:i+n] for i, word in enumerate(words) if i+n <= len(words)]

    return result


def build_trigram_model(model, sentences):
    for sentence in sentences:
        trigram = get_ngrams(sentence, 3, pad_left=True, pad_right=True)

        for w1, w2, w3 in trigram:
            model[(w1, w2)][w3] += 1

    for w1_w2, w3_cnt in model.items():
        total_cnt = sum(w3_cnt.values())
        for w3, cnt in w3_cnt.items():
            w3_cnt[w3] /= total_cnt

    return model


def get_next_word(w1, w2, model):
    # next word with max freq
    next_word = max(model[(w1, w2)].items(), key=lambda x: x[1])[0]
    return next_word


def generate_sentence(model, w1 = None, w2 = None):
    sentence_end = False
    sentence = [w1, w2]

    while not sentence_end:
        word = get_next_word(w1, w2, model)
        sentence.append(word)

        w1 = w2
        w2 = word

        if sentence[-2:] == [None, None]:
            sentence_end = True

    sentence = list(filter(lambda x: x is not None, sentence))
    return ' '.join(sentence)


if __name__ == '__main__':
    lines = fu.read('data/big_story.txt')
    sentences = fu.get_sentences(lines)

    trigram_model = build_trigram_model(defaultdict(lambda: defaultdict(lambda: 0)), sentences)
    sentence = generate_sentence(trigram_model, None, None)
    print(sentence)

