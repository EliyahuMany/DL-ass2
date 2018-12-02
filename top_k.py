import tools
import numpy as np


def find_similar(word, n):
    word_dict = tools.word_dict('vocab.txt', 'wordVectors.txt')
    u = word_dict[word]
    distances = []
    for x in word_dict:
        calc = cos_distance(u, word_dict[x])
        distances.append([x, calc])
    distances = sorted(distances, key=distance)
    top_k = sorted(distances, key=distance, reverse=True)[1:n + 1]
    top_k = [item[0] for item in top_k]
    return top_k


def cos_distance(u, v):
    d = np.max([float(np.linalg.norm(u, 2) * np.linalg.norm(v, 2)), 1e-8])
    n = np.dot(u, v)
    return n / d


def distance(word):
    return word[1]


def main():
    print("Top 5 most similar to dog: " + str(find_similar("dog", 5)))
    print("Top 5 most similar to England: " + str(find_similar("england", 5))))
    print("Top 5 most similar to John: " + str(find_similar("john", 5)))
    print("Top 5 most similar to explode: " + str(find_similar("explode", 5)))
    print("Top 5 most similar to office: " + str(find_similar("office", 5)))

    if __name__ == "__main__":
        main()
