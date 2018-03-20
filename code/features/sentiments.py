import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import nltk
import joblib

__all__ = ['NRCSentimentFlowMeanSD', 'NRCSentimentFlowSum']


def load_nrc_emotion_lexicons(path):
    emotion_dic = joblib.load(path)
    return emotion_dic




class NRCSentimentFlow(BaseEstimator, TransformerMixin):
    __resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.nrc_emotion_dict = load_nrc_emotion_lexicons(
            os.path.join(self.__resource_dir, 'nrc_emotion_lexicons_dict.pkl'))

    def get_emotion_types(self):
        return sorted(list(self.nrc_emotion_dict.keys()))

    def _get_sentiments(self, d):
        """
        list of list

        """
        scores = []
        for sentence in nltk.sent_tokenize(d.content):
            sentence_emotion_vector = np.zeros(len(self.get_emotion_types()))
            for word in nltk.word_tokenize(sentence.lower()):
                for emo_idx, emotion in enumerate(self.get_emotion_types()):
                    if word in self.nrc_emotion_dict[emotion]:
                        sentence_emotion_vector[emo_idx] += 1.0

            scores.append(sentence_emotion_vector)

        return scores

    def feature_size(self):
        pass

    def fit(self, x, y=None):
        return self

    def create_vector(self, X):
        pass

    def transform(self, books):
        doc_feature_vecs = np.zeros((len(books), self.chunk_size, self.feature_size()), dtype=np.dtype(float))
        for book_idx, book in enumerate(books):
            X = np.array(self._get_sentiments(book))
            # print (X.shape)
            if X.shape[0] < self.chunk_size:

                data_chunks = np.array_split(range(X.shape[0]), X.shape[0])
            else:
                data_chunks = np.array_split(range(X.shape[0]), self.chunk_size)

            for i, ids in enumerate(data_chunks):
                vector = self.create_vector(X[ids])
                doc_feature_vecs[book_idx, i, :] = vector

        return doc_feature_vecs


class NRCSentimentFlowMeanSD(NRCSentimentFlow):
    def feature_size(self):
        return 2 * len(self.get_emotion_types())

    def create_vector(self, X):
        mean = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        return np.hstack((mean, sd))


class NRCSentimentFlowSum(NRCSentimentFlow):
    def __init__(self, chunk_size, normalize=None):
        super(NRCSentimentFlowSum, self).__init__(chunk_size=chunk_size)
        self.normalize = normalize

    def feature_size(self):
        return len(self.get_emotion_types())

    def create_vector(self, X):
        if self.normalize:
            from sklearn.preprocessing import normalize
            return normalize([np.sum(X, axis=0)], norm=self.normalize)[0]
        else:
            return np.sum(X, axis=0)


if __name__ == '__main__':
    from collections import namedtuple

    BookDataWrapper = namedtuple('BookDataWrapper', ['book_id', 'content'])

    # sent_vectorizer=NRCSentimentFlowSum(chunk_size=1)
    sent_vectorizer = NRCSentimentFlowMeanSD(chunk_size=1)

    data = [BookDataWrapper(book_id="b1", content='This is a test violation thwart. What is the shackle? '),
            BookDataWrapper(book_id="b2", content='romance riot . scream? '),
            BookDataWrapper(book_id="b3", content=' Schizophrenia violation thwart. Respects is the rod? '),
            BookDataWrapper(book_id="b3", content=' afas. afasfs dalfa.  ')
            ]

    print(sent_vectorizer.fit_transform(data))
