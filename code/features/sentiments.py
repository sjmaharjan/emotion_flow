import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import nltk
import joblib
import tqdm

__all__ = ['NRCSentimentFlowMeanSD', 'NRCSentimentFlowSum','NRCSentimentFlowPerSentenceDumper']


def load_nrc_emotion_lexicons(path):
    emotion_dic = joblib.load(path)
    return emotion_dic



class NRCSentimentFlow(BaseEstimator, TransformerMixin):
    __resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, chunk_size, saved_features_path=None):
        self.chunk_size = chunk_size
        self.saved_features_path = saved_features_path
        self.loaded = False
        self.nrc_emotion_dict = load_nrc_emotion_lexicons(
            os.path.join(self.__resource_dir, 'nrc_emotion_lexicons_dict.pkl'))

    def load_saved_features(self):
        vectors = None
        if self.saved_features_path:
            vectors = joblib.load(self.saved_features_path)
            print(" Done loading saved feature vectors")

        self.loaded = True

        return vectors

    def get_emotion_types(self):
        return sorted(list(self.nrc_emotion_dict.keys()))

    def _get_sentiments(self, d):
        """
        list of list

        """
        if not self.loaded:
            self.saved_features = self.load_saved_features()

        if self.saved_features and (d.book_id in self.saved_features):
            # print("Reading from dictionary of saved features")

            return self.saved_features[d.book_id]
        else:
            print("Not in saved features so computing ...")
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
        for book_idx, book in tqdm.tqdm(enumerate(books)):
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
    def get_feature_names(self):
        return np.array(
            ['avg_' + ele for ele in self.get_emotion_types()] + ['sd_' + ele for ele in self.get_emotion_types()])

    def feature_size(self):
        return 2 * len(self.get_emotion_types())

    def create_vector(self, X):
        mean = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        return np.hstack((mean, sd))


class NRCSentimentFlowSum(NRCSentimentFlow):
    def __init__(self, chunk_size, saved_features_path, normalize=None):
        super(NRCSentimentFlowSum, self).__init__(chunk_size=chunk_size, saved_features_path=saved_features_path)
        self.normalize = normalize

    def get_feature_names(self):
        return np.array(
            ['avg_' + ele for ele in self.get_emotion_types()] )

    def feature_size(self):
        return len(self.get_emotion_types())

    def create_vector(self, X):
        if self.normalize:
            from sklearn.preprocessing import normalize
            return normalize([np.sum(X, axis=0)], norm=self.normalize)[0]
        else:
            return np.sum(X, axis=0)


class NRCSentimentFlowPerSentenceDumper(NRCSentimentFlow):
    def __init__(self, root):
        self.root = os.path.expanduser(root)

        super(NRCSentimentFlowPerSentenceDumper, self).__init__(chunk_size=1)

    def transform(self, books):
        pass

    def dump(self, books, name):
        dump_path = os.path.join(self.root, name + '.pkl')
        feature_vectors = {}
        for book_idx, book in tqdm.tqdm(enumerate(books)):
            feature_vectors[book.book_id] = self._get_sentiments(book)

        print("Dumping the feature_vector in {}".format(dump_path))
        joblib.dump(feature_vectors, dump_path)


#
# def dump_nrc_sentiment_flow(root, name, corpus):
#     feature_dumper = NRCSentimentFlowPerSentenceDumper(root=root)
#     all_books = corpus.X_train + corpus.X_test
#     print("Total Books {}".format(len(all_books)))
#     feature_dumper.dump(all_books, name=name)
#     print ("Done")



if __name__ == '__main__':
    from collections import namedtuple

    BookDataWrapper = namedtuple('BookDataWrapper', ['book_id', 'content'])

    sent_vectorizer=NRCSentimentFlowSum(chunk_size=1)


    data = [BookDataWrapper(book_id="b1", content='This is a love  violation thwart. What is the shackle? '),
            BookDataWrapper(book_id="b2", content='romance riot . scream? '),
            BookDataWrapper(book_id="b3", content=' love love  violation thwart. Respects is the rod? '),
            BookDataWrapper(book_id="b3", content=' afas. afasfs dalfa.  ')
            ]

    print(sent_vectorizer.fit_transform(data))

