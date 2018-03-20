import numpy as np
import yaml
from sklearn.cross_validation import StratifiedShuffleSplit
from .book import success_le


class Corpus:
    def __init__(self, reader,label_extractor=success_le):
        self.corpus_reader = reader
        self.X, self.Y = self.load_data(label_extractor)
        self.nr_instances = self.Y.shape[0]
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    @classmethod
    def from_splitfile(cls, reader, split_file,label_extractor=success_le):
        kls = cls(reader,label_extractor)
        with open(split_file, 'r') as stream:
            try:
                data = yaml.load(stream)
                print ("Total test instances: {} and Training instances {}".format(len(data['test']), len(data['train'])))
                X, Y = kls.X, kls.Y
                X_train, Y_train, X_test, Y_test = [],[],[],[]
                books_added=set()
                for x, y in zip(X, Y):
                    if x.book_id not in books_added:
                        if x.book_id in data['test']:
                            X_test.append(x)
                            Y_test.append(y)
                            books_added.add(x.book_id)
                        elif x.book_id in data['train']:
                            X_train.append(x)
                            Y_train.append(y)
                            books_added.add(x.book_id)
                        else:
                            print('Book id {} not in train and test set'.format(x.book_id))
                kls.X_train, kls.Y_train, kls.X_test, kls.Y_test = np.array(X_train), np.array(Y_train), np.array(
                    X_test), np.array(Y_test)

                print("Total unique books: {}".format(len(books_added)))
                print("Training instances {}, Test instances {}".format(kls.X_train.shape,kls.X_test.shape))
            except yaml.YAMLError as exc:
                print(exc)
                raise OSError("Error reading the split file")
        return kls

    @classmethod
    def with_splits(cls, reader, train_per=0.8, test_per=0.2,label_extractor=success_le):
        kls = cls(reader,label_extractor)
        stratified_split = StratifiedShuffleSplit(kls.Y, n_iter=1, train_size=train_per, test_size=test_per,
                                                  random_state=1234)
        for train_index, test_index in stratified_split:
            kls.X_train, kls.X_test, kls.Y_train, kls.Y_test = kls.X[train_index], kls.X[test_index], kls.Y[
                train_index], kls.Y[test_index]
        print("Training instances {}, Test instances {}".format(kls.X_train.shape,kls.X_test.shape))
        return kls

    def load_data(self,label_extractor):
        X, Y = [], []
        for data in self.corpus_reader:
            X.append(data)
            Y.append(label_extractor(data))


        return np.array(X), np.array(Y)

    def labels(self):

        #TODO fix for regression
        return np.unique(self.Y_train)



