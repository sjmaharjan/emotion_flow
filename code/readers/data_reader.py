import joblib
from collections import defaultdict
import os
import scipy.sparse as sp
import numpy as np
from keras.utils import np_utils
import pandas as pd

class Dataset:
    TEXT_FEATURE_DUMP = '../vectors/'
    META_DATA= '../data/books_meta_info.tsv'

    def __init__(self, features, label_extractor, genre=True):
        self.features = features
        self.label_extractor = label_extractor
        self.meta_df = pd.read_csv(self.META_DATA, sep='\t', header=0)
        self.data = self.load_data()
        self.use_genre=genre
        self.genres=["Fiction", "Science_fiction", "Detective_and_mystery_stories", "Poetry", "Short_stories",
                 "Love_stories", "Historical_fiction", "Drama"]
        self.g2i= { g:i for i,g in enumerate(sorted(self.genres))}
        self.i2g={i:g for g,i in self.g2i.items() }


    def load_data(self):
        data = defaultdict(lambda: defaultdict(None))
        self.info=defaultdict(None)

        self.info['Y_train'] = self.meta_df[self.meta_df.Split=='Train'][['Actual','Genre']].values
        self.info['train_books'] =  self.meta_df[self.meta_df.Split=='Train']['Books'].values.tolist()

        self.info['Y_test'] = self.meta_df[self.meta_df.Split=='Test'][['Actual','Genre']].values
        self.info['test_books'] =  self.meta_df[self.meta_df.Split=='Test']['Books'].values.tolist()

        self.info['Y_val'] =  self.meta_df[self.meta_df.Split=='Val'][['Actual','Genre']].values
        self.info['val_books'] =  self.meta_df[self.meta_df.Split=='Val']['Books'].values.tolist()

        for feature in self.features:
                target_file = os.path.join(self.TEXT_FEATURE_DUMP, feature + '.pkl')
                target_index_file = os.path.join(self.TEXT_FEATURE_DUMP, feature + '_index.pkl')



                if os.path.exists(target_file):
                    X_train, X_test = joblib.load(target_file)
                    train_books, test_books = joblib.load(target_index_file)

                    data[feature]['X_train'] = X_train
                    data[feature]['train_books'] = train_books

                    data[feature]['X_test'] = X_test
                    data[feature]['test_books'] = test_books

                else:
                    raise ValueError("Feature not found")

        return data

    def get_data(self, fold='train', books=None):
        X_data = {}
        for feature in self.features:
                if fold=='val': fold='train'
                X = []
                data = self.data[feature]['X_' + fold]
                book_info = self.data[feature][fold + '_books']
                sparse = sp.issparse(data)

                for book in books:
                    if book in book_info:
                        book_index = book_info.index(book)
                        if sparse:
                            X.append(data[book_index].toarray()[0])
                        else:
                            X.append(data[book_index])
                    else:
                        raise ValueError("Test book not found")

                X = np.array(X)

                X_data[feature] = X
        return X_data

    def get_training_data(self):
        Y_train = {}


        train_books = self.info['train_books']
        train_book_Ys = self.info['Y_train']

        X_train = self.get_data(fold='train', books=train_books)

        if self.use_genre:
            X_train['genre']=np.array([[self.g2i[genre]] for genre in train_book_Ys[:, 1].ravel()])

        if self.label_extractor == 'success_genre':

            success, genre=self.get_labels(train_book_Ys)

            Y_train['success' + '_output'] = success
            Y_train['genre' + '_output'] = genre
        else:
            Y_train[self.label_extractor + '_output'] = self.get_labels(train_book_Ys)

        return X_train, Y_train

    def get_test_data(self):

        Y_test = {}

        test_books = self.info['test_books']
        test_book_Ys = self.info['Y_test']

        X_test = self.get_data(fold='test', books=test_books)
        if self.use_genre:
            X_test['genre']=np.array([[self.g2i[genre]] for genre in test_book_Ys[:, 1].ravel()])

        if self.label_extractor == 'success_genre':
            success, genre = self.get_labels(test_book_Ys)

            Y_test['success' + '_output'] = success
            Y_test['genre' + '_output'] = genre

        else:
            Y_test[self.label_extractor + '_output'] = self.get_labels(test_book_Ys)

        return X_test, Y_test

    def get_validaiontion_data(self):
        Y_val = {}

        val_books = self.info['val_books']
        val_book_Ys = self.info['Y_val']

        X_val = self.get_data(fold='val', books=val_books)
        if self.use_genre:
            X_val['genre']=np.array([[self.g2i[genre]] for genre in val_book_Ys[:, 1].ravel()])

        if self.label_extractor == 'success_genre':
            success, genre = self.get_labels(val_book_Ys)


            Y_val['success' + '_output'] = success
            Y_val['genre' + '_output'] = genre
        else:

            Y_val[self.label_extractor + '_output'] = self.get_labels(val_book_Ys)

        return X_val, Y_val

    def get_labels(self, Ys):
        if self.label_extractor == 'success':
            return Ys[:, 0]
        elif self.label_extractor == 'genre':
            return np_utils.to_categorical(np.array([self.g2i[genre] for genre in Ys[:, 1].ravel()]))
        elif self.label_extractor == 'success_genre':
            return Ys[:, 0], np_utils.to_categorical(np.array([self.g2i[genre] for genre in Ys[:, 1].ravel()]))

        else:
            raise ValueError("Label not found")
