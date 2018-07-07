import click
from config import config
import os
import sys
import logging
import time
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
# Set the path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

app = config[os.getenv('BOOKS_CONFIG') or 'default']


@click.group()
def manager():
    pass


@manager.command()
def dump_vectors():
    from readers.corpus import Corpus
    from readers.goodreads import GoodreadsReader
    from features.utils import fetch_features_vectorized

    data_dir = os.path.join(basedir, '../data')
    split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.from_splitfile(reader=goodread_reader, split_file=split_file)
    print("Done loading data")
    for i, feature in enumerate(app.FEATURES):
        print("Running Feature Extraction for feature  {}".format(feature))
        fetch_features_vectorized(data_dir=app.VECTORS, features=feature, corpus=goodreadscorpus)
        print('Done {}/{}'.format(i + 1, len(app.FEATURES)))
    print('Done....')




@manager.command()
def dump_nrc_sentiment_flow():
    from readers.corpus import Corpus
    from readers.goodreads import GoodreadsReader
    from features.sentiments import NRCSentimentFlowPerSentenceDumper

    data_dir = os.path.join(basedir, '../data/data_1000')  # for all data change path to data_all
    # data_dir = os.path.join(basedir, '../data/data_all')  # for all data change path to data_all
    split_file = os.path.join(basedir, '../data/train_test_split_goodreads.yaml')
    goodread_reader = GoodreadsReader(dirname=data_dir)
    goodreadscorpus = Corpus.from_splitfile(reader=goodread_reader, split_file=split_file)

    print("Done loading data")

    feature_dumper = NRCSentimentFlowPerSentenceDumper(root=app.VECTORS)

    all_books = goodreadscorpus.X_train.tolist() + goodreadscorpus.X_test.tolist()
    print("Total Books {}".format(len(all_books)))
    feature_dumper.dump(all_books, name='nrc_1000')
    print ("Done")




if __name__ == "__main__":
    manager()
