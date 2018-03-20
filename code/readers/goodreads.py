# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from .book import Book
from manage import app
import pandas as pd


class GoodreadsReader(object):
    def __init__(self, dirname, genres=None, meta_file=app.BOOK_META_INFO):
        self.dirname = dirname
        if genres is not None:
            self.genres = genres
        else:
            self.genres = sorted(
                ["Fiction", "Science_fiction", "Detective_and_mystery_stories", "Poetry", "Short_stories",
                 "Love_stories", "Historical_fiction", "Drama"])
        self.book_df = pd.read_excel(meta_file)

    def __iter__(self):
        for genre in self.genres:
            for category in ['failure', 'success']:
                for fid in os.listdir(os.path.join(self.dirname, genre, category)):
                    fname = os.path.join(self.dirname, genre, category, fid)
                    if fid.startswith('.DS_Store') or not os.path.isfile(fname):
                        continue
                    if category.startswith("failure"):
                        success = 0
                    else:
                        success = 1
                    avg_rating = self.book_df[self.book_df['FILENAME'] == fid]['AVG_RATING_2016'].values[0]

                    sentic_file = os.path.join(self.dirname, genre, 'sentic',
                                               fid.replace('.txt', '_st_parser.txt.json'))  # sentic  st_parser
                    if not os.path.exists(sentic_file):
                        raise OSError("Sentic file does not exit: {}".format(sentic_file))

                    stanford_parse_file = os.path.join(self.dirname, genre, 'st_parser',
                                                       fid.replace('.txt', '_st_parser.txt'))
                    if not os.path.exists(stanford_parse_file):
                        raise OSError("Stanford parse file does not exit: {}".format(stanford_parse_file))

                    yield Book(book_path=fname, book_id=fid, genre=genre, success=success,
                               avg_rating=round(avg_rating, 3), sentic_file=sentic_file,
                               stanford_parse_file=stanford_parse_file)
