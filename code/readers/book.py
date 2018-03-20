from .loader import read_book
from nltk import sent_tokenize
from helpers.decorators import lazy

# label extractors for book object

genre_le = lambda book: book.genre

success_le = lambda book: book.success

avg_rating_le = lambda book: book.avg_rating

genre_success_le = lambda book: "{}_{}".format(book.genre, book.success)


def success_3class_le(book):
    if book.avg_rating >= 3.5:
        return 2  # success
    elif book.avg_rating < 3.5 and book.avg_rating >= 2.5:
        return 1  # mild success
    else:
        return 0  # failure


class Book(object):
    """
    Wraps essential data for features extraction

    """

    def __init__(self, book_path, book_id, genre, success, avg_rating):
        self.book_id = book_id
        self._book_path = book_path
        self.book_title = Book.book_title(self.book_id)
        self.genre = genre
        self.success = success
        self.avg_rating = avg_rating

    @staticmethod
    def book_title(book_id):
        return book_id

    def of_size(self, size):
        content = ' '.join(sent_tokenize(self.content)[:size])
        sub_book = Book(book_id=self.book_id,
                        genre=self.genre, book_path=self._book_path, success=self.success, avg_rating=self.avg_rating)
        sub_book.content = content
        setattr(sub_book, 'size', size)  # set the size attribute for the sub object

        return sub_book

    @property
    @lazy
    def content(self):
        self._content = read_book(self._book_path, encoding='latin1')
        return self._content

    @content.setter
    def content(self, content):
        self._content = content

    @property
    def book2vec_id(self):
        return '%s_%s' % (self.genre, self.book_id)
