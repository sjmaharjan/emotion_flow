


def read_book(filename, encoding='utf-8'):
    content = ''
    with open(filename, mode='r', encoding=encoding) as f_in:
        content = f_in.read()
    return content

