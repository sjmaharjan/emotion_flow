from functools import wraps

# A lazy decorator
def lazy(func):
    """ A decorator function designed to wrap attributes that need to be
        generated, but will not change. This is useful if the attribute is
        used a lot, but also often never used, as it gives us speed in both
        situations.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        name = "_" + func.__name__
        try:
            return getattr(self, name)
        except AttributeError:
            value = func(self, *args, **kwargs)
            setattr(self, name, value)
            return value

    return wrapper



