import numpy as np
from contextlib import contextmanager

#-------------------------------------------------------------------------------
def vectorize(x):
    return np.array(x, copy=False, ndmin=1)
#end vectorize

#-------------------------------------------------------------------------------
@contextmanager
def ignored(*exceptions):
    """
    Return a context manager that ignores the specified expections if they
    occur in the body of a with-statement.
    
    For example::
    from contextlib import ignored
    
    with ignored(OSError):
        os.remove('somefile.tmp')
    
    This code is equivalent to:
        try:
            os.remove('somefile.tmp')
        except OSError:
            pass
    
    This will be in python 3.4
    """
    try:
        yield
    except exceptions:
        pass
#end ignored
