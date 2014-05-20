import numpy as np
from contextlib import contextmanager
from . import constants as const

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

#-------------------------------------------------------------------------------
def f_sz(nu):
    """
    The frequency dependence of the thermal SZ effect
    
    Parameters
    ----------
    nu : {float, array_like}
        the frequency in GHz
    """
    x = const.h_planck*nu*const.giga / (const.k_b*const.T_cmb)
    return x*(np.exp(x) + 1.) / (np.exp(x) - 1.) - 4.
#end f_sz

#-------------------------------------------------------------------------------
def fourier_top_hat(x):
    """
    A top hat filter in Fourier space
    """
    return 3.*(np.sin(x) - x*np.cos(x)) / x**3
#end fourier_top_hat

#-------------------------------------------------------------------------------
