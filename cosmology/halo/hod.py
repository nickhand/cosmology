"""
 hod.py
 cosmology: class to implement an Halo Occupation Distribution
 
 original author: Steven-Murray
 edits: Nick Hand
"""
import numpy as np
import scipy.special as sp
import sys
import copy
_allmodels = ["White11", "Zheng05"]


class HOD(object):
    """
    Halo Occupation Distribution model base class.
    
    This class defines three methods -- the average central galaxies, average
    satellite galaxies and total galaxies. 
    
    The total number of galaxies can take two forms: one if there MUST be a
    central galaxy to have a satellite, and the other if not. 
    
    This class should not be called directly. The user
    should call a derived class.
    
    Derived classes of :class:`HOD` should define two methods: :method:`nc` and 
    :method:`ns` (central and satellite distributions respectively).
    Additionally, any parameters of the model should have their names and
    defaults defined as class variables. 
        
    See the derived classes in this module for examples of how to define derived
    classes of :class:`HOD`.
    """
    _defaults = {}

    def __init__(self, central=True, **model_parameters):
        for k in model_parameters:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the HOD" % k)

        self.params = copy.copy(self._defaults)
        self.params.update(model_parameters)
        self._central = central

    def nc(self, M):
        pass

    def ns(self, M):
        pass

    def ntot(self, M):
        if self._central:
            return self.nc(M) * (1.0 + self.ns(M))
        else:
            return self.nc(M) + self.ns(M)

#-------------------------------------------------------------------------------
#end HOD

class White11(HOD):
    """
    The five-parameter HOD model of White et al. (2011). Defaults are the
    BOSS CMASS results from this sample.
    
    
    Parameters
    ----------        
    M1 : float, default = 14.06
        Mass of a halo which on average contains 1 satellite
        
    alpha : float, default = 0.90
        Index of power law for satellite galaxies
        
    sigma : float, default = 0.98
        Width of smoothed cutoff
        
    Mcut : float, default = 13.08
        Minimum mass of halo containing satellites
        
    kappa : float, default = 1.13
        The factor multiplying the cutoff mass
    """
    _defaults = {"M1" : 14.06,
                 "alpha" : 0.90, 
                 "Mcut" : 13.08,
                 "sigma" : 0.98, 
                 "kappa" : 1.13}

    def __init__(self, **kwargs):
        
        # n_tot = n_cen + n_sat
        kwargs['central'] = False
        
        # initialize the base class
        super(White11, self).__init__(**kwargs)
    
    def nc(self, M):
        """
        Number of central galaxies at mass `M` [units: `M_sun/h`]
        """
        nc = 0.5 * sp.erfc( np.log(10**self.params['Mcut'] / M) / (np.sqrt(2.)*self.params['sigma']) )
        return nc

    def ns(self, M):
        """
        Number of satellite galaxies at mass `M` [units: `M_sun/h`]
        """
        ns = np.zeros_like(M)
        inds = M > self.params['kappa'] * 10**self.params["Mcut"]
        ns[inds] = ((M[inds] - self.params['kappa']*10**self.params["Mcut"]) / 10**self.params["M1"])**self.params["alpha"]
        return ns
        
#endclass White11
#-------------------------------------------------------------------------------


class Zheng05(HOD):
    """
    The five-parameter HOD model of Zheng et al. (2005). Defaults are a
    BOSS CMASS-like sample.
    
    
    Parameters
    ----------
    Mmin : float, default = 12.99139
        The minimum mass of halo that supports a central galaxy
        
    M1 : float, default = 14.07659
        Mass of a halo which on average contains 1 satellite
        
    alpha : float, default = 0.8242392
        Index of power law for satellite galaxies
        
    sigma_logm : float, default = 0.3076970
        Width of smoothed cutoff
        
    Mcut : float, default = 13.19778
        Minimum mass of halo containing satellites
    """
    _defaults = {"Mmin" : 12.99139,
                 "M1" : 14.07659,
                 "alpha" : 0.8242392, 
                 "Mcut" : 13.19778,
                 "sigma_logm" : 0.3076970}

    def __init__(self, **kwargs):

        # n_tot = n_cen * (1 + n_sat)
        kwargs['central'] = True
        
        # initialize the base class
        super(Zheng05, self).__init__(**kwargs)
    
    def nc(self, M):
        """
        Number of central galaxies at mass `M` [units: `M_sun/h`]
        """
        nc = 0.5 * (1 + sp.erf((np.log10(M) - self.params["Mmin"]) / self.params["sigma_logm"]))
        return nc

    def ns(self, M):
        """
        Number of satellite galaxies at mass `M` [units: `M_sun/h`]
        """
        ns = np.zeros_like(M)
        inds = M > 10**self.params["Mcut"]
        ns[inds] = ((M[inds] - 10**self.params["Mcut"]) / 10**self.params["M1"])**self.params["alpha"]
        return ns
        
#endclass Zheng05
#-------------------------------------------------------------------------------
