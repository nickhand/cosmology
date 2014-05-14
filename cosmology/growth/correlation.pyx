from ..parameters import Cosmology, default_params
from . import Power, growth_function

import copy
import numpy as np
cimport numpy as np

#-------------------------------------------------------------------------------
class Correlation(object):
    """
    A class to represent a correlation function in configuration space
    """
    _corr_atts = ['linear_corr', 'nonlinear_corr', 'average_corr']
    
    def __init__(self, r=np.arange(1., 150., 1),
                       z=0., 
                       cosmo={'default' : default_params, 'flat' : True}, 
                       transfer_fit="CAMB",
                       camb_kwargs={}):
        
        # set the r values and redshift
        self.r = np.array(r, copy=False, ndmin=1)
        self.z = z 
         
        # store the cosmology
        if isinstance(cosmo, Cosmology):
            self._cosmo = cosmo
        else:
            self._cosmo = Cosmology(**cosmo)
        
        # initialize the power class
        self._power = Power(k=self._k_spline, z=self.z, transfer_fit=transfer_fit, 
                            cosmo=self._cosmo, **camb_kwargs)
            
    #end __init__ 
    #---------------------------------------------------------------------------
    def _delete_corr(self):
        """
        Delete the correlation attributes
        """
        for att in Correlation._corr_atts:
            delattr(self, att)
    #end _delete_corr
    
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Optimally update quantities
        """
        cpdict = self._cosmo.dict()

        # first update the cosmology
        cp = {k:v for k, v in kwargs.iteritems() if k in Cosmology._cp}
        if cp:
            true_cp = {}
            for k, v in cp.iteritems():
                if k not in cpdict:
                    true_cp[k] = v
                elif k in cpdict:
                    if v != cpdict[k]:
                        true_cp[k] = v
                        
            # delete the entries we've used from kwargs
            for k in cp:
                del kwargs[k]
                
            # now actually update the Cosmology class and Power class
            cpdict.update(true_cp)
            self._cosmo = Cosmology(**cpdict)
            self._power.update(**cpdict)
            
            # delete everything if anything other than sigma_8 changed
            ckeys = true_cp.keys()
            if len(ckeys) > 0:
                self._delete_corr()
                del self.hmf
            
        # now do any other parameters
        for key, val in kwargs.iteritems():
            if hasattr(self._power, key):
                self._power.update(**{key:val})
            try:
                if np.any(getattr(self, key) != val):
                    try:
                        setattr(self, key, val)
                    except:
                        setattr(self, '_' + key, val)
            except AttributeError:
                pass
                            
            # now do the deletions
            self._delete_corr()
            
    #end update
    #---------------------------------------------------------------------------
    @property
    def r(self):
        return self.__r
    
    @r.setter
    def r(self, val):
        self._delete_corr()
        del self._k_spline
        self.__r = val
    
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self.__z 
        
    @z.setter
    def z(self, val):     
        self.__z = val 
        
        if hasattr(self, '_power'):
            self.update
    #---------------------------------------------------------------------------
    @property
    def kmin(self):
        return 0.01 / np.amax(self.r)
    #---------------------------------------------------------------------------
    @property
    def _k_spline(self):
        try:
            return self.__k_spline
        except AttributeError:
            # must have wide enough k region to converge
            kmin = self.kmin
            kmax = 100.
            
            self.__k_spline = np.logspace(np.log10(kmin), np.log10(kmax), 500)
            return self.__k_spline
    
    @_k_spline.deleter
    def _k_spline(self):
        try:
            del self.__k_spline
        except AttributeError:
            pass
    #--------------------------------------------------------------------------
    # CORRELATION ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def linear_corr(self):
        """
        Linear correlation function [units: None]     
        """
        cdef np.ndarray output, klin, Plin, r
        try:
            return self.__linear_corr
        except:   
                     
            # set up C arrays to pass
            output = np.ascontiguousarray(np.empty(len(self.r)), dtype=np.double)
            r      = np.ascontiguousarray(self.r, dtype=np.double)
            klin   = np.ascontiguousarray(self._k_spline, dtype=np.double)
            Plin   = np.ascontiguousarray(self._power.power, dtype=np.double)
            
            
            # call the C function 
            correlation_integral(<double *>r.data, len(self.r), <double *>klin.data,
                                <double *>Plin.data, len(self._k_spline),  self.kmin, 
                                <double *>output.data)
            self.__linear_corr = output
            return self.__linear_corr

    @linear_corr.deleter
    def linear_corr(self):
        try:
            del self.__linear_corr
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def nonlinear_corr(self):
        """
        Nonlinear correlation function [units: None]     
        """
        cdef np.ndarray output, knl, Pnl, r
        try:
            return self.__nonlinear_corr
        except:   
                     
            # set up C arrays to pass
            output = np.ascontiguousarray(np.empty(len(self.r)), dtype=np.double)
            r      = np.ascontiguousarray(self.r, dtype=np.double)
            knl    = np.ascontiguousarray(self._k_spline, dtype=np.double)
            Pnl    = np.ascontiguousarray(self._power.nonlinear_power, dtype=np.double)
            
            
            # call the C function
            correlation_integral(<double *>r.data, len(self.r), <double *>knl.data,
                                <double *>Pnl.data, len(self._k_spline),  self.kmin, 
                                <double *>output.data)
            self.__nonlinear_corr = output
            return self.__nonlinear_corr

    @nonlinear_corr.deleter
    def nonlinear_corr(self):
        try:
            del self.__nonlinear_corr
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def average_corr(self):
        r"""
        Linear correlation function, averaged over a sphere of radius r 
        [units: None] 
        
        Notes
        -----
        Definition taken from Juszkiewicz et al. 1999:
        
        ..math: \bar{\xi}(x, a) = 3 x^{-3} \int_0^x \xi(y, a) y^2 dy
        """
        cdef np.ndarray output, r_prime, xi_prime, r
        try:
            return self.__average_corr
        except:   
                
            # compute a spline for xi(r) for the integral over R
            r_spline   = np.logspace(-2, np.log10(200.), 1000)
            new_corr   = copy.deepcopy(self)
            new_corr.r = r_spline
            xi_spline  = new_corr.linear_corr
            
            # set up C arrays to pass
            output   = np.ascontiguousarray(np.empty(len(self.r)), dtype=np.double)
            r        = np.ascontiguousarray(self.r, dtype=np.double)
            r_prime  = np.ascontiguousarray(r_spline, dtype=np.double)
            xi_prime = np.ascontiguousarray(xi_spline, dtype=np.double)
            
            
            # call the C function
            avg_correlation_integral(<double *>r.data, len(self.r), <double *>r_prime.data,
                                     <double *>xi_prime.data, len(r_spline),
                                     <double *>output.data)
            self.__average_corr = output
            return self.__average_corr

    @average_corr.deleter
    def average_corr(self):
        try:
            del self.__average_corr
        except AttributeError:
            pass
    #---------------------------------------------------------------------------