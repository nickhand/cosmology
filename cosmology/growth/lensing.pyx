from ..parameters import Cosmology, default_params
from ..evolution import lens_kernel_delta, Dm, H
from ..utils import tools, constants as c
from . import Power

import scipy.integrate as integ
cimport numpy as np
import numpy as np

class LensingPower(object):
    """
    A class to represent a lensing power spectrum, assuming a flat universe.
    """
    
    def __init__(self, nz1,
                       nz2=None,
                       zlim=[0., 10.],
                       cosmo={'default' : default_params, 'flat' : True}, 
                       linear=False,
                       transfer_fit="CAMB",
                       camb_kwargs={}):
        
        self.zmin, self.zmax = zlim
        if self.zmin == 0.: self.zmin = 1e-5
        
        self.nz1 = nz1
        self.nz2 = nz2
        
        # store the cosmology
        if isinstance(cosmo, Cosmology):
            self.cosmo = cosmo
        else:
            self.cosmo = Cosmology(**cosmo)
        if not self.cosmo.flat:
            raise ValueError("Lensing power spectrum only valid for a flat cosmology.")
        
        # we're not actually going to use this but initializing will 
        # help with setting the C globabl variables we need
        self.transfer_fit = transfer_fit
        self._power = Power(z=0., transfer_fit=transfer_fit, 
                            cosmo=self.cosmo, **camb_kwargs)
            
        self.linear = linear
        
        # set the cosmo/transfer global C variables
        self._set_transfer()
    
    #end __init__
    #---------------------------------------------------------------------------
    def _set_transfer(self):
        """
        Set the parameters
        """
        transfer_int = self._power.__dict__['_Power__transfer_int']
        set_parameters(self.cosmo.omegam, self.cosmo.omegab, self.cosmo.omegal, 
                        self.cosmo.omegar, self.cosmo.sigma_8, self.cosmo.h, 
                        self.cosmo.n, self.cosmo.Tcmb, self.cosmo.w, transfer_int)
        if self.transfer_fit == "CAMB":
            self._initialize_CAMB_transfer()

    #---------------------------------------------------------------------------
    def _initialize_CAMB_transfer(self):
        """
        Initialize the CAMB transfer spline.
        """
        # set up the spline
        cdef np.ndarray xarr, yarr 
        
        # compute the CAMB transfer and intialize
        xarr, yarr = self._power.__dict__['_Power__camb_k'], self._power.__dict__['_Power__camb_T']
        xarr = np.ascontiguousarray(xarr, dtype=np.double)
        yarr = np.ascontiguousarray(yarr, dtype=np.double)
        
        set_CAMB_transfer(<double*>xarr.data, <double*>yarr.data, xarr.shape[0])
    
    #---------------------------------------------------------------------------
    @property
    def _z_kern(self):
        try:
            return self.__z_kern
        except AttributeError:
            self.__z_kern = np.logspace(np.log10(self.zmin), np.log10(self.zmax), 1000)
            return self.__z_kern
    #---------------------------------------------------------------------------
    @property
    def _Dm_spline(self):
        try:
            return self.__Dm_spline
        except AttributeError:
            self.__Dm_spline = Dm(self._z_kern, params=self.cosmo)
            return self.__Dm_spline
    #---------------------------------------------------------------------------
    @property
    def _Hz_spline(self):
        try:
            return self.__Hz_spline
        except AttributeError:
            self.__Hz_spline = H(self._z_kern, params=self.cosmo)
            return self.__Hz_spline
    #---------------------------------------------------------------------------
    def _compute_kernel(self, nz):
        """
        Compute the lensing kernel.
        """
        cdef np.ndarray z, Dm_spline, nz_spline, kern
        if np.isscalar(nz):
            kern = lens_kernel_delta(self._z_kern, nz, params=self.cosmo)
            return kern
        else:
            Dm_z   = self._Dm_spline
            H_z    = self._Hz_spline
            clight = c.c_light / (c.km/c.second) # in km/s
            
            A = 1.5*self.cosmo.omegam*self.cosmo.H0**2*(1. + self._z_kern)*Dm_z/clight/H_z

            # now do the integral
            N         = len(self._z_kern)
            kern      = np.ascontiguousarray(np.empty(N), dtype=np.double)
            z         = np.ascontiguousarray(self._z_kern, dtype=np.double)
            Dm_spline = np.ascontiguousarray(Dm_z, dtype=np.double)
            nz_spline = np.ascontiguousarray(nz(self._z_kern), dtype=np.double)
            
            lens_kern_integral(<double *>z.data, N, <double *>z.data, 
                               <double *>nz_spline.data, <double *>Dm_spline.data,
                               N, self.zmax, <double *>kern.data)
            
            # compute the n(z) normalization
            norm = integ.quad(nz, self.zmin, self.zmax)[0]
            
            return A * kern / norm
    #---------------------------------------------------------------------------
    def evaluate(self, ell):
        """
        Evaluate the lensing power spectrum at the multipoles given by `ell`.
        
        Parameters
        ----------
        ell : {float, array_like}
            The multipole number to evaluate the power spectrum at [units: None]
            
        Returns
        -------
        P_ell : {float, array_like}
            The lensing power spectrum at `ell` [units: None]
        """
        cdef np.ndarray integ_spline, l, output, Dm_spline, z_spline   
        ell = tools.vectorize(ell)
             
        # first compute the two kernels
        self.kern1 = self._compute_kernel(self.nz1)
        
        # and do the second kernel, if necessary
        if self.nz2 is not None:
            self.kern2 = self._compute_kernel(self.nz2)
        else:
            self.kern2 = self.kern1
        
        # now set up the integrand spline
        integrand = self._Hz_spline / self._Dm_spline**2 * self.kern1 * self.kern2
        
        # set up the c arrays to pass
        N_ell = len(ell)
        N_spline = len(self.kern1)
        
        output       = np.ascontiguousarray(np.empty(N_ell), dtype=np.double)
        l            = np.ascontiguousarray(ell/self.cosmo.h, dtype=np.double)
        z_spline     = np.ascontiguousarray(self._z_kern, dtype=np.double)
        integ_spline = np.ascontiguousarray(integrand, dtype=np.double)
        Dm_spline    = np.ascontiguousarray(self._Dm_spline, dtype=np.double)
        
        # do the magic
        lens_power_integral(<double *>l.data, N_ell, <double *>z_spline.data, 
                            <double *>integ_spline.data, <double *>Dm_spline.data, 
                            N_spline, self.zmin, self.zmax, self.linear, 
                            <double *>output.data)
        
        clight = c.c_light / (c.km/c.second) # in km/s
        P_ell = output / clight / self.cosmo.h**3 # should be unitless
        
        return P_ell
    #end evaluate
    
    #---------------------------------------------------------------------------    
    
