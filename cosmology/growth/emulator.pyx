#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from ..parameters import Cosmology, default_params
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
cimport numpy as np
    
#-------------------------------------------------------------------------------
cpdef nonlinear_power(object z, object k=None, bint use_cmbh=False, object params=default_params):
    """
    The nonlinear power as computed by the FrankenEmu cosmic emulator. Returns
    the power at the specified `k` [units: `h/Mpc`] in units of `(Mpc/h)^3`

    Parameters
    ----------
    z : float, array_like
        Redshift(s) to compute the growth function at.
    k : float, array_like, optional
        The wavenumber in units of `h/Mpc` to compute the power at.
    use_cmbh : bool, optional
        Whether to use CMB constraints to compute `H0`. Default is `False`.
    params : str, dict, cosmo.Cosmology, optional
        The cosmological parameters to use. Default is Planck 2013 parameters.
        
    Returns 
    -------
    k_nl : float, array_like
        The array of wavenumbers where P(k) is defined in units of `h/Mpc`
    Pk_nl : float, array_like
        The nonlinear matter power spectrum in units of `(Mpc/h)^3`
    """
    # number of k points
    nk = 582
    
    # convert the cosmology to a Cosmology class
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    # check the bounds
    check_cosmo_bounds(params, z, use_cmbh)
    
    cdef np.ndarray[double, ndim=1] ystar = np.empty(nk*2)
    ystar = np.ascontiguousarray(ystar, dtype=np.double)
    
    ombh2 = params.omegab*params.h**2
    ommh2 = params.omegam*params.h**2
    
    # compute the nonlinear power
    emulator_wrapper(ombh2, ommh2, params.n, 100*params.h, params.w, params.sigma_8, z, int(use_cmbh), <double *>ystar.data)
    k_nl, Pk_nl = ystar[:nk]/params.h, ystar[nk:]*params.h**3
    
    if k is None:
        return k_nl, Pk_nl
    else:
        
        # check the bounds of k
        if np.amin(k) < np.amin(k_nl):
            raise ValueError("Cannot compute power for k < %.4f h/Mpc" %np.amin(k_nl))
        if np.amax(k) > np.amax(k_nl):
            raise ValueError("Cannot compute power for k > %.4f h/Mpc" %np.amax(k_nl))
    
        power_spline = spline(k_nl, Pk_nl)
        return k, power_spline(k)
#end nonlinear_power

#-------------------------------------------------------------------------------
def check_cosmo_bounds(params, z, use_cmbh):
    """
    Check that the cosmology parameters are within the allowable bounds for
    the FrankenEmu
    """
    ombh2 = params.omegab*params.h**2
    ommh2 = params.omegam*params.h**2
    
    # omega_m h^2
    if not (0.120  <= ommh2 <= 0.155):
        raise ValueError("Omega_m h^2 must be between [0.12, 0.155] to use nonlinear power emulator.")

    # omega_b h^2
    if not (0.0215 <= ombh2 <= 0.0235):
        raise ValueError("Omega_b h^2 must be between [0.0215, 0.0235] to use nonlinear power emulator.")
    
    # n_s
    if not (0.85 <= params.n <= 1.05):
        raise ValueError("Spectral index must be between [0.85, 1.05] to use nonlinear power emulator.")
    
    # sigma_8
    if not (0.61 <= params.sigma_8 <= 0.9):
        raise ValueError("Sigma_8 must be between [0.61, 0.9] to use nonlinear power emulator.")
    
    # w
    if not (-1.30 <= params.w <= -0.70):
        raise ValueError('Dark energy equation of state must be between [-1.3, -0.7] to use nonlinear power emulator.')
    
    # redshift
    if not (0. <= z <= 1.):
        raise ValueError("Redshift must be between [0, 1] to use nonlinear power emulator.")
        
    if not use_cmbh:
        if not (55.0 <= 100*params.h <= 85.0):
            raise ValueError("H0 must be between [55, 85] to use nonlinear power emulator.")
            
    return 0
#end check_cosmo_bounds

#-------------------------------------------------------------------------------
def galaxy_nonlinear_power(z, k=None, **hod_params):
    """
    The galaxy nonlinear power in real space as computed by the HODEmu cosmic 
    emulator. Returns the power at the specified `k` [units: `h/Mpc`] 
    in units of `(Mpc/h)^3`.
    
    Notes
    -----
    This returns the galaxy power spectrum for a fixed cosmology, close to 
    WMAP7.

    Parameters
    ----------
    z : float, array_like
        Redshift(s) to compute the growth function at.
    k : float, array_like, optional
        The wavenumber in units of `h/Mpc` to compute the power at.
    use_cmbh : bool, optional
        Whether to use CMB constraints to compute `H0`. Default is `False`.
    params : str, dict, cosmo.Cosmology, optional
        The cosmological parameters to use. Default is Planck 2013 parameters.
        
    Returns 
    -------
    k_nl : float, array_like
        The array of wavenumbers where P(k) is defined in units of `h/Mpc`
    Pk_nl : float, array_like
        The nonlinear matter power spectrum in units of `(Mpc/h)^3`
    """
    # h for this cosmology
    h = 0.71
    
    # number of k points
    nk = 2025
        
    # set the hod parameter defaults
    hod_params.setdefault('M1', 14.06)
    hod_params.setdefault('alpha', 0.90)
    hod_params.setdefault('Mcut', 13.08)
    hod_params.setdefault('sigma', 0.98)
    hod_params.setdefault('kappa', 1.13)
    
    # check the hod parameter bounds
    check_hod_bounds(hod_params, z)
    
    cdef np.ndarray[double, ndim=1] ystar = np.empty(nk*2)
    ystar = np.ascontiguousarray(ystar, dtype=np.double)
    
    # compute the galaxy power
    hod_emulator_wrapper(hod_params['Mcut'], hod_params['M1'], hod_params['sigma'], 
                            hod_params['kappa'], hod_params['alpha'], z, <double *>ystar.data)
    k_gal, Pk_gal = ystar[:nk]/h, ystar[nk:]*h**3
    
    if k is None:
        return k_gal, Pk_gal
    else:
        
        # check the bounds of k
        if np.amin(k) < np.amin(k_gal):
            raise ValueError("Cannot compute power for k < %.4f h/Mpc" %np.amin(k_gal))
        if np.amax(k) > np.amax(k_gal):
            raise ValueError("Cannot compute power for k > %.4f h/Mpc" %np.amax(k_gal))
    
        power_spline = spline(k_gal, Pk_gal)
        return k, power_spline(k)
#end galaxy_nonlinear_power

#-------------------------------------------------------------------------------
def check_hod_bounds(hod_params, z):
    """
    Check that the hod parameters are within the allowable bounds for
    the HODEmu
    """
    # Mcut
    if not (12.85 <= hod_params['Mcut'] <= 13.85):
        raise ValueError("Log10(Mcut) must be between [12.85, 13.85] to use galaxy power emulator.")

    # M1
    if not (13.3 <= hod_params['M1'] <= 14.3 ):
        raise ValueError("Log10(M1) must be between [13.3, 14.3] to use galaxy power emulator.")
    
    # sigma
    if not (0.5 <= hod_params['sigma'] <= 1.2):
        raise ValueError("Sigma must be between [0.5, 1.2] to use galaxy power emulator.")
    
    # kappa
    if not (0.5 <= hod_params['kappa'] <= 1.5):
        raise ValueError("Kappa must be between [0.5, 1.5] to use galaxy power emulator.")
    
    # alpha
    if not (0.5 <= hod_params['alpha'] <= 1.5):
        raise ValueError('Alpha must be between [0.5, 1.5] to use galaxy power emulator.')
    
    # redshift
    if not (0. <= z <= 1.):
        raise ValueError("Redshift must be between [0, 1] to use galaxy power emulator.")
        
    return 0
#end check_hod_bounds   

#-------------------------------------------------------------------------------