from . import bias
from ..parameters import Cosmology, default_params
from ..evolution import H, mass_to_radius
from ..growth import growth_rate, Correlation
from ..utils import tools

import scipy.integrate as integ
import numpy as np

#-------------------------------------------------------------------------------
def pairwise_velocity(R, z, bias=1., corr_kwargs={}, params=default_params):
    r"""
    Compute the linear theory pairwise velocity at redshift `z` as a 
    function of pair separation [units: km/s]
    
    Notes
    -----
    As given in Eq. 8 of Sheth et al. 2001. Given by
    
    ..math: v_{12}(R, z) = [-2/3 * f * H / (1+z) ] * b * R \bar{\xi}(R) / (1 + b^2 \xi(R) )
    
    Parameters
    ----------
    R : {float, array_like}
        The separations to compute v12 at [units :math: `Mpc h^{-1}`]
    z : float
        The redshift to compute the function at
    bias : float, optional
        The large-scale linear bias to use. Default is 1.
    corr_kwargs : dict, optional
        Keyword arguments to pass to the ``Correlation`` instance. 
    params : {str, dict, Cosmology}, optional
        The cosmological parameters to use. Default is set by the value     
        of ``parameters.default_params``
    
    Returns
    -------
    v12 : {float, array_like}
        The pairwise velocity in km/s
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    R = tools.vectorize(R)
    
    # need the conformal H(z) and growth rate f
    conformalH = H(z, params=params) / (1+z)
    f = growth_rate(z, params=params)
    
    # initialize the correlation object
    corr = Correlation(r=R, z=z, cosmo=params, **corr_kwargs)
    xi = corr.linear_corr
    xi_avg = corr.average_corr
    
    # compute v12 now
    A = -2./3*conformalH*f
    v12 = A * corr.r*xi_avg*bias / (1 + bias**2*xi) / params.h
    
    return v12
#-------------------------------------------------------------------------------
def sigma_evrard(M_Msunh, z, params=default_params):
    """
    Compute the small-scale dark matter halo velocity dispersion in km/s from 
    virial motions using the simulation results from Evrard et al. 2008. 
    
    Parameters
    ----------
    M_Msunh : {float, array_like}
        The halo mass in units of M_sun / h
    z : float
        The redshift to compute the velocity dispersion at. 
    params : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use, specified by the name of a predefined
        cosmology, a parameter dictionary, or a Cosmology class.
        
    Returns
    -------
    sigma : float
        The velocity in km/s
    """
    sigma_DM = 1082.9 # normalization for M = 1e15 M_sun/h (in km/s)
    alpha    = 0.3361 # power law index
        
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    Ez = H(z, params=params)/params.H0
    return sigma_DM * (Ez*M_Msunh/1e15)**alpha
#end sigma_evrard

#-------------------------------------------------------------------------------
def sigma_v2(mf):
    """
    Compute the small-scale velocity dispersion, averaged over all halos, using
    the input ``HaloMassFunction`` instance. 
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
        
    Notes
    -----
    .. math:: \sigma^2_{v^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) v_{\parallel}^2 
    
    Returns
    -------
    sigma_v2 : float
        The velocity in km/s 
    """
    def integrand(lnM):
        M = np.exp(lnM)
        return mf.dndlnm_spline(M)*M*sigma_evrard(M, mf.z, params=mf.cosmo)**2
        
    integral = integ.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return np.sqrt(integral/mf.cosmo.mean_dens)
#end sigma_v2

#-------------------------------------------------------------------------------
def sigma_bv2(mf, bias_model):
    """
    Compute the small-scale velocity dispersion, weighted by halo bias and 
    averaged over all halos, using the input ``HaloMassFunction`` instance.
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
    bias_model : {'Tinker', 'PS', 'SMT'}
        The name of the halo bias model to use, either Tinker, SMT or PS
        
    Notes
    -----
    .. math:: \sigma^2_{bv^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) b(M) v_{\parallel}^2 
    
    Returns
    -------
    sigma_bv2 : float
        The velocity in km/s
    """
    if bias_model not in bias.available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" \
                         %(bias_model, bias.available_bias))
                         
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = getattr(bias, 'bias_%s' %bias_model)
    
    def integrand(lnM):
        M = np.exp(lnM) # in units of M_sun/h
        R = mass_to_radius(M, mf.cosmo.mean_dens) # in Mpc/h
        sigma = mf._power.sigma_r(R, mf.z)
        b = bias_func(sigma, mf.delta_c, *bias_args)
        return mf.dndlnm_spline(M)*M*b*sigma_evrard(M, mf.z, params=mf.cosmo)**2
        
    integral = integ.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return np.sqrt(integral/mf.cosmo.mean_dens)
#end sigma_bv2

#-------------------------------------------------------------------------------
def sigma_bv4(mf, bias_model):
    """
    Compute the small-scale velocity dispersion, weighted by halo bias and 
    averaged over all halos, using the input ``HaloMassFunction`` instance.
    
    Parameters
    ----------
    mf : hmf.HaloMassFunction
        The object holding the relevant halo mass function and related quantities
    bias_model : {'Tinker', 'PS', 'SMT'}
        The name of the halo bias model to use, either Tinker, SMT or PS
        
    Notes
    -----
    .. math:: \sigma^2_{bv^2} = \frac{1}{\bar{\rho}} \int dM M (dN/dM) b(M) v_{\parallel}^4 
    
    Returns
    -------
    sigma_bv4 : float
        The velocity in km/s
    """
    if bias_model not in bias.available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" \
                         %(bias_model, bias.available_bias))
                         
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = getattr(bias, 'bias_%s' %bias_model)
    
    def integrand(lnM):
        M = np.exp(lnM) # in units of M_sun/h
        R = mass_to_radius(M, mf.cosmo.mean_dens) # in Mpc/h
        sigma = mf._power.sigma_r(R, mf.z)
        b = bias_func(sigma, mf.delta_c, *bias_args)
        return mf.dndlnm_spline(M)*M*b*sigma_evrard(M, mf.z, params=mf.cosmo)**4
        
    integral = integ.quad(integrand, np.log(1e8), np.log(1e16), epsabs=0, epsrel=1e-4)[0]
    return (integral/mf.cosmo.mean_dens)**(0.25)
#end sigma_bv4

#-------------------------------------------------------------------------------