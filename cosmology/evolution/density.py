from ..parameters import Cosmology, default_params
from ..utils import constants as c
from .core import H, E

#-------------------------------------------------------------------------------
def critical_density(z, params=default_params, cgs=False):
    """
    The critical mass density at redshift z [units :math: `h^2 M_\odot / Mpc^3`],
    or if `cgs = True`, units are :math: `h^2 g/cm^3`.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the density at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    cgs : bool, optional
        If ``True``, return the density with units of :math: `h^2 g/cm^3`
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    H = 100. * (c.km / c.second / c.Mpc)
    rho_crit_cgs = 3*H**2/(8*c.pi*c.G)
    
    if cgs:
        return rho_crit_cgs
    else:
        return rho_crit_cgs / (c.M_sun/c.Mpc**3)
#end critical_density

#---------------------------------------------------------------------------
def mean_density(z, params=default_params, cgs=False):
    """
    The mean density of the universe at redshift z 
    [units :math: `h^2 M_\odot / Mpc^3`], or if `cgs = True`, 
    units are :math: `h^2 g/cm^3`.
    
    Notes
    -----
    This is given by: 
    
    ..math: \bar{\rho}_m(z) = \Omega_m(z) * \rho_{crit}(z) = \bar{\rho}_m(0) * (1+z)^3
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the density at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    cgs : bool, optional
        If ``True``, return the density with units of :math: `h^2 g/cm^3`
    """
    return omega_m_z(z, params=params) * critical_density(z, params=params, cgs=cgs) 
#end mean_density

#---------------------------------------------------------------------------
def omega_m_z(z, params=default_params):
    """
    The matter density omega_m as a function of redshift.
    
    Notes
    -----
    From Lahav et al. 1991 Eqs 11b-c. This is equivalent to 
    Eq. 10 of Eisenstein & Hu 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the function at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    return params.omegam * (1.+z)**3. / E(z, params=params)**2.
#end omega_m_z

#---------------------------------------------------------------------------
def omega_l_z(z, params=default_params):
    """
    The dark energy density omega_l as a function of redshift.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the function at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    return params.omegal / E(z, params=params)**2
#end omega_l_z
    
#-------------------------------------------------------------------------------
def mass_to_radius(M, mean_dens):
    """
    Calculate radius of a region of space from its mass.
    
    Notes
    ------
    The units of ``M`` don't matter as long as they are consistent with 
    ``mean_dens``
    
    Parameters
    ----------
    M : {float, np.ndarray}
        the masses
        
    mean_dens : float
        the mean density of the universe
        
    Returns
    ------
    R : {float, np.ndarray}
        The corresponding radii to M
    """
    return (3.*M/(4.*c.pi*mean_dens))**(1./3.)
#end mass_to_radius

#-------------------------------------------------------------------------------
def radius_to_mass(R, mean_dens):
    """
    Calculates mass of a region of space from its radius
    
    Notes
    ------
    The units of ``R`` don't matter as long as they are consistent with 
    ``mean_dens``

    Parameters
    ----------
    R : {float, np.ndarray}
        the radii

    mean_dens : float
        the mean density of the universe

    Returns
    ------
    M : {float, np.ndarray}
        The masses corresponding to the radii
    """
    return 4*c.pi*R**3*mean_dens/3
#end radius_to_mass

#-------------------------------------------------------------------------------