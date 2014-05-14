from ..parameters import Cosmology, default_params
from ..utils import tools
from ..evolution import omega_m_z, critical_density, mean_density, mass_to_radius
from ..growth import Power

import numpy as np
from scipy.optimize import newton, brentq

#-------------------------------------------------------------------------------
def virial_overdensity(z, params=default_params):
    """
    The overdensity corresponding to a virialized halo for a flat universe, 
    with Omega_m + Omega_l = 1, with respect to the critical density of the 
    universe.
    
    Notes
    -----
    As given in Eq. 6 of Bryan and Norman 1998. Defined with respect to the
    critical density of the universe
    
    Parameters
    ----------
    z : {float, array_like}
        The redshift to compute the virial overdensity at.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    if not params.flat:
        raise ValueError("Virial overdensity only defined for a flat cosmology.")
    
    z = tools.vectorize(z)
    
    # this is omega_m(z) if flat omegam + omegal universe
    Om = params.omegam*(1+z)**3 / (params.omegam * (1+z)**3 + params.omegal)
    
    return (18.*np.pi**2 + 82.*(Om - 1.) - 39.*(Om - 1.)**2)
#end virial_overdensity

#-------------------------------------------------------------------------------
def concentration(mass, z, definition):
    """
    The concentration as a function of mass and redshift, taken from 
    Duffy et al. 2008 for all clusters and z = 0-2. The definiton of mass
    can be either 200x crit density, 200x meandensity, or virial.
    
    Notes
    -----
    Fits taken from the Table 1 of Duffy et al. 2008 and are applicable to 
    the full halo sample from z = 0 to z = 2 for NFW density fits.
    
    Parameters
    ----------
    mass : {float, array_like}
        The mass to compute the concentration at [units :math: `M_\odot h^{-1}`]
    z : {float, array_like}
        The redshift to compute at
    definition : {`critical`, `mean`, `virial`}
        The mass definition
    """
    mass = tools.vectorize(mass)
    z = tools.vectorize(z)
    
    if len(mass) > 1 and len(z) > 1:
        raise ValueError("Cannot compute concentration for multiple masses and redshifts at once.")
        
    M_pivot = 2e12 # in M_sun/h
    
    # concentration is 200x critical density
    if definition == 'critical':
        A = 5.71   # +/- 0.12
        B = -0.084 # +/- 0.006
        C = -0.47  # +/- 0.04
    # concentration is 200x mean background density
    elif definition == 'mean':
        A = 10.14  # +/- 0.22
        B = -0.081 # +/- 0.006
        C = -1.01  # +/- 0.04
    # virial definition as defined by Bryan and Norman (1998)
    elif definition == 'virial':
        A = 7.85   # +/- 0.12
        B = -0.081 # +/- 0.006
        C = -0.71  # +/- 0.04
    else:
        choices = ['critical', 'mean', 'virial']
        raise ValueError("Mass definition '%s' must be one of: %s" %(definition, choices))
            
    c = A * (mass/M_pivot)**B * (1. + z)**C
    
    # check for undefined redshift regions
    inds = np.where(z > 2.0)[0]
    if len(inds) > 0:
        print "Warning: concentration not defined for z > 2"
    c[inds] = np.nan
    
    return c
#end concentration

#-------------------------------------------------------------------------------
def nfw_dimensionless_mass(x):
    """
    The dimensionless function f(x) giving the mass enclosed within a 
    dimensionless radius for an NFW profile. 
    
    Notes
    -----
    As given in Eq. C3 of Hu and Kravtsov 2002. Given by:
    
    ..math: f(x) = x^3 * [ ln(1 + 1/x) - 1/(1+x) ] 
    
    and the mass enclosed at radius r_h is:
    
    ..math: M_h(r_h) = 4*\pi*rho_s*r_h^3 * f(r_s/r_h)
    """
    return x**3 * ( np.log(1. + 1./x) - 1./(1+x) )
#end nfw_dimensionless_mass

#-------------------------------------------------------------------------------
def convert_halo_mass(mass, z, input_definition, output_definition, 
                      output_delta, params=default_params):
    """
    Convert between definitions of halo mass, assuming a NFW profile and the 
    concentration relation from Duffy et al. 2008. 
    
    Notes
    -----
    Input mass must be defined either wrt 200x crit density, 200x 
    mean density, or virial radius.
    
    Parameters
    ----------
    mass : {float, array_like}
        The input mass to convert [units :math: `M_\odot h^{-1}`]
    z : float
        The redshift of the halo
    input_definition : {`critical`, `mean`, `virial`}
        The input mass definition
    output_definition : {`critical`, `mean`, `virial`}
        The output mass definition
    output_delta : float
        The desired overdensity for the output mass.
    params : {str, dict, Cosmology}
        The cosmological parameters to use. Default is set by the value 
        of ``parameters.default_params``
    """
    mass = tools.vectorize(mass)
    
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    # check the input defnitionas
    choices = ['critical', 'mean', 'virial']
    if input_definition not in choices:
        raise ValueError("Input mass definition '%s' must be one of: %s" %(input_definition, choices))
    if output_definition not in choices:
        raise ValueError("Output mass definition '%s' must be one of: %s" %(output_definition, choices))
    
    alpha_out = 1.
    if output_definition == 'mean':
        alpha_out = omega_m_z(z, params=params)
    elif output_definition == 'critical':
        alpha_out = 1.
    elif output_definition == 'virial':
        alpha_out = 1.
        output_delta = virial_overdensity(z, params=params)
        
    input_delta = 200.
    alpha_in = 1.
    if input_definition == 'mean':
        alpha_in = omega_m_z(z, params=params)
    elif input_definition == 'critical':
        alpha_in = 1.
    elif input_definition == 'virial':
        alpha_in = 1.
        input_delta = virial_overdensity(z, params=params)

    # get the concentration for the input mass
    concen = concentration(mass, z, input_definition)
    f_in = nfw_dimensionless_mass(1./concen)
    
    def objective(x):
        return nfw_dimensionless_mass(x) - f_in*output_delta*alpha_out/(input_delta*alpha_in)
    
    x_final = newton(objective, 1.)
    return (x_final*concen)**(-3.) * (alpha_out*output_delta) / (alpha_in*input_delta) * mass
    
#end convert_halo_mass

#-------------------------------------------------------------------------------
def convert_halo_radius(radius, z, input_definition, output_definition, 
                        output_delta, params=default_params):
    """
    Convert between definitions of halo radius, assuming a NFW profile and the 
    concentration relation from Duffy et al. 2008. 
    
    Notes
    ------
    Input radius must be defined either wrt 200x crit density, 200x 
    mean density, or virial radius.
    
    Parameters
    ----------
    radius : {float, array_like}
        The input radius to convert [units :math: `Mpc h^{-1}`]
    z : float
        The redshift of the halo
    input_definition : {`critical`, `mean`, `virial`}
        The input mass definition
    output_definition : {`critical`, `mean`, `virial`}
        The output mass definition
    output_delta : float
        The desired overdensity for the output mass.
    params : {str, dict, Cosmology}
        The cosmological parameters to use. Default is set by the value 
        of ``parameters.default_params``
    """
    radius = tools.vectorize(radius)
    
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    rho_crit = critical_density(z, params=params) # h^2 Msun / Mpc^3
    om_m = omega_m_z(z, params=params)
    conversion_factor = 4./3*np.pi*rho_crit
    
    # compute the input mass scale
    input_delta = 200.
    if input_definition == 'virial':
        input_delta = virial_overdensity(z, params=params)
    
    alpha = 1.
    if input_definition == 'mean':
        alpha = om_m
    input_mass = conversion_factor * alpha * radius**3 * input_delta # in Msun/h
        
    # now convert to an output mass
    output_mass = convert_halo_mass(input_mass, z, input_definition, 
                                    output_definition, output_delta, params=params)
    
    # now convert back to a radius
    if output_delta == 'virial':
        output_delta = virial_overdensity(z, params=params)
    
    alpha = 1.
    if output_definition == 'mean':
        alpha = om_m
    output_radius = (output_mass / (conversion_factor * alpha * output_delta) )**(1./3)

    return output_radius
#end convert_halo_radius

#------------------------------------------------------------------------------- 
@np.vectorize
def nonlinear_mass_scale(z, delta_c=1.686, params=default_params, power_kwargs={}):
    """
    Compute the nonlinear mass scale in :math: `M_\odot h^{-1}`, defined where 
    :math: ``sigma(M, z) = delta_c``.
    
    Parameters
    ----------
    z : {float, array_like}
        The redshift to compute the mass scale at.
    delta_c : float, optional
        The critical collapse fraction. Default is 1.686
    params : {str, dict, Cosmology}, optional
        The cosmological parameters to use. Default is set by the value 
        of ``parameters.default_params``
    power_kwargs : dict, optional
        Keyword arguments to pass to the ``Power`` instance.
        
    Returns
    -------
    M_nl : {float, array_like}
        The nonlinear masses in :math: `M_\odot h^{-1]}`
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    # initialize the power instance
    power = Power(cosmo=params, **power_kwargs)
    rho_mean = mean_density(z, params=params)
    
    def objective(M):
        R = mass_to_radius(M*1e14, rho_mean)
        sigma = power.sigma_r(R, z)
        return sigma - delta_c
        
    M = brentq(objective, 1e-4, 1e5)
    return M*1e14 
#end nonlinear_mass_scale  

#-------------------------------------------------------------------------------
def nfw_rs(M_halo, z, mass_definition, delta_c, params=default_params):
    """
    Compute the scale radius of a NFW profile `r_s` [units :math: `Mpc h^{-1}`] 
    
    Notes
    -----
    As given in Eq. 7 of Yoshida et al (2001). Given by:
    
    ..math: r_s = r_{vir} / c_{vir} = 1/c * (3M / 4 \pi \rho_{crit} \delta_c )
    
    Parameters
    ----------
    M_halo : {float, array_like}
        The halo mass [units :math: `M_\odot h^{-1}`]
    z : float
        The redshift of the halo
    mass_definition : {`critical`, `mean`, `virial`}
        The halo mass definition
    delta_c : float
        The overdensity for the halo mass definition.
    params : {str, dict, Cosmology}, optional
        The cosmological parameters to use. Default is set by the value 
        of ``parameters.default_params``    

    Returns
    -------
    Rs : {float, array_like}
        The NFW scale radius in units of Mpc/h.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    rho_crit = critical_density(z, params=params) # h^2 Msun / Mpc^3
    
    conversion_factor = 4./3*np.pi*rho_crit
    alpha = 1.
    if mass_definition == 'mean':
        alpha = omega_m_z(z, params=params)
        
    if mass_definition == 'virial':
        delta_c = virial_overdensity(z, params=params)
        
    # get the concentration
    c = concentration(M_halo, z, mass_definition)

    # compute r_s in Mpc
    R_halo = (M_halo / (conversion_factor * alpha * delta_c) )**(1./3)
    return R_halo / c
#end nfw_rs

#---------------------------------------------------------------------------    
def nfw_rhos(M_halo, z, mass_definition, delta_c, params=default_params):
    """
    Compute the scale density of a NFW profile `\rho_s` 
    [units :math: `h^2 M_\odot / Mpc^3`] 
    
    Notes
    -----
    As given in Eq. 7 of Yoshida et al (2001). Given by:
    
    ..math: \rho_s = \rho_c * \delta_c / 3 * c^3 / (log(1 + c) - c / (1 + c))
    
    Parameters
    ----------
    M_halo : {float, array_like}
        The halo mass [units :math: `M_\odot h^{-1}`]
    z : float
        The redshift of the halo
    mass_definition : {`critical`, `mean`, `virial`}
        The halo mass definition
    delta_c : float
        The overdensity for the halo mass definition. 
    params : {str, dict, Cosmology}, optional
        The cosmological parameters to use. Default is set by the value     
        of ``parameters.default_params``
    
    Returns
    -------
    rho_s : {float, array_like}
        The NFW scale density in units of :math: `h^2 M_\odot / Mpc^3`
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    rho_crit = critical_density(z, params=params) # h^2 Msun / Mpc^3
    
    alpha = 1.
    if mass_definition == 'mean':
        alpha = omega_m_z(z, params=params)
        
    if mass_definition == 'virial':
        delta_c = virial_overdensity(z, params=params)
        
    # get the concentration
    c = concentration(M_halo, z, mass_definition)

    # compute the scale density
    A = rho_crit*alpha*delta_c
    rho_s =  A*c**3 / ( 3.*( np.log(1. + c) - c/(1.+c) ) )
    
    return rho_s
#end nfw_rhos

#---------------------------------------------------------------------------