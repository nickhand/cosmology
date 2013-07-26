"""
 halo_tools.py
 cosmology: tools for computing halo-related quantities
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 07/25/2013
"""
import numpy as np
import scipy.optimize as opt
from cosmology import cosmology, cosmo
from utils import physical_constants as pc

def virial_overdensity(z, cosmo_params='Planck13'):
    """
    The overdensity corresponding to a virialized halo for a flat universe,
    as taken from Bryan and Norman (1998).
    """
    c = cosmology(cosmo_params)
    x = c.omega_m_z(z) - 1.
    return (18.*np.pi**2 + 82.*x - 39.*x**2) / (1. + x)
#end virial_overdensity

#-------------------------------------------------------------------------------
def concentration(mass, z, definition):
    """
    The concentration as a function of mass and redshift, taken from 
    Duffy et al. 2008 for all clusters and z = 0-2. The definiton of mass
    can be either 200x crit density, 200x background density, or virial.
    
    Parameters
    ----------
    mass : float or numpy.ndarray
        the mass in solar masses
    z : float or numpy.ndarray
        the redshift to compute at
    definition : str
        the definition, either 'critical', 'background', or 'virial'
    """
    h = 0.719 # the WMAP5 h value used in Duffy et al. 2008
    M_pivot = 2.0e12/h # in M_sun
    
    # concentration is 200x critical density
    if definition == 'critical':
        A = 5.71   # +/- 0.12
        B = -0.084 # +/- 0.006
        C = -0.47  # +/- 0.04
    # concentration is 200x mean background density
    elif definition == 'background':
        A = 10.14  # +/- 0.22
        B = -0.081 # +/- 0.006
        C = -1.01  # +/- 0.04
    # virial definition as defined by Bryan and Norman (1998)
    elif definition == 'virial':
        A = 7.85   # +/- 0.12
        B = -0.081 # +/- 0.006
        C = -0.71  # +/- 0.04
    else:
        choices = ['critical', 'background', 'virial']
        raise ValueError("input definition '%s' must be one of %s" %(definition,
                                                                     choices))
    return A*(mass/M_pivot)**B*(1.+z)**C
#end concentration

#-------------------------------------------------------------------------------
def nfw_dimensionless_mass(x):
    """
    The dimensionless function f(x) giving the mass enclosed within a 
    dimensionless radius for an NFW profile. The function is given by: 
    f(x) = x**3 * (ln(1+1/x) - 1/(1+x)) and the mass enclosed is given by
    M_h(r) = 4*pi*rho_s*r_h**3 * f(r/r_h)
    """
    return x**3 * (np.log(1. + 1./x) - 1./(1+x) )
#end nfw_dimensionless_mass

#-------------------------------------------------------------------------------
def convert_halo_mass(mass, z, def_in, def_out, delta_out, cosmo_params='Planck13'):
    """
    Convert between definitions of halo mass, assuming a NFW profile and the 
    concentration relation from Duffy et al. 2008. 
    
    Note : Input mass must be defined either wrt 200x crit density, 200x 
    background density, or virial radius.
    
    Parameters
    ----------
    mass : float or numpy.ndarray
        the input mass in M_sun to convert
    z : float or numpy.ndarray
        the redshift of the halo
    def_in : str
        the input mass definition, either 'critical', 'background', or 'virial'
    def_out : str
        the output mass definition, either 'critical', 'background', or 'virial'
    delta_out : float
        the desired overdensity for the output mass
    cosmo_params: str or dict
        the cosmology parameters to use
    """
    def_choices = ['critical', 'background', 'virial']
    assert(def_in in def_choices)
    assert(def_out in def_choices)
    
    c = cosmology(cosmo_params)
    alpha_out = 1.
    if def_out == 'background':
        alpha_out = c.omega_m_z(z)
    elif def_out == 'critical':
        alpha_out = 1.
    elif def_out == 'virial':
        alpha_out = c.omega_m_z(z)
        delta_out = virial_overdensity(z, cosmo_params)
        
    delta_in = 200.
    alpha_in = 1.
    if def_in == 'background':
        alpha_in = c.omega_m_z(z)
    elif def_in == 'critical':
        alpha_in = 1.
    elif def_in == 'virial':
        alpha_in = c.omega_m_z(z)
        delta_in = virial_overdensity(z, cosmo_params)

    # get the concentration for the input mass
    concen = concentration(mass, z, def_in)
    f_in = nfw_dimensionless_mass(1./concen)
    
    def objective(x):
        return nfw_dimensionless_mass(x) - f_in*delta_out*alpha_out/delta_in/alpha_in
    
    x_final = opt.newton(objective, 1.)
    return (x_final*concen)**(-3.) * alpha_out * delta_out / delta_in / alpha_in * mass
#end convert_halo_mass

#-------------------------------------------------------------------------------
def convert_halo_radius(radius, z, def_in, def_out, delta_out, cosmo_params='Planck13'):
    """
    Convert between definitions of halo radius, assuming a NFW profile and the 
    concentration relation from Duffy et al. 2008. 
    
    Note : Input radius must be defined either wrt 200x crit density, 200x 
    background density, or virial radius.
    
    Parameters
    ----------
    radius : float or numpy.ndarray
        the input radius in Mpc to convert
    z : float or numpy.ndarray
        the redshift of the halo
    def_in : str
        the input mass definition, either 'critical', 'background', or 'virial'
    def_out : str
        the output mass definition, either 'critical', 'background', or 'virial'
    delta_out : float
        the desired overdensity for the output mass
    cosmo_params: str or dict
        the cosmology parameters to use
    """
    c = cosmology(cosmo_params)
    rho_crit = c.rho_crit(z)
    Om = c.omega_m_z(z)
    
    if def_in == 'virial':
        delta_in = virial_overdensity(z, cosmo_params)
        input_mass = 4./3.*np.pi*(radius*pc.mega*pc.parsec)**3*delta_in*Om*rho_crit
        input_mass /= pc.M_sun
    elif def_in == 'background':
        delta_in = 200.
        input_mass = 4./3.*np.pi*(radius*pc.mega*pc.parsec)**3*delta_in*Om*rho_crit
        input_mass /= pc.M_sun
    elif def_in == 'critical':
        delta_in = 200.
        input_mass = 4./3.*np.pi*(radius*pc.mega*pc.parsec)**3*delta_in*rho_crit
        input_mass /= pc.M_sun
        
    output_mass = convert_halo_mass(input_mass, z, def_in, def_out, delta_out, cosmo_params)
    
    if def_out == 'virial':
        delta_out = virial_overdensity(z, cosmo_params)
        output_radius = (3.*output_mass*pc.M_sun/(4*np.pi*delta_out*Om*rho_crit))**(1./3)
    elif def_in == 'background':
        output_radius = (3.*output_mass*pc.M_sun/(4*np.pi*delta_out*Om*rho_crit))**(1./3)
    elif def_in == 'critical':
        output_radius = (3.*output_mass*pc.M_sun/(4*np.pi*delta_out*rho_crit))**(1./3)
        
    return output_radius/pc.mega/pc.parsec
#end convert_halo_radius
    