"""
 parameters.py: this module contains dictionaries with sets of parameters for a
 given cosmology.

 Each cosmology has the following parameters defined:

     ==========  =====================================
     omega_c_0   Omega cold dark matter at z=0
     omega_b_0   Omega baryon at z=0
     omega_m_0   Omega matter at z=0
     flat        Is this assumed flat?  If not, omega_l_0 must be specified
     omega_l_0   Omega dark energy at z=0 if flat is False
     h           Dimensionless Hubble parameter at z=0 in km/s/Mpc
     n           Density perturbation spectral index
     Tcmb_0      Current temperature of the CMB
     Neff        Effective number of neutrino species
     sigma_8     Density perturbation amplitude
     tau         Ionization optical depth
     z_reion     Redshift of hydrogen reionization
     z_star      Redshift of the surface of last scattering
     t0          Age of the universe in Gyr
     w0          The dark energy equation of state
     w1          The redshift derivative of w0
     reference   Reference for the parameters
     ==========  =====================================

 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2013
"""
import warnings, sys
from utils import physical_constants as pc

def Planck13_wBAO():
    """
    Planck 2013 DR1 + lensing + WMAP low ell polarization + highL ACT/SPT data
    (Table 5 of arXiv:1303.5076, best fit)
    """ 
    c = {
            'omega_c_0' : 0.25666,
            'omega_b_0' : 0.048093, 
            'omega_m_0' : 0.30475, 
            'h' : 0.6794,
            'n' : 0.9624,
            'sigma_8' : 0.8271,
            'tau' : 0.0943, 
            'z_reion' : 11.42,
            'z_star' : 1090., 
            't0' : 13.7914,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0., 
            'reference' : "Planck Collaboration 2013, Paper XVI, " + \
                          "arXiv:1303.5076 Table 5 " + \
                          "(Planck + lensing + WP + highL)", 
            'name': 'Planck13'}
            
    return c

def Planck13_wBAO():
    """
    Planck 2013 DR1 + WMAP low ell polarization + highL ACT/SPT data
    (Table 5 of arXiv:1303.5076, best fit)
    """ 
    c = {
            'omega_c_0' : 0.25886,
            'omega_b_0' : 0.048252, 
            'omega_m_0' : 0.30712, 
            'h' : 0.6777,
            'n' : 0.9611,
            'sigma_8' : 0.8288,
            'tau' : 0.0952, 
            'z_reion' : 11.52,
            'z_star' : 1090., 
            't0' : 13.7965,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0., 
            'reference' : "Planck Collaboration 2013, Paper XVI, " + \
                          "arXiv:1303.5076 Table 5 " + \
                          "(Planck + WP + highL + BAO)", 
            'name': 'Planck13'}
            
    return c

#-------------------------------------------------------------------------------
def WMAP9():
    """
    WMAP9 + eCMB + BAO + H0 parameters from Hinshaw et al. 2012
    arxiv:1212.5226, (Table 4, last column)
    """
    c = {
            'omega_c_0' : 0.2402,
            'omega_b_0' : 0.04628, 
            'omega_m_0' : 0.2865, 
            'h' : 0.6932,
            'n' : 0.9608,
            'sigma_8' : 0.820,
            'tau' : 0.081, 
            'z_reion' : 10.1,
            'z_star' : 1091.,
            't0' : 13.772,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True,
            'w0' : -1. ,
            'w1' : 0., 
            'reference' : "Hinshaw et al. 2012, arXiv 1212.5226." + \
                             " Table 4 (WMAP + eCMB + BAO + H0)", 
            'name': 'WMAP9'}

    return c
        
#-------------------------------------------------------------------------------    
def WMAP7():
    """
    WMAP 7 year parameters from Komatsu et al. 2011, ApJS, 192, 18. 
    Table 1 (WMAP + BAO + H0 ML)
    """
    c = {
            'omega_c_0' : 0.226,
            'omega_b_0' : 0.0455, 
            'omega_m_0' : 0.272, 
            'h' : 0.704,
            'n' : 0.967,
            'sigma_8' : 0.810,
            'tau' : 0.085, 
            'z_reion' : 10.3,
            'z_star' : 1091.,
            't0' : 13.76,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0.,
            'reference' : "Komatsu et al. 2011, ApJS, 192, 18. " + \
                             " Table 1 (WMAP + BAO + H0 ML)", 
            'name': 'WMAP7'}
    return c
#-------------------------------------------------------------------------------
def WMAP5():
    """
    WMAP 5 year parameters from Komatsu et al. 2009, ApJS, 180, 330. 
    Table 1 (WMAP + BAO + SN ML).
    """
    c = {
            'omega_c_0' : 0.231,
            'omega_b_0' : 0.0459, 
            'omega_m_0' : 0.277, 
            'h' : 0.702,
            'n' : 0.962,
            'sigma_8' : 0.817,
            'tau' : 0.088, 
            'z_reion' : 11.3,
            'z_star' : 1091.,
            't0' : 13.72,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0.,
            'reference' : "Komatsu et al. 2009, ApJS, 180, 330. " + \
                             " Table 1 (WMAP + BAO + SN ML)", 
            'name': 'WMAP5'}
    return c
#-------------------------------------------------------------------------------
def Matter_Dominated():
    """
    A flat, matter-dominated universe, using Planck 13 parameters for 
    ancilliary parameters
    """
    c = {
            'omega_c_0' : 0.,
            'omega_b_0' : 0., 
            'omega_m_0' : 1.0,
            'omega_r_0' : 0.,  
            'h' : 0.6777,
            'n' : 0.9611,
            'sigma_8' : 0.8288,
            'tau' : 0.0952, 
            'z_reion' : 11.52,
            'z_star' : 1090., 
            't0' : 13.7965,
            'Tcmb_0' : 0.,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0.,  
            'name': 'Matter-Dominated'}
            
    return c
#-------------------------------------------------------------------------------
def Radiation_Dominated():
    """
    A flat, matter-dominated universe, using Planck 13 parameters for 
    ancilliary parameters
    """
    c = {
            'omega_c_0' : 0.,
            'omega_b_0' : 0., 
            'omega_m_0' : 0.,
            'omega_l_0' : 0.,
            'omega_r_0' : 1.0, 
            'h' : 0.6777,
            'n' : 0.9611,
            'sigma_8' : 0.8288,
            'tau' : 0.0952, 
            'z_reion' : 11.52,
            'z_star' : 1090., 
            't0' : 13.7965,
            'Tcmb_0' : 2.72528,
            'Neff' : 3.046, 
            'flat' : True, 
            'w0' : -1. ,
            'w1' : 0.,  
            'name': 'Radiation-Dominated'}
            
    return c
#-------------------------------------------------------------------------------

available = (Planck13, WMAP9, WMAP7, WMAP5, Matter_Dominated, Radiation_Dominated)
default = Planck13
#-------------------------------------------------------------------------------
def get_cosmology_from_string(arg):
    """ 
    Return a cosmology instance from a string.
    """
    try:
        cosmo_func = getattr(sys.modules[__name__], arg)
        cosmo_dict = cosmo_func()
    except AttributeError:
        s = "Unknown cosmology '%s'. Valid cosmologies:\n%s" % (
                arg, [x()['name'] for x in available])
        raise ValueError(s)
    return cosmo_dict
    
#-------------------------------------------------------------------------------
def convert_to_cosmolopy_dict(cosmo_params):
    out = {}
    out['N_nu'] = cosmo_params['Neff']
    out['Y_He'] = 0.24
    out['h'] = cosmo_params['h']
    out['n'] = cosmo_params['n']
    out['omega_M_0'] = cosmo_params['omega_m_0']
    out['omega_b_0'] = cosmo_params['omega_b_0']
    
    if cosmo_params['Tcmb_0'] > 0:
        
        # Compute photon density from Tcmb
        _constant = pc.a_rad / pc.c_light**2
        rho_crit0 = 8.62739108435017e-30
        omega_gam_0 =  _constant*cosmo_params['Tcmb_0']**4 / rho_crit0

        # compute neutrino omega
        # The constant in front is 7/8 (4/11)^4/3 -- see any
        #  cosmology book for an explanation; the 7/8 is FD vs. BE
        #  statistics, the 4/11 is the temperature effect
        omega_nu_0 = 0.2271073 * cosmo_params['Neff'] * omega_gam_0
    else:
        omega_gam_0 = 0.0
        omega_nu_0 = 0.0
    
    if 'omega_r_0' not in cosmo_params.keys():
        omega_r_0 = omega_nu_0 + omega_gam_0
    else:
        omega_r_0 = cosmo_params['omega_r_0']
    
    # compute curvature density
    if cosmo_params.get('flat', False):
        out['omega_lambda_0'] = 1. - out['omega_M_0'] - omega_r_0
        out['omega_k_0'] = 0.
    else:
        out['omega_k_0'] = 1. - out['omega_M_0'] - out['omega_lambda_0'] - omega_r_0
    
    out['omega_n_0'] = omega_nu_0
    out['sigma_8'] = cosmo_params['sigma_8']
    out['t_0'] = cosmo_params['t0']
    out['tau'] = cosmo_params['tau']
    out['z_reion'] = cosmo_params['z_reion']
    
    return out