"""
 parameters.py
 cosmology: some pre-defined sets of cosmological parameters 
 (e.g. from WMAP, Planck)
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2013
"""
from utils import param_dict

def add_extras(cosmo):
    """
    @brief Sets various additional parameters
    """
    extras = {'Y_He': 0.24,
              'w0' : -1.0,
              'w1' : 0.0,
            }
            
    if 'omega_r_0' not in cosmo.keys():
        extras['omega_r_0'] = 4.15e-5 / cosmo['h']**2
    if 'omega_k_0' not in cosmo.keys():
        extras['omega_k_0'] = 1.-cosmo['omega_m_0'] - \
                                cosmo['omega_l_0'] - extras['omega_r_0']
                                

    cosmo.update(extras)
    return cosmo
#-------------------------------------------------------------------------------
class planck_wp_highL_BAO_2013(param_dict.param_dict):
    """
    Planck + WMAP low ell polarization + highL ACT/SPT data
    (Table 5 of arxiv:arXiv1303.5076)
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

          If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
          = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

          If True, sets neutrino number N_nu = 0, neutrino density
          omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
    
        cosmo = {
             'omega_l_0' : 0.6914,
             'h' : 0.6777,
             'n' : 0.9611,
             'sigma_8' : 0.8288,
             'tau' : 0.0952, 
             'z_reion' : 11.52,
             't_0' : 13.7965,
             }
        omega_b_0 = 0.022161/cosmo['h']**2
        omega_c_0 = 0.11889/cosmo['h']**2
        cosmo['omega_m_0'] = omega_b_0 + omega_c_0
        cosmo['omega_b_0'] = omega_b_0

        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        
        
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class planck_wp_2013(param_dict.param_dict):
    """
    Planck + WMAP low ell polarization data 
    (Table 2 of arxiv:arXiv1303.5076)
    """
    
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_b_0 = 0.0490
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : 0.3183, 
                 'omega_l_0' : 0.6817,
                 'h' : 0.6704,
                 'n' : 0.9619,
                 'sigma_8' : 0.8347,
                 'tau' : 0.0925, 
                 'z_reion' : 11.37,
                 't_0' : 13.8242,
                 }
         
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class WMAP9_eCMB_BAO_H0(param_dict.param_dict):
    """
    WMAP9 + eCMB + BAO + H0 parameters from Hinshaw et al.
    (arxiv:1212.5226v1)
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_c_0 = 0.2402
        omega_b_0 = 0.0463 
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.7135,
                 'h' : 0.693,
                 'n' : 0.961,
                 'sigma_8' : 0.820,
                 'tau' : 0.081, 
                 'z_reion' : 10.1,
                 't_0' : 13.77,
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------    
class WMAP9(param_dict.param_dict):
    """
    WMAP9 parameters from Hinshaw et al.
    (arxiv:1212.5226v1)
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_c_0 = 0.233
        omega_b_0 = 0.0463 
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.721,
                 'h' : 0.700,
                 'n' : 0.972,
                 'sigma_8' : 0.821,
                 'tau' : 0.089, 
                 'z_reion' : 10.6,
                 't_0' : 13.74,
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------        
class WMAP9_BAO_H0_mean(param_dict.param_dict):
    """
    WMAP9 + BAO + H_0 parameters from Hinshaw et al.
    (arxiv:1212.5226v1)
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_c_0 = 0.2408
        omega_b_0 = 0.0472 
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.712,
                 'h' : 0.693,
                 'n' : 0.971,
                 'sigma_8' : 0.830,
                 'tau' : 0.088, 
                 'z_reion' : 10.5,
                 't_0' : 13.75,
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask

#-------------------------------------------------------------------------------
class WMAP7_BAO_H0_mean(param_dict.param_dict):
    """
    WMAP7 + BAO + H_0 parameters from Komatsu et al.
    (ApJSS, 192:18 (47pp), 2011)
    """

    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_c_0 = 0.228
        omega_b_0 = 0.0451 
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.725,
                 'h' : 0.702, 
                 'n' : 0.968,
                 'sigma_8' : 0.816,
                 'tau' : 0.088, 
                 'z_reion' : 10.6,
                 't_0' : 13.76,
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class WMAP7_ML(param_dict.param_dict):
    """
    WMAP7 ML parameters from Komatsu et al. (ApJSS, 192:18 (47pp), 2011)
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.
        """
        omega_c_0 = 0.226
        omega_b_0 = 0.0451
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.729,
                 'h' : 0.703,
                 'n' : 0.966,
                 'sigma_8' : 0.809,
                 'tau' : 0.085,
                 'z_reion' : 10.4,
                 't_0' : 13.79,
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class WMAP5_BAO_SN_mean(param_dict.param_dict):
    """
    WMAP5 + BAO + SN parameters from Komatsu et al. (2009ApJS..180..330K).
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

        Notes
        -----

        From the abstract of the paper:

        The six parameters and the corresponding 68% uncertainties,
        derived from the WMAP data combined with the distance
        measurements from the Type Ia supernovae (SN) and the Baryon
        Acoustic Oscillations (BAO) in the distribution of galaxies,
        are: 

        Omega_B h^2 = 0.02267+0.00058-0.00059, 
        Omega_c h^2 = 0.1131 +/- 0.0034, 
        Omega_Lambda = 0.726 +/- 0.015, 
        n_s = 0.960 +/- 0.013, 
        tau = 0.084 +/- 0.016, and 
        Delata^2 R = (2.445 +/- 0.096) * 10^-9 at k = 0.002 Mpc^-1. 

        From these, we derive 

        sigma_8 = 0.812 +/- 0.026, 
        H0 = 70.5 +/- 1.3 km s^-11 Mpc^-1, 
        Omega_b = 0.0456 +/- 0.0015, 
        Omega_c = 0.228 +/- 0.013, 
        Omega_m h^2 = 0.1358 + 0.0037 - 0.0036, 
        zreion = 10.9 +/- 1.4, and 
        t0 = 13.72 +/- 0.12 Gyr.
        """
        omega_c_0 = 0.228
        omega_b_0 = 0.0456
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.726,
                 'h' : 0.706,
                 'n' : 0.960,
                 'sigma_8' : 0.812,
                 'tau' : 0.084,
                 'z_reion' : 10.9,
                 't_0' : 13.72
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class WMAP5_ML(param_dict.param_dict):
    """
    WMAP5 parameters (using WMAP data alone) from Komatsu et
    al. (2009ApJS..180..330K).
    """
    
    def __init__(flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

        If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
        = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

        If True, sets neutrino number N_nu = 0, neutrino density
        omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

        Notes
        -----

        Values taken from "WMAP 5 Year ML" column of Table 1 of the paper.
        """
        omega_c_0 = 0.206
        omega_b_0 = 0.0432
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.751,
                 'h' : 0.724,
                 'n' : 0.961,
                 'sigma_8' : 0.787,
                 'tau' : 0.089,
                 'z_reion' : 11.2,
                 't_0' : 13.69
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask
#-------------------------------------------------------------------------------
class WMAP5_mean(param_dict.param_dict):
    """
    WMAP5 parameters (using WMAP data alone) from Komatsu et
    al. (2009ApJS..180..330K).
    """
    
    def __init__(self, flat=False, extras=True, ask=False):
        """
        Parameters
        ----------

        flat: boolean

          If True, sets omega_l_0 = 1 - omega_m_0 to ensure omega_k_0
          = 0 exactly. Also sets omega_k_0 = 0 explicitly.

        extras: boolean

          If True, sets neutrino number N_nu = 0, neutrino density
          omega_n_0 = 0.0, Helium mass fraction Y_He = 0.24.

        Notes
        -----

        Values taken from "WMAP 5 Year Mean" of Table 1 of the paper.
        """
        omega_c_0 = 0.214
        omega_b_0 = 0.0441
        cosmo = {'omega_b_0' : omega_b_0,
                 'omega_m_0' : omega_b_0 + omega_c_0,
                 'omega_l_0' : 0.742,
                 'h' : 0.719,
                 'n' : 0.963,
                 'sigma_8' : 0.796,
                 'tau' : 0.087,
                 'z_reion' : 11.0,
                 't_0' : 13.69
                 }
        if flat:
            cosmo['omega_l_0'] = 1. - cosmo['omega_m_0']
            cosmo['omega_k_0'] = 0.0
        if extras:
            add_extras(cosmo)
        for k,v in cosmo.iteritems(): self[k] = v
        self.ask=ask