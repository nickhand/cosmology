"""
 tf_eh.py
 cosmology: class to implement the Eisenstein & Hu 1998 fitting formulae
 for the matter transfer function
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/02/2013
"""

import numpy
from scipy import integrate

import utils.physical_constants as pc

class tf_eh(object):
    """
    A class to implement the Eisenstein & Hu 1998 fitting formulae
     for the matter transfer function. All wavenumbers in Mpc^-1
    """
    
    def __init__(self, cosmo):
        self._set_params(cosmo)
    #end __init__
    
    #---------------------------------------------------------------------------
    def _set_params(self, cosmo):
        """
        Set the various parameters needed for the fitting
        """    
        if cosmo.get('omega_b_0', None) is None: cosmo.omega_b_0 = 1e-8
        
        self.p = {}
        
        omhh      = cosmo.omega_m_0*cosmo.h**2 # matter physical density
        obhh      = cosmo.omega_b_0*cosmo.h**2 # baryon physical density
        theta_cmb = cosmo.Tcmb_0 / 2.7         # normalized temp of cmb
        f_baryon  = obhh / omhh
        
        z_equality = 2.5e4*omhh/theta_cmb**4    # rad-matter equality
        k_equality = 0.0746*omhh/theta_cmb**2 # in Mpc^-1
        
        # when baryons decouple from photons in early universe
        z_drag_b1 = 0.313*omhh**-0.419 * (1.+0.607*omhh**0.674)
        z_drag_b2 = 0.238*omhh**0.223
        z_drag    = 1291*omhh**0.251/(1.+0.659*omhh**0.828) * \
                    (1.+z_drag_b1*obhh**z_drag_b2)
        
        # ratio of baryon to photon momentum
        R_drag     = 31.5*obhh / theta_cmb**4 * (1000. / (1.+z_drag))
        R_equality = 31.5*obhh / theta_cmb**4 * (1000. / (1+z_equality))
        
        # sound horizon (eqn 6)
        sound_horizon = 2./3./k_equality*numpy.sqrt(6./R_equality) * \
                        numpy.log((numpy.sqrt(1.+R_drag)+ \
                        numpy.sqrt(R_drag+R_equality)) / \
                        (1+numpy.sqrt(R_equality)))
        
        # damping scale
        k_silk = 1.6*obhh**0.52 * omhh**0.73 * (1. + (10.4*omhh)**-0.95)
        
        # TF fit parameters
        alpha_c_a1 = (46.9*omhh)**0.67 * (1. + (32.1*omhh)**-0.532)
        alpha_c_a2 = (12.*omhh)**0.424 * (1. + (45.*omhh)**-0.582)
        alpha_c    = alpha_c_a1**-f_baryon * alpha_c_a2**(-f_baryon**3)
        
        beta_c_b1 = 0.944/(1 + (458*omhh)**-0.708)
        beta_c_b2 = (0.395*omhh)**-0.0266
        beta_c    = 1./(1. + beta_c_b1*((1.-f_baryon)**beta_c_b2)-1.)
        
        y = z_equality/(1+z_drag)
        alpha_b_G = y*(-6.*numpy.sqrt(1+y)+(2.+3*y) * \
                    numpy.log((numpy.sqrt(1+y)+1) / (numpy.sqrt(1+y)-1)))
        alpha_b = 2.07*k_equality*sound_horizon*(1+R_drag)**-0.75*alpha_b_G
        
        beta_node = 8.41*omhh**0.435
        beta_b    = 0.5+f_baryon+(3.-2*f_baryon)*numpy.sqrt((17.2*omhh)**2 + 1)
                            
        alpha_gamma = 1.-0.328*numpy.log(431.*omhh)*f_baryon + \
                      0.38*numpy.log(22.3*omhh)*f_baryon*f_baryon
                      
        # store the values
        self.p['f_baryon']      = f_baryon
        self.p['omhh']          = omhh
        self.p['obhh']          = obhh
        self.p['alpha_gamma']   = alpha_gamma
        self.p['k_equality']    = k_equality
        self.p['sound_horizon'] = sound_horizon
        self.p['beta_c']        = beta_c
        self.p['alpha_c']       = alpha_c
        self.p['beta_node']     = beta_node
        self.p['alpha_b']       = alpha_b
        self.p['beta_b']        = beta_b
        self.p['k_silk']        = k_silk
    #end __set_params                  
    
    #---------------------------------------------------------------------------
    def full(self, k):
        """
        Returns the value of the full transfer function fitting formula.
        This is the form given in Section 3 of Eisenstein & Hu (1998).
        k is the wavenumber at which to calculate transfer function, in Mpc^-1.
        """
        if self.p['obhh'] <= 1e-8:
            raise ValueError("using transfer function that includes "+ \
                "baryons without nonzero omega_b_0")
                
        k  = abs(k)
        q  = k/13.41/self.p['k_equality']
        xx = k*self.p['sound_horizon']
        
        T_c_ln_beta   = numpy.log(numpy.e + 1.8*self.p['beta_c']*q)
        T_c_ln_nobeta = numpy.log(numpy.e + 1.8*q)
        T_c_C_alpha   = 14.2/self.p['alpha_c'] + 386./(1.+69.9*q**1.08)
        T_c_C_noalpha = 14.2 + 386./(1.+69.9*q**1.08)
    
        T_c_f = 1./(1. + (xx/5.4)**4)
        T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*q*q) + \
                (1.-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*q*q)
                
        s_tilde  = self.p['sound_horizon'] * \
                    (1.+(self.p['beta_node']/xx))**(-1./3)
        xx_tilde = k*s_tilde
        
        T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*q*q)
        T_b = numpy.sin(xx_tilde)/xx_tilde * (T_b_T0/(1.+(xx/5.2)**2) + \
                self.p['alpha_b']/(1+(self.p['beta_b']/xx)**3) * \
                numpy.exp(-(k/self.p['k_silk'])**1.4))
        
        T_full = self.p['f_baryon']*T_b + (1.-self.p['f_baryon'])*T_c
        return T_full
    #end full_tf
    
    #---------------------------------------------------------------------------
    def no_wiggles(self, k):
        """
        The value of an approximate transfer function that captures the
        non-oscillatory part of a partial baryon transfer function. The 
        baryon oscillations are left out, but the suppression of power below
        the sound horizon is included.
        See equations 30 and 31 of Eisenstein & Hu (1998).
        k is the wavenumber at which to calculate transfer function, in Mpc^-1.
        """
        if self.p['obhh'] <= 1e-8:
            raise ValueError("using transfer function that includes "+ \
                "baryons without nonzero omega_b_0")
                
        q  = k/13.41/self.p['k_equality']
        xx = k*self.sound_horizon_fit(self.p['omhh'], self.p['f_baryon'], 1.0)
        
        ag = self.p['alpha_gamma']
        gamma_eff = self.p['omhh']*(ag+(1.-ag)/(1. + (0.43*xx)**4))
        q_eff = q*self.p['omhh']/gamma_eff
        
        T_nowiggles_L0 = numpy.log(2.*numpy.e + 1.8*q_eff)
        T_nowiggles_C0 = 14.2 + 731./(1. + 62.5*q_eff)
        return T_nowiggles_L0/(T_nowiggles_L0+T_nowiggles_C0*q_eff*q_eff)
    #end no_wiggles
    
    #---------------------------------------------------------------------------
    def no_baryons(self, k):
        """
        The value of the transfer function for a zero-baryon universe, 
        as fit in Eisenstein and Hu 1998
        k is the wavenumber at which to calculate transfer function, in Mpc^-1.
        """
        q  = k/13.41/self.p['k_equality']
        T_0_L0 = numpy.log(2.*numpy.e + 1.8*q)
        T_0_C0 = 14.2 + 731./(1. + 62.5*q)
        return T_0_L0 / (T_0_L0 + T_0_C0*q*q)
    #end no_baryons
    
    #---------------------------------------------------------------------------
    def sound_horizon_fit(self, omega_m_0, f_baryon, hubble):
        """
        The approximate value of the sound horizon, in h^-1 Mpc
        """
        omhh = omega_m_0*hubble*hubble
        sound_horizon_fit_mpc = 44.5*numpy.log(9.83/omhh) / \
                            numpy.sqrt(1. + 10.*(omhh*f_baryon)**0.75)
        return sound_horizon_fit_mpc*hubble
    #end sound_horizon_fit
    
    #---------------------------------------------------------------------------