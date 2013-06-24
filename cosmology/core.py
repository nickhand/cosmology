"""
 core.py
 cosmology: tools for computing quantities depending on redshift
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2013
"""
import numpy
from scipy import integrate

from cosmology import cosmo, parameters
from utils import pytools
import utils.samplinator as s
import utils.physical_constants as pc

class cosmology(s.with_sampleable_methods):
    """
    Used to automate cosmological computations of quantities that
    evolve with redshift (distances, times).
    
    Notes
    -----
    All distance are in Mpc and all times come are in Gyr.
    Default parameters are the Planck 2013 data.
    """
    _allowable_params = [ 'omega_c_0', 
                          'omega_b_0', 
                          'omega_m_0', 
                          'flat', 
                          'omega_l_0', 
                          'h', 
                          'n',
                          'Tcmb_0', 
                          'Neff', 
                          'sigma_8', 
                          'tau', 
                          'z_reion', 
                          't0', 
                          'w0',
                          'w1',
                          'name', 
                          'reference']
                          
    _default_params = parameters.Planck13()
                          
    def __init__(self, cosmo_dict=None):
        """
        
        Parameters
        ----------
        cosmo_dict : dict
            allowable keys are: 
            omega_c_0   Omega cold dark matter at z=0
            omega_b_0   Omega baryon at z=0
            omega_m_0   Omega matter at z=0
            flat        Is this assumed flat? If not, omega_l_0 must be specified
            omega_l_0   Omega dark energy at z=0 if flat is False
            h           Dimensionless Hubble parameter at z=0 in km/s/Mpc
            n           Density perturbation spectral index
            Tcmb_0      Current temperature of the CMB
            Neff        Effective number of neutrino species
            sigma_8     Density perturbation amplitude
            tau         Ionization optical depth
            z_reion     Redshift of hydrogen reionization
            t0          Age of the universe in Gyr
            w0          The dark energy equation of state
            w1          The redshift derivative of w0
            reference   Reference for the parameters
        """
        
        if cosmo_dict is None: 
            print("Warning: No default cosmology has been specified, "
                          "using Planck 2013 parameters.")
            cosmo.unify(parameters.Planck13())
        else:
            self.set_current(cosmo_dict)
            
        # verify and set params
        self._verify_params()
        self._set_extras()
    #end __init__
    
    #---------------------------------------------------------------------------
    def _verify_params(self):
        """
        Verify the input cosmology parameters
        """
        # remove any params that shouldn't be there
        for k in cosmo.keys():
            if k not in self._allowable_params: cosmo.pop(k, None)
        
        # determine if we are using flat cosmology
        flat = False
        if 'flat' in cosmo.keys():
            if cosmo.flat: flat = True
        else:
            if default_params['flat']: flat = True
        
        # replace missing with default params
        used_default = False
        for k in self._allowable_params:
            if k is 'reference' or k is 'name': continue
            if k is 'omega_l_0':
                if flat: continue
            if k not in cosmo.keys():
                used_default = True
                cosmo[k] = self._default_params[k]
                
        # print a warning
        if used_default:
            print("Warning: Missing cosmology parameters, "
                          "using Planck 2013 parameters for these.")
                          
    #end _verify_params      
        
    #---------------------------------------------------------------------------
    def _set_extras(self):
        """
        Set the extra cosmological parameters
        """

        self._H0 = 100.*cosmo.h
        self._hubble_time = pc.mega*pc.parsec/(self._H0*pc.kilo*pc.meter) \
                            /(pc.giga*pc.year)
        self._hubble_distance = pc.c_light/(pc.kilo*pc.meter) / self._H0
        
        
        # compute photon density, Tcmb, neutrino parameters
        # Tcmb_0 = 0 removes both photons and neutrinos, is handled
        # as a special case for efficiency
        if cosmo.Tcmb_0 > 0:
            
            # Compute photon density from Tcmb
            _constant = pc.a_rad / pc.c_light**2
            self._omega_gam_0 =  _constant*cosmo.Tcmb_0**4 / self.rho_crit(0.)

            # compute neutrino omega
            # The constant in front is 7/8 (4/11)^4/3 -- see any
            #  cosmology book for an explanation; the 7/8 is FD vs. BE
            #  statistics, the 4/11 is the temperature effect
            self._omega_nu_0 = 0.2271073 * cosmo.Neff * self._omega_gam_0
        else:
            self._omega_gam_0 = 0.0
            self._omega_nu_0 = 0.0
        
        self._omega_r_0 = self._omega_nu_0 + self._omega_gam_0

        # compute curvature density
        if cosmo.flat:
            cosmo.omega_l_0 = 1. - cosmo.omega_m_0 - self._omega_r_0
            self._omega_k_0 = 0.
        else:
            self._omega_k_0 = 1. - cosmo.omega_m_0 - cosmo.omega_l_0 - \
                                                                self._omega_r_0
                                                                
        # matter-radiation equality
        if self._omega_r_0 == 0:
            self._z_rm = numpy.inf
        else:
            self._z_rm = (cosmo.omega_m_0 / self._omega_r_0) - 1

        if cosmo.omega_m_0 == 0:
            self._a_rm = numpy.inf
        else:
            self._a_rm = self._omega_r_0 / cosmo.omega_m_0
    #end _set_extras
   
    #-------------------------------------------------------------------------------
    def get_current(self):
        """ 
        Get the current cosmology. The default is the planck 2013 parameters.
        """
        return cosmo
    #end get_current
    
    #-------------------------------------------------------------------------------
    def set_current(self, cosmo_dict):
        """ 
        Set the current cosmology.

        Call this with an empty string ('') to get a list of the strings
        that map to available pre-defined cosmologies.

        Parameters
        ----------
        cosmo_dict : str or dict 
            the cosmology to use
        """
        if cosmo_dict == "":
             print("Valid cosmologies:\n%s" 
                        %([x()['name'] for x in parameters.available]))
             return
             
        if isinstance(cosmo_dict, basestring):
            _current = parameters.get_cosmology_from_string(cosmo_dict)
        elif isinstance(cosmo_dict, dict):
            _current = cosmo_dict
        else:
            raise ValueError(
                "Argument must be a string or dictionary. Valid strings:"
                "\n%s" %available)
        
        # make sure to remove omega_c_0 and omega_b_0 if omega_m_0 is 
        # specified and they are not
        keys = _current.keys()
        if 'omega_m_0' in keys:
            if 'omega_c_0' not in keys and 'omega_b_0' not in keys:
                _current['omega_c_0'] = None
                _current['omega_b_0'] = None

        if 'name' not in keys or 'reference' not in keys:
            _current['name'] = None
            _current['reference'] = None
        
        cosmo.update(_current)
        
        # verify and set params
        self._verify_params()
        self._set_extras()
    #end set_current
        
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def _E(self, z):
        """
        The unitless Hubble expansion rate at redshift z, 
        modified to include non-constant w parameterized linearly 
        with z ( w = w0 + w1*z )
        """
        if z == numpy.inf: return numpy.inf
        return numpy.sqrt(cosmo.omega_m_0 * (1. + z)**3 \
                          + self._omega_r_0 * (1. + z)**4 \
                          + self._omega_k_0 * (1.+ z)**2 \
                          + cosmo.omega_l_0 * numpy.exp(3*cosmo.w1*z) \
                          * (1.+ z)**(3 * (1 + cosmo.w0 - cosmo.w1)))
    #end _E
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def H(self, z):
        """
        The value of the Hubble constant at redshift z in km/s/Mpc
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return 100. * cosmo.h * self._E(z)
    #end H
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def w(self,z):
        """
        The dark energy equation of state: w(z) = w0 + w1*z
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return cosmo.w0 + cosmo.w1 * z
    #end w
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def a(self, z):
        """
        The scale factor at redshift z
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return 1. / (1. + z)
    #end a
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def lookback_time(self, z):
        """
        The lookback time, defined as the difference between the age 
        of the Universe now and the age at z
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        f = lambda z: 1/(1.+z)/self._E(z)
        I = integrate.quad(f, 0, z)
        return self._hubble_time * I[0]
    #end lookback_time
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def age(self, z):
        """
        The age of the universe at redshift z in Gyr.
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        f = lambda z: 1/(1.+z)/self._E(z)
        I = integrate.quad(f, z, numpy.inf)
        return self._hubble_time * I[0]
    #end age
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def Dc(self, z):
        """
        The line of sight comoving distance in Mpc (eqn 15 from Hogg 1999).
        Remains constant with epoch if objects are in the Hubble flow
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        if z==0:
            return 0
        else:
            f = lambda z: 1.0/self._E(z)
            I = integrate.quad(f, 0, z)
            return self._hubble_distance * I[0]
    #end Dc
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def Dp(self, z):
        """
        The proper or physical distance
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return self.Dc(z)/(1.+z)
    #end Dp
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def Dm(self, z):
        """
        The transverse comoving distance in Mpc (eqn 16 from Hogg 1999).
        At same redshift but separated by angle dtheta;
        Dm * dtheta is transverse comoving distance
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        sOk = numpy.sqrt(numpy.abs(self._omega_k_0))
        Dh = self._hubble_distance
        if self._omega_k_0 < 0.0:
            return Dh*numpy.sin(sOk * self.Dc(z)/Dh ) / sOk
        elif self._omega_k_0 == 0.0:
            return self.Dc(z)
        else:
            return Dh * numpy.sinh(sOk * self.Dc(z)/Dh ) / sOk
    #end Dm
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def Da(self, z):
        """
        The angular diameter distance in Mpc (eqn 18 from Hogg 1999).
        Ratio of an objects physical transvserse size to its
        angular size in radians
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return self.Dm(z) / (1.+ z)
    #end Da
    
    #---------------------------------------------------------------------------
    def Da12(self, z1, z2):
        """
        The angular diameter distance between objects at 2 redshifts in Mpc.
        Useful for gravitational lensing (eqn 19 of Hogg 1999)

        Parameters
        ----------
        z1 : float
            the first redshift
        z1 : float
            the second redshift
        """
        # does not work for negative curvature
        assert(self._omega_k_0) >= -1e4

        # z1 < z2
        if (z2 < z1):
            z1,z2 = z2,z1

        Dm1 = self.Dm(z1)
        Dm2 = self.Dm(z2)
        Ok  = self._omega_k_0
        Dh  = self._hubble_distance

        return 1. / (1 + z2) * ( Dm2 * numpy.sqrt(1. + Ok * Dm1**2 / Dh**2)\
                                 - Dm1 * numpy.sqrt(1. + Ok * Dm2**2 / Dh**2) )
    #end Da12
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def Dl(self, z):
        """
        The luminosity distance in Mpc (eqn 21 of Hogg 1999)
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return (1. + z) * self.Dm(z)
    #end Dl
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def D_hor(self, z):
        """
        The horizon distance at redshift z in Mpc (eqn)
        (physical dist that light can travel from z' = infty to z' = z)
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        f = lambda zp : 1 / self._E(zp)
        I = integrate.quad(f, z, numpy.inf)
        return self._hubble_distance * I[0]
    #end D_hor

    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def mu(self, z):
        """
        The distance modulus
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return 5. * numpy.log10(self.Dl(z) * pc.mega) - 5.
    #end mu

    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def dVc(self, z):
        """
        The differential comoving volume element dV_c/dz/dSolidAngle.
        Dimensions are volume per unit redshift per unit solid angle.
        Units are Mpc**3 Steradians^-1. From eqn 28 of Hogg 1999.
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        Dm = self.Dm(z)
        E = self._E(z)
        return self._hubble_distance * Dm**2. / E
    #end dVc
    
    #----------------------------------------------------------------------------
    @pytools.call_as_array
    def Vc(self, z):
        """
        The comoving volume out to redshift z in Mpc^3. 
        From eqn 29 of Hogg 1999.
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        Ok = self._omega_k_0
        Dh = self._hubble_distance
        sOk = numpy.sqrt(numpy.abs(Ok))
        Dm = self.Dm(z)
        if Ok < 0.0:
            norm = 4*numpy.pi*Dh**3. / (2.*Ok)
            return norm*(Dm/Dh*numpy.sqrt(1+Ok*(Dm/Dh)**2) - \
                    numpy.arcsin(sOk * Dm/Dh ) / sOk )
        elif Ok == 0.0:
            return 4*numpy.pi*Dm**3 / 3.
        else:
            norm = 4*pi*Dh**3/(2.*Ok)
            return norm*(Dm/Dh*numpy.sqrt(1+Ok*(Dm/Dh)**2) - \
                    numpy.arcsinh(sOk * Dm/Dh ) / sOk)
    #end Vc

    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def rho_crit(self, z):
        """
        The critical (mass) density at redshift z in g/cm^3
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at.
        """
        if z == 0.: 
            H =  self._H0
        else:
            H = self.H(z)
        return 3.*(H*pc.kilo*pc.meter/pc.mega/pc.parsec)**2 / \
                    (8 * numpy.pi * pc.G)             
    #end rho_crit
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def rho_mean(self, z):
        """
        The mean density of the universe, at redshift z in g/cm^3
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at.
        """
        return self.omega_m_z(z) * self.rho_crit(z) 
    #end rho_mean
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def e_crit(self, z):
        """
        The critical (energy) density at redshift z in erg/cm^3
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at.
        """
        return self.rho_crit(z) * pc.c_light**2
    #end e_crit
               
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def lens_kernel(self, z, z_source):
        """
        The value of the lens kernel at a redshift z, given a
        source at redshift z_source

        units are strange: cm Mpc^2 / g
        
        Parameters
        ----------
        z : float
            the redshift to compute the lens kernel at
        z_source : float
            the redshift of the source
        """
        Ds = self.Dc(z_source)
        D = self.Dc(z)

        return 4*numpy.pi*pc.G / (pc.c_light)**2 * self.Dh / self._E(z) * \
               D*(Ds-D)/Ds / (1.+z)**2
    #end lens_kernel
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def omega_m_z(self, z):
        """
        The matter density omega_m as a function of redshift

        From Lahav et al. 1991 equations 11b-c. This is equivalent to 
        equation 10 of Eisenstein & Hu 1999.
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return cosmo.omega_m_0 * (1.+z)**3. / self._E(z)**2.
    #end omega_m_z
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def omega_l_z(self, z):
        """
        The dark energy density omega_l as a function of redshift
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return cosmo.omega_l_0 / self._E(z)**2
    #end omega_l_z
        
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def angular_size(self, z, diameter):
        """
        The angular size of an object of input diameter and at redshift z
        
        Parameters
        ----------
        z : float
            the redshift to compute angular size at 
        diameter :  float
            the physical diameter of the object in Mpc

        Returns
        -------
        ang_size: float
            the angular size in degrees
        """
        return diameter / self.Da(z) * 180./numpy.pi
    #end angular_size
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def physical_size(self, z, ang_size):
        """
        Computes the physical size corresponding to an angular size at z
        
        Parameters
        ----------
        z : float
            the redshift to compute sizes at
        ang_size : float
            the angular size, in degrees (float)

        Returns
        -------
        diameter : float
            physical size in Mpc
        """
        return ang_size*numpy.pi/180.*self.Da(z)
    #end physical_size
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def dump(self, z=None):
        print '-----------------------------'
        print "Cosmology: "
        print "   Omega_m = %.4g" % cosmo.omega_m_0
        print "   Omega_l = %.4g" % cosmo.omega_l_0
        print "   Omega_r = %.4g" % self._omega_r_0
        print "   H0      = %.4g km/s/Mpc" %self._H0
        print "   Dh      = %.4g Mpc" %self._hubble_distance
        print "   e_crit  = %.4g erg/cm^3" % self.e_crit(0)
        print "   CMB dens= %.4g erg/cm^3" % (pc.a_rad * cosmo.Tcmb_0**4)
        print "     (frac = %.4g)" %self._omega_gam_0
        print ''
        print "   radiation-matter equality at z = %.4g" %self._z_rm
        print "   horizon distance at r-m equality:",
        print "%.4g Mpc" % self.D_hor(self._z_rm)
        print "   horizon distance today: %.4g Mpc" %self.D_hor(0)
        print ''
        if z!=None:
            print "For z = %.2f:" % z
            print "   Hubble Parameter H(z)          %.2f km/s/Mpc" % self.H(z)
            print '   Lookback time                  %.2f Gyr' % self.lookback_time(z)
            print '   Age of the universe            %.2f Gyr' % self.age(z)
            print '   Scale Factor a                 %.2f'     % self.a(z)
            print '   Comoving L.O.S. Distance (w)   %.2f Mpc' % self.Dc(z)
            print '   Angular diameter distance      %.2f Mpc' % self.Da(z)
            print '   Luminosity distance            %.2f Mpc' % self.Dl(z)
            print '   Distance modulus               %.2f mag' % self.mu(z)

        print '-----------------------------'
    #end dump
    #---------------------------------------------------------------------------
#end class cosmology