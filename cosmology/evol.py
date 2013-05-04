"""
 evol.py
 cosmology: tools for computing quantities depending on redshift
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2013
"""
import numpy
from scipy import integrate

from cosmology import cosmo, parameters
import utils.samplinator as s
import utils.physical_constants as pc

class evol(s.with_sampleable_methods):
    """
    @brief used to automate cosmological computations of quantities that
    evolve with redshift (distances, times).
    Note that all distance are in Mpc and all times come are in Gyr
    Defaults are the Planck 2013 data
    """
    
    def __init__(self, **kwargs):
        """
        @brief the allowable keyword parameters are:
        'omega_m_0' : matter parameter today
        'omega_l_0' : vacuum parameter today
        'omega_b_0' :  baryon matter parameter today
        'omega_r_0' : radiation parameter today
        'omega_k_0' : curvature parameter
        'h' : Hubble parameter
        'n' : spectral index
        'sigma_8' : power spectrum normalization today
        'tau' : reionization optical depth, 
        'z_reion' : redshift of reionization,
        't_0' : age of universe in Gyr,
        """
        
        self.update_cosmo(**kwargs)
        
    #end __init__
    
    #---------------------------------------------------------------------------
    def set_extras(self):
        """
        @brief set the extra parameters
        """
        # add extras
        parameters.add_extras(cosmo) 
        
        if cosmo.omega_r_0 == 0:
            cosmo.z_rm = numpy.inf
        else:
            cosmo.z_rm = (cosmo.omega_m_0 / cosmo.omega_r_0) - 1            
        
        if cosmo.omega_m_0 == 0:
            cosmo.a_rm = numpy.inf
        else:
            cosmo.a_rm = cosmo.omega_r_0 / cosmo.omega_m_0
            
        # constants
        cosmo.H0 = 100.*cosmo.h
        cosmo.Th = pc.mega*pc.parsec/(cosmo.H0*pc.kilo*pc.meter) \
                            /(pc.giga*pc.year) # in Gyr
        cosmo.Dh = pc.c_light/(pc.kilo*pc.meter) / cosmo.H0
    #end set_extras
   
    #---------------------------------------------------------------------------
    def get_cosmo(self):
        """
        @brief return the cosmology parameters being used
        """
        return cosmo
    #end get_cosmo
    
    #---------------------------------------------------------------------------
    def update_cosmo(self, **kwargs):
        """
        @brief update the cosmological parameters
        """
        # update the cosmo dictionary
        for k, v in kwargs.iteritems(): cosmo[k] = v
        
        self.set_extras()
    #end update_cosmo
        
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def __E(self, z):
        """
        @brief the unitless Hubble expansion rate at redshift z, 
        modified to include non-constant w parameterized linearly 
        with z ( w = w0 + w1*z )
        """
        if z == numpy.inf: return numpy.inf
        return numpy.sqrt(cosmo.omega_m_0 * (1. + z)**3 \
                          + cosmo.omega_r_0 * (1. + z)**4 \
                          + cosmo.omega_k_0 * (1.+ z)**2 \
                          + cosmo.omega_l_0 * numpy.exp(3*cosmo.w1*z) \
                          * (1.+ z)**(3 * (1 + cosmo.w0 - cosmo.w1)))
    #end __E
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def H(self, z):
        """
        @brief the value of the Hubble constant at redshift z in km/s/Mpc
        """
        return 100. * cosmo.h * self.__E(z)
    #end H
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def w(self,z):
        """
        @brief equation of state: w(z) = w0 + w1*z
        """
        return cosmo.w0 + cosmo.w1 * z
    #end w
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def a(self, z):
        """
        @brief scale factor at redshift z
        """
        return 1. / (1. + z)
    #end a
    
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def lookback_time(self, z):
        """
        @brief lookback time, defined as the difference between the age 
        of the Universe now and the age at z
        """
        f = lambda z: 1/(1.+z)/self.__E(z)
        I = integrate.quad(f, 0, z)
        return cosmo.Th * I[0]
    #end Tl
    
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def age(self, z):
        """
        @brief the age of the universe at redshift z in Gyr.
        """
        f = lambda z: 1/(1.+z)/self.__E(z)
        I = integrate.quad(f, z, numpy.inf)
        return cosmo.Th * I[0]
    #end age
    
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def Dc(self, z):
        """
        @brief line of sight comoving distance in Mpc (eqn 15 from Hogg 1999).
        Remains constant with epoch if objects are in the Hubble flow
        """
        if z==0:
            return 0
        else:
            f = lambda z: 1.0/self.__E(z)
            I = integrate.quad(f, 0, z)
            return cosmo.Dh * I[0]
    #end Dc
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def Dp(self, z):
        """
        @brief proper or physical distance
        """
        return self.Dc(z)/(1.+z)
    #end Dp
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def Dm(self, z):
        """
        @brief transverse comoving distance in Mpc (eqn 16 from Hogg 1999).
        At same redshift but separated by angle dtheta;
        Dm * dtheta is transverse comoving distance
        """
        sOk = numpy.sqrt(numpy.abs(cosmo.omega_k_0))
        if cosmo.omega_k_0 < 0.0:
            return cosmo.Dh * numpy.sin(sOk * self.Dc(z)/cosmo.Dh ) / sOk
        elif self.Ok == 0.0:
            return self.Dc(z)
        else:
            return cosmo.Dh * numpy.sinh(sOk * self.Dc(z)/cosmo.Dh ) / sOk
    #end Dm
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def Da(self, z):
        """
        @brief angular diameter distance in Mpc (eqn 18 from Hogg 1999).
        Ratio of an objects physical transvserse size to its
        angular size in radians
        """
        return self.Dm(z) / (1.+ z)
    #end Da
    
    #---------------------------------------------------------------------------
    def Da12(self, z1, z2):
        """
        @brief angular diameter distance between objects at 2 redshifts in Mpc.
        Useful for gravitational lensing (eqn 19 of Hogg 1999)
        """
        # does not work for negative curvature
        assert(cosmo.omega_k_0) >= -1e4

        # z1 < z2
        if (z2 < z1):
            z1,z2 = z2,z1

        Dm1 = self.Dm(z1)
        Dm2 = self.Dm(z2)
        Ok  = cosmo.omega_k_0
        Dh  = cosmo.Dh

        return 1. / (1 + z2) * ( Dm2 * numpy.sqrt(1. + Ok * Dm1**2 / Dh**2)\
                                 - Dm1 * numpy.sqrt(1. + Ok * Dm2**2 / Dh**2) )
    #end Da12
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def Dl(self, z):
        """
        @brief luminosity distance in Mpc (eqn 21 of Hogg 1999)
        """
        return (1. + z) * self.Dm(z)
    #end Dl
    
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def D_hor(self, z):
        """
        @brief horizon distance at redshift z in Mpc (eqn)
        (physical dist that light can travel from z' = infty to z' = z)
        """
        f = lambda zp : 1 / self.__E(zp)
        I = integrate.quad(f, z, numpy.inf)
        return cosmo.Dh * I[0]
    #end D_hor

    #---------------------------------------------------------------------------
    @s.call_as_array
    def mu(self, z):
        """
        @brief the distance Modulus
        """
        return 5. * numpy.log10(self.Dl(z) * 1e6) - 5.
    #end mu

    #---------------------------------------------------------------------------
    @s.call_as_array
    def dVc(self, z):
        """
        @brief the differential comoving volume element dV_c/dz/dSolidAngle.
        Dimensions are volume per unit redshift per unit solid angle.
        Units are Mpc**3 Steradians^-1. From eqn 28 of Hogg 1999.
        """
        Dm = self.Dm(z)
        E = self.__E(z)
        return cosmo.Dh * Dm**2. / E
    #end dVc
    
    #----------------------------------------------------------------------------
    @s.call_as_array
    def Vc(self, z):
        """
        @brief the comoving volume out to redshift z in Mpc^3. 
        From eqn 29 of Hogg 1999.
        """
        Ok = cosmo.omega_k_0
        Dh = cosmo.Dh
        sOk = numpy.sqrt(numpy.abs(Ok))
        Dm = self.Dm(z)
        if Ok < 0.0:
            norm = 4*numpy.pi*Dh**3. / (2.*Ok)
            return norm*(Dm/Dh*numpy.sqrt(1+Ok*(Dm/Dh)**2) - \
                    numpy.arcsin(sOk * Dm/Dh ) / sOk )
        elif Ok == 0.0:
            return 4*numpy.pi*Dh**3 / 3.
        else:
            norm = 4*pi*Dh**3/(2.*Ok)
            return norm*(Dm/Dh*numpy.sqrt(1+Ok*(Dm/Dh)**2) - \
                    numpy.arcsinh(sOk * Dm/Dh ) / sOk)
    #end Vc

    #---------------------------------------------------------------------------
    @s.call_as_array
    def rho_crit(self,z=0):
        """
        @brief calculate the critical (mass) density at z
        returns value in g/cm^3
        """
        return 3.*(self.H(z)*pc.kilo*pc.meter/pc.mega/pc.parsec)**2 / \
                    (8 * numpy.pi * pc.G)
               
    #end rho_crit
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def rho_mean(self, z=0):
        """
        @brief calculate the mean density of the universe, at redshift z.
        returns values in g/cm^3
        """
        return self.omega_m_z(z) * self.rho_crit(z) 
    #end rho_mean
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def e_crit(self,z=0):
        """
        @brief calculate the critical (energy) density at z

        returns value in erg/cm^3
        """
        return 3*(pc.c_light)**2 / (8 * numpy.pi * pc.G) *\
               (self.H(z) / 1e3 / pc.parsec)**2
    #end e_crit
               
    #---------------------------------------------------------------------------
    @s.call_as_array
    def lens_kernel(self,z, z_source):
        """
        @brief returns the value of the lens kernel at a redshift z, given a
        source at redshift z_source

        units are strange: cm Mpc^2 / g
        """
        Ds = self.Dc(z_source)
        D = self.Dc(z)

        return 4*numpy.pi*pc.G / (pc.c_light)**2 * self.Dh / self.__E(z) * \
               D*(Ds-D)/Ds / (1.+z)**2
    #end lens_kernel
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def omega_m_z(self, z):
        """
        @brief matter density omega_m as a function of redshift

        From Lahav et al. 1991 equations 11b-c. This is equivalent to 
        equation 10 of Eisenstein & Hu 1999.
        """
        return cosmo.omega_m_0 * (1.+z)**3. / self.__E(z)**2.
    #end omega_m_z
    
    #---------------------------------------------------------------------------
    @s.call_as_array
    def omega_l_z(self, z):
        """
        @brief vacuum energy density omega_l as a function of redshift
        """
        return cosmo.omega_l_0 / self.__E(z)**2
    #end omega_l_z
        
    #---------------------------------------------------------------------------
    @s.call_item_by_item
    def dump(self, z=None):
        print '-----------------------------'
        print "Cosmology: "
        print "   Omega_m = %.4g" % cosmo.omega_m_0
        print "   Omega_L = %.4g" % cosmo.omega_l_0
        print "   Omega_r = %.4g" % cosmo.omega_r_0
        print "   H0      = %.4g km/s/Mpc" %cosmo.H0
        print "   Dh      = %.4g Mpc" %cosmo.Dh
        print "   e_crit  = %.4g erg/cm^3" % self.e_crit(0)
        print "   CMB dens= %.4g erg/cm^3" % (pc.a_rad * pc.T_cmb**4)
        print "     (frac = %.4g)" %(pc.a_rad*pc.T_cmb**4/self.e_crit(0))
        print ''
        print "   radiation-matter equality at z = %.4g" %cosmo.z_rm
        print "   horizon distance at r-m equality:",
        print "%.4g Mpc" % self.D_hor(cosmo.z_rm)
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
#end class evol

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    import pylab

    C = evol()
    C.dump(1.0)

    a = numpy.linspace(1./1100,1.0,1000)
    z = (1./a-1)
    G = C.linear_growth_factor(z)

    pylab.plot(a,G)
    pylab.ylabel('G')
    pylab.xlabel('a')
    pylab.title("Linear Growth Factor")
    pylab.show()
    
    