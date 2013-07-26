"""
 linear_growth.py
 cosmology
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/02/2013
"""
import numpy
from scipy import integrate

from cosmology import cosmo, core, tf_eh, parameters
from utils import pytools
import utils.physical_constants as pc

class linear_power(core.cosmology):
    """
    A class used to compute quantities from linear perturbation theory and 
    linear power spectrum analysis. 
    
    Notes
    -----
    Default cosmology is the Planck 2013 parameter set.
    """
    def __init__(self, tf='EH_full', cosmo_params=None):
        """
        The available transfer functions are: 
            'BBKS' : approximation from Bardeen et al. 1986
            'EH_full' : full CDM + baryon with wiggles TF from EH 1998
            'EH_no_wiggles' : full CDM + baryon w/o wiggles from EH 1998
            'EH_no_baryons' :  CDM TF from EH 1998
        """
        # set up the cosmo dict
        if cosmo_params is None and cosmo.is_empty():  
            print("Warning: No default cosmology has been specified, "
                          "using %s parameters." %parameters.default()['name'])
            cosmo.unify(parameters.default())
        elif cosmo_params is not None:
            self.set_current(cosmo_params)
            
        # verify and set params
        self._verify_params()
        self._set_extras()
        
        # setup the TF
        if tf == 'BBKS':
            self.tf = self.T_BBKS
        elif tf == 'EH_full':
            tf = tf_eh.tf_eh(cosmo)
            self.tf = tf.full
        elif tf == 'EH_no_wiggles':
            tf = tf_eh.tf_eh(cosmo)
            self.tf = tf.no_wiggles
        elif tf == 'EH_no_baryons':
            tf = tf_eh.tf_eh(cosmo)
            self.tf = tf.no_baryons
        else:
            raise KeyError("keyword 'transfer = %s' not recognized" %transfer)
        
        self._P0 = None
        
    #end __init__
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def growth_rate(self, z):
        """
        The growth rate, which is the logarithmic derivative of the growth 
        factor with respect to scale factor. Fitting formula taken from 
        E. V. Linder, Phys. Rev. D 72, 2005
        """
        gamma = 0.55 + 0.05*(1. + self.w(1.))
        return self.omega_m_z(z)**gamma
    #end growth_rate
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def growth_factor(self, z, integrate=False):
        """
        The linear growth factor. If integrate = False, use approximation
        from Carol, Press, & Turner (1992), else integrate the ODE.
        Normalized to 1 at z = 0.
        """
        if not integrate:
            om_m = self.omega_m_z(z)
            om_l = self.omega_l_z(z)
            
            norm = 2.5*cosmo.omega_m_0/(cosmo.omega_m_0**(4./7.) - \
                    cosmo.omega_l_0 + (1.+cosmo.omega_m_0/2.) * \
                    (1.+cosmo.omega_l_0/70.))
            return 2.5*om_m / (om_m**(4./7.) - om_l + (1. + om_m/2.) * \
                (1.+om_l/70.)) / (norm*(1+z))
        else:
            return self._growth_factor_integrate(z)
    #end growth_factor
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def _growth_factor_integrate(self, z):
        """
        Solves the ODE for the linear growth of matter overdensity:
         d_k'' + 2Hd_k' - 3/2 Omega_m H^2 d_k = 0

        d_k(z) = D(z) d_k(z=0)

        uses boundary conditions D(z=0) = 1
                                 D(z=1100) = a(z=1100)
        
        solves equation 7.72 in Dodelson, and return value is proportional to 
        Eisenstein & Hu 1999, equation 10
        """


        # y = [D(a),D'(a)]
        # y' = J * y
        # J is the Jacobian (gradient) matrix associated with the
        #   differential equation above

        if cosmo.w1 != 0 or cosmo.w0 != -1:
            raise ValueError, "growth_factor defined only for w=-1"

        numerator = lambda a: -(3*cosmo.omega_m_0*a**-3 \
                                + 4*cosmo.omega_r_0*a**-4 \
                                + 2*cosmo.omega_k_0*a**-2)
        denominator = lambda a: 2*(cosmo.omega_m_0*a**-3 \
                                   + cosmo.omega_r_0*a**-4 \
                                   + cosmo.omega_k_0*a**-2 \
                                   + cosmo.omega_l_0)

        jacobian = lambda Y,a: [[0,1],
                                [3.*cosmo.omega_m_0/a**5/denominator(a),
                                 -(3. + numerator(a)/denominator(a))/a ,]]

        Yprime = lambda Y,a: numpy.dot(jacobian(Y,a),Y)

        a0 = 1./1100
        Y0 = [a0,1]
        a = 1./(1.+z)

        #need to sort in order of increasing a
        # will need to unsort at the end
        sort = numpy.argsort(a)
        unsort = numpy.argsort( numpy.arange(len(a))[sort] )

        a = a[sort]

        #need the first value to be a0
        start_index = 0
        if a[0] < a0:
            raise ValueError, "G(z) valid only for z < %.2g" % (1+1./a0)
        elif a[0] > a0:
            start_index = 1
            a = numpy.concatenate([[a0],a])

        #need to evaluate G(a=1) for normalization purposes
        end_index = len(a)
        if a[-1] > 1:
            raise ValueError, "G(z) not valid for z < 0"
        elif a[-1] == 1:
            pass
        else:
            a = numpy.concatenate([a,[1.0]])


        Y = integrate.odeint(Yprime,Y0,a,Dfun=jacobian)
        D =  Y[start_index:end_index,0][unsort]
        D0 = Y[-1,0]

        if numpy.shape(z) == ():
            D = D[0]

        return D/D0
    #end growth_factor

    #---------------------------------------------------------------------------
    def T_BBKS(self, k):
        """
        The linear transfer function due to Bardeen et al 1986
        equation 7.70 from Dodelson's Modern Cosmology. q = k / (omega_m*h**2)
        
        Parameters
        ----------
        k : float or numpy.ndarray
            the wavenumbers in units of 1 / Mpc
        """
        q = k / (cosmo.omega_m_0*cosmo.h**2)
        return numpy.log(1 + 2.34*q)/(2.34*q) * \
               (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
    #end T_BBKS   
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def P_k(self, k, z):
        """
        Uses the specified transfer function to define an approximation 
        to the linear power spectrum, appropriately normalized via sigma8, 
        at redshift z. The primordial spectrum is assumed to be proportional 
        to k^n
        
        Uses equation 25 of Eisenstein & Hu (EH) 1999 to compute P(k)

        Parameters
        ----------
        k : numpy.ndarray or float
            the wavenumber in units of 1 / Mpc
        z : float
            the redshift to compute the spectrum at
        
        Returns
        -------
        P_k : numpy.ndarray or float
            linear matter power spectrum in units of Mpc**3
        """
        # compute P0 if it is not yet computed
        if self._P0 == None:
            self._compute_P0()
          
        # set the spectral index  
        if 'n' in cosmo.keys():
            n = cosmo.n
        else:
            n = 1.0
        
        Tk = self.tf(k)
        fg = self.growth_factor(z)
        P_k = self._P0 * (k)**n * (Tk*fg)**2 
        
        return P_k
    #end Pk
    
    #---------------------------------------------------------------------------
    def w_tophat(self, k, r):
        """
        The k-space Fourier transform of a spherical tophat.
        """
        return 3.0 * (numpy.sin(k*r) - k*r*numpy.cos(k*r)) / (k*r)**3
    #end w_tophat
    
    #---------------------------------------------------------------------------
    def _compute_P0(self):
        """
        Compute the power spectrum normalization based on the
        value of sigma_8, which is sigma_r at r = 8/h Mpc
        """
        self._P0 = 1.0
        
        I_dk = lambda k: k**2 * self.P_k(k, 0.) * \
                        self.w_tophat(k, 8./cosmo.h)**2
        I = integrate.quad( I_dk, 0, numpy.inf )
        self._P0 = (cosmo.sigma_8**2) * (2 * numpy.pi**2) / I[0]
    #end _compute_P0
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def _sigma_r_integral(self, r, k, Pk):
        """
        Internal module to compute sigma_r integral
        """ 
        dx = numpy.log(k[1]) - numpy.log(k[0])
        integrand = k**2 * Pk * self.w_tophat(k, r)**2
        return integrate.simps(integrand*k, dx=dx )
    #end _sigma_r_integral
    
    #---------------------------------------------------------------------------
    def sigma_r(self, r, z):
        """
        This returns the average mass fluctuation within a sphere of 
        radius r, based on the power spectrum defined in self.P_BBKS
        """
        # compute power spectrum at z = 0 to use in integral
        k = numpy.logspace(-7, 5, 1e5)
        Pk_0 = self.P_k(k, 0.)
        
        I = self._sigma_r_integral(r, k, Pk_0)
        fg = self.growth_factor(z)

        return fg * numpy.sqrt( I / (2*numpy.pi**2) )
    #end compute_sigma
    
    #---------------------------------------------------------------------------    
    @pytools.call_item_by_item
    def _xi_r_integral(self, r, k, Pk):
        """
        Internal function to compute the correlation function integral
        """
        # integrand is in log space: x = log(k) is evenly spaced
        #                           dx = dk/k
        # integrate using simpson's rule in log space: 
        #     Bessel func oscillations are better behaved
        dk = numpy.log(k[1]) - numpy.log(k[0])
        integrand = k**2 * numpy.sin(k*r) * Pk 

        return integrate.simps(integrand, dx=dk)
    #end _xi_r_integral
    
    #---------------------------------------------------------------------------
    def xi_r(self, r, z):
        r"""
        Uses the specified transfer function to define an approximation 
        to the matter correlation function, at redshift z. Given by
        
        \xi(r, z) = \int_0^\infty \frac{k sin(kr)}{2 \pi^2 r}~P(k, z)~
                  = \xi(r, 0) \left(\frac{D_1(z)}{D_1(0)}\right)^2
        
        Units of r are assumed to be Mpc. See P_k documentaton for available 
        transfer functions. 
        """
        # compute power spectrum at z = 0 to use in integral
        k = numpy.logspace(-7, 5, 1e5)
        Pk_0 = self.P_k(k, 0.)
        
        I = self._xi_r_integral(r, k, Pk_0)
        fg = self.growth_factor(z)
        
        return fg**2 * I * 0.5/ numpy.pi**2. / r 
    #end xi_r
    
    #---------------------------------------------------------------------------
    @pytools.call_item_by_item
    def xi_avg_r(self, r, z, bins=50):
        r"""
        Uses the specified transfer function to define an approximation 
        to the matter correlation function, at redshift z, averaged over
        a sphere of radius r
        
        Units of r are assumed to be Mpc. See P_k documentaton for available 
        transfer functions. 
        """
        def integrand(x):
            return x**2*self.xi_r(x, z)
            
        rs = numpy.linspace(1e-5, r, bins)
        I = integrate.simps(integrand(rs), x=rs)        

        return 3 * I / r**3
    #end xi_avg_r
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def mass_to_radius(self, mass, z):
        """
        Convenience function to convert a mass in M_sun to the 
        radius in Mpc of the corresponding sphere
        """
        # the radius corresponding to the input mass
        rho_mean = self.rho_mean(z) / (pc.M_sun / (pc.mega*pc.parsec)**3)
    
        # compute the radii in Mpc corresponding to the masses
        return (3.*mass/(4.*numpy.pi*rho_mean))**(1./3.)
    #end mass_to_radius
    
    #---------------------------------------------------------------------------
    @pytools.call_as_array
    def radius_to_mass(self, R, z):
        """
        Convenience function to convert a radius in Mpc to the mass in 
        M_sun within the sphere
        """
        # the radius corresponding to the input mass
        rho_mean = self.rho_mean(z) / (pc.M_sun / (pc.mega*pc.parsec)**3)
    
        # compute the radii in Mpc corresponding to the masses
        return rho_mean*(4./3.*numpy.pi*R**3)
    #end radius_to_mass
        
        
        