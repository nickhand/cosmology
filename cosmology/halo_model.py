"""
 halo_model.py
 cosmology: implements a halo model class
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/03/2013
"""
import numpy
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate, integrate

from cosmology import cosmo, core, linear_growth, parameters
import utils.physical_constants as pc

class halo_model(core.cosmology):
    """
    A class that implements various halo model relation quantities.
    
    Notes
    -----
    Default cosmology is the Planck 2013 parameter set.
    """
    
    def __init__(self, tf='EH_full', fitting_func=1, cosmo_params=None):
        """
        Parameters
        ----------
        Mass functions used are based on fitting_func:
            1 = Tinker 2008
            2 = Jenkins 2001
            3 = Sheth-Tormen 1999
            4 = Warren 2005
            5 = Press-Schechter
        """
        # set up the cosmo dict
        if cosmo_params is None and cosmo.is_empty(): 
            print("Warning: No default cosmology has been specified, "
                          "using Planck 2013 parameters.")
            cosmo.unify(parameters.Planck13())
        elif cosmo_params is not None:
            self.set_current(cosmo_params)
            
        # verify and set params
        self._verify_params()
        self._set_extras()
        
        self.growth = linear_growth.linear_growth(tf=tf)
        self.fitting_func = fitting_func
    
    def halo_mass_function(self, logMassMin, logMassMax, z):
        r"""
        Compute the halo mass number abundance at z in 1/Mpc^3 given by
    
        dn(M, z) = f(\sigma) * \frac{\bar{\rho}_m}{M} \ 
                                * \frac{d ln(\sigma^{-1})}{dM} * dM
    
        where f(\sigma) is the multiplicity function, computed in 
        multiplicity_function() below. 
        """
        # number of bins
        nBins = 1000
    
        # go factor of sqrt(10) below/above
        new_min = logMassMin
        new_max = logMassMax
    
        # generate masses array, in M_sun
        masses = numpy.logspace(new_min, new_max, nBins) 
    
        # compute the critical density and mean density in units of M_sun / Mpc**3
        rho_mean = self.rho_mean(0.) / (pc.M_sun / (pc.mega*pc.parsec)**3)
    
        # compute the radii in Mpc corresponding to the masses
        radii = (3.*masses/(4.*numpy.pi*rho_mean))**(1./3.)
    
        # compute the sigmas corresponding to these radii
        sigmas = self.growth.sigma_r(radii, z) 
        
        # keep track of cumulative number density
        dn_cum = 0.0

        # create the output arrays
        # 1) (dn/dM)*dM (differential number density of halos, per Mpc^3 
        dn_dM = numpy.zeros(nBins, dtype='float64')
    
        # Loop over masses, going BACKWARD, and calculate dn/dm as well as the 
        # cumulative mass function.
        for j in xrange(nBins-1):
        
            i = (nBins - 2) - j

            # calc dsigmadm - has units of M_sun^-1)
            dsigmadm = (sigmas[i+1]-sigmas[i]) / (masses[i+1] - masses[i])

            # calculate dn(M,z)/dM 
            # this has units of 1/Mpc^(3)/Msun
            fsigma = self.multiplicity_function(sigmas[i], z)
            dn = -1.0 / sigmas[i] * dsigmadm * rho_mean * fsigma / masses[i] 

            # keep track of cumulative number density
            if dn > 1.0e-20:
                dn_cum += dn

            # Store the results
            dn_dM[i] = dn
        
        return masses, dn_dM
    #end halo_mass_function
    
    #---------------------------------------------------------------------------
    def multiplicity_function(self, sigma, z):
        """
        Compute the multiplicity function: this is where the various 
        fitting functions/analytic theories are different.  The various places
        where I found these fitting function sare listed below.
        """
        
        # the collapse fraction at z=0
        delta_crit = 1.686
        
        # dimensionaless mass parameter used for Press-Schechter
        nu = delta_crit / sigma
    
        if self.fitting_func == 1:
            # Tinker et al. 2008, eqn 3 for delta = 200
            
            A = 0.186*(1+z)**(-0.14)
            a = 1.47*(1+z)**(-0.06)
            alpha = 10**(-(0.75/numpy.log10(200./75))**1.2)
            b = 2.57*(1+z)**(-alpha)
            c = 1.19
            fsigma = A * ( (sigma/b)**(-a) + 1 )*numpy.exp(-c/sigma**2)
            
        elif self.fitting_func == 2:
            # Jenkins et al. 2001, equation 9
            fsigma = 0.315 * numpy.exp( -1. * abs( numpy.log(1./sigma) + \
                    0.61)**(3.8) )
        
        elif self.fitting_func == 3:
            # Sheth-Tormen 1999, eqtn 10, using expression from 
            # Jenkins et al. 2001, eqn. 7
            A=0.3222
            a=0.707
            p=0.3
            fsigma = A*numpy.sqrt(2.*a/numpy.pi)*(1.+ 1.0/(nu*nu*a)**p ) * \
                    nu * numpy.exp(-0.5*a*nu**2)
        
        elif self.fitting_func == 4:
            # Warren et al. 2005, eqn. 5 
            A=0.7234
            a=1.625 
            b=0.2538 
            c=1.1982
            fsigma = A*( sigma**(-a) + b)*numpy.exp(-c / sigma**2)
    
        elif self.fitting_func == 5:
            # Press-Schechter (This form from Jenkins et al. 2001, MNRAS 321, 372-384, eqtn. 5)
            fsigma = numpy.sqrt(2./numpy.pi) * nu * numpy.exp(-0.5*nu**2)
        
        else:
            raise ValueError("Don't understand this. " + \
                        "Fitting function requested is %d" %self.fitting_func)
    
        return fsigma
    #end multiplicity_function
    
    #---------------------------------------------------------------------------
    def cluster_count(self, area, M_cut, zlim, Nz = 10):
        """
        Compute cluster count, given a mass function and cosmology, in 
        a given survey area and halo mass range
        """
        area *= (numpy.pi/180.)**2 # put the area into steradians from deg2
        logMassCut = numpy.log10(M_cut)
        z = numpy.linspace(zlim[0], zlim[1], Nz)
        
        # now uses simpson integration to do the double integral
        def integrand(z):
            out = []
            for x in z:
                masses, dn_dM = self.halo_mass_function(logMassCut, 16, x) 
                dx = numpy.log(masses[1]) - numpy.log(masses[0])
                out.append(self.dVc(x)*integrate.simps(dn_dM*masses, dx=dx))
            return numpy.array(out)

        y = integrand(z)
        N = integrate.simps(y, dx=z[1]-z[0])

        return N*area
    #end cluster_count
    
    #---------------------------------------------------------------------------     
    def bias_of_M(self, mass, z):
        """
        The bias as a function of mass and redshift as computed from simulations
        in Tinker et al (2010) equation 6
        """

        # the overdensity factor
        y = numpy.log10(200.0) 

        # the parameters
        A = 1.0 + 0.24*y*numpy.exp(-(4./y)**4.)
        a = 0.44*y - 0.88

        B = 0.183
        b = 1.5

        C = 0.019 + 0.107*y + 0.19*numpy.exp(-(4./y)**4.)
        c = 2.4

        # assume a constant \delta_crit with redshift
        deltaCritofz = 1.686 
        
        # compute the radii in Mpc corresponding to the masses
        R = self.growth.mass_to_radius(mass)

        # the variance of the mass perturbations on this scale R
        sigmaofz = self.growth.sigma_r(R, z) 

        # the variable used in the fit
        nu = (deltaCritofz / sigmaofz)

        # compute the bias 
        b_200 = 1.0 - A * (nu**a)/(nu**a + deltaCritofz**a) + B*nu**b + C*nu**c

        return b_200
    #end bias_of_M
    
    #---------------------------------------------------------------------------   
    def nonlinear_scales(self, z):
        """
        The nonlinear mass and radius defined as where 
        sigma(M, z) = delta_crit
        """
        # compute the overdensity variance at several radii
        Rs = numpy.logspace(-5, 1, 1000)
        sigmas = self.growth.sigma_r(Rs, z) 
    
        # find where sigma = delta_crit
        delta_crit = 1.686
        x = abs(sigmas - delta_crit)
        inds = numpy.where(x == numpy.amin(x))[0]
    
        R = Rs[inds][0]
        
        # return the corresponding mass
        return self.growth.radius_to_mass(R, z), R
    #end nonlinear_scales
    
    #--------------------------------------------------------------------------- 
    def concentration(self, mass, z):
        """
        The concentration, taken from Yoshida et al (2001) equation 7 
        (which is from Bullock et al 2001)
        """
    
        M_nl, R = self.nonlinear_scales(z)
        return 12.*(mass / M_nl)**(-0.13)
    #end concentration
    
    #---------------------------------------------------------------------------
    def r_s(self, mass, z):
        """
        The scale radius of a NFW profile in Mpc, as computed
        from Yoshida et al (2001)
        """
    
        # nonlinear collapse overdensity for flat universe
        delta_nl = 18.*numpy.pi**2 
    
        # compute the critical density in units of M_sun / Mpc**3
        rho_crit = self.growth.rho_crit(z) / (pc.M_sun / (pc.mega*pc.parsec)**3)

        # get the concentration
        c = self.concentration(mass, z)
    
        # compute r_s in Mpc
        return 1./c * (3.*mass / (4.*numpy.pi*rho_crit*delta_nl))**(1./3)
    #end r_s
    
    #---------------------------------------------------------------------------    
    def rho_s(mass, z, **cosmo):
        """
        The scale density of a NFW profile in M_sun / Mpc^3, as computed
        from Yoshida et al (2001)
        """

        # nonlinear collapse overdensity for flat universe
        delta_nl = 18.*numpy.pi**2 
    
        # compute the critical density in units of M_sun / Mpc**3
        rho_crit = self.growth.rho_crit(z) / (pc.M_sun / (pc.mega*pc.parsec)**3)

        # get the concentration
        c = self.concentration(mass, z)

        # compute the scale density
        return (rho_crit*delta_nl*c**3) / ( 3.*(numpy.log(1.+c) - c/(1.+c)) )
    #end rho_s
    
    #---------------------------------------------------------------------------
