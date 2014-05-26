import numpy as np
import scipy.integrate as integ
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline

from ..parameters import Cosmology, default_params
from . import convert_halo_mass, convert_halo_radius, concentration, nfw_rs, nfw_rhos, virial_overdensity
from ..utils import constants as const, tools, functionator as fn
from ..evolution import critical_density, Da

#-------------------------------------------------------------------------------
class HaloProfile(object):
    """
    A class to compute several different halo (spherically-symmetric) profiles 
    as a function of both 3D and projected 2D radius.
    """
    def __init__(self, mass, 
                       z, 
                       mass_definition, 
                       mass_delta, 
                       X=0.76,
                       pressure_fit='Battaglia',
                       cosmo={'default': default_params, 'flat': True}):

        # store the cosmological parameters
        if isinstance(cosmo, Cosmology):
            self.cosmo = cosmo
        else:
            self.cosmo = Cosmology(**cosmo)
        
        self.pressure_fit    = pressure_fit
        self.z               = z
        self.X               = X
        
        # update the mass params
        self.update_mass(mass, mass_definition, mass_delta)                   
                           
    #end __init__
    #---------------------------------------------------------------------------
    def update_mass(self, mass, definition, delta):
        """
        Optimally update the mass parameters
        """
        # first check the input mass parameters
        choices = ['critical', 'mean', 'virial']
        if definition not in choices:
            raise ValueError("Mass definition '%s' must be one of: %s" %(definition, choices))
        
        if definition != 'virial' and delta != 200.:
            raise ValueError("Mass must be defined w.r.t 200x critical/mean density.")
            
        self.mass            = mass 
        self.mass_definition = definition
        self.mass_delta      = delta
        
        # store the new virial mass
        if definition != 'virial':
            self.virial_mass = convert_halo_mass(self.mass, self.z, self.mass_definition, 
                                                'virial', None, params=self.cosmo)[0]
        else:
            self.virial_mass = self.mass
        
        # delete dependences
        del self.concentration, self.radius_200c, self.scale_radius
    
    #end update_mass
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, val):
        self._z = val
        
        if hasattr(self, 'mass'):
            self.update(self.mass, self.mass_definition, self.mass_delta)
        
        del self.delta_c, self.Da
    #---------------------------------------------------------------------------
    @property
    def concentration(self):
        """
        The NFW concentration, `r_vir / r_s`
        """
        try:
            return self._concentration
        except AttributeError:
            self._concentration = concentration(self.virial_mass, self.z, 'virial')[0]
            return self._concentration
    
    @concentration.deleter
    def concentration(self):
        try:
            del self._concentration
        except AttributeError:
            pass
            
        # delete everything
        del self.gamma, self._B, self._eta_0, self.rho_gas_0
    #---------------------------------------------------------------------------
    @property
    def delta_c(self):
        """
        The virial overdensity at this redshift, as given by Bryan and Norman 1999.
        """
        try:
            return self._delta_c
        except AttributeError:
            self._delta_c = virial_overdensity(self.z, params=self.cosmo)[0]
            return self._delta_c 

    @delta_c.deleter
    def delta_c(self):
        try:
            del self._delta_c
        except AttributeError:
            pass    
         
    #---------------------------------------------------------------------------
    @property
    def radius_200c(self):
        """
        The radius where the average density is 200 times the critical density
        of the universe at that redshift [units :math: `Mpc h^{-1}`]
        """
        try: 
            return self._radius_200c
        except AttributeError:
            
            self._radius_200c = convert_halo_radius(self.virial_radius, self.z, 
                                                  'virial', 'critical', 200.,
                                                   params=self.cosmo)[0]
            return self._radius_200c
    
    @radius_200c.deleter
    def radius_200c(self):
        try:
            del self._radius_200c
        except AttributeError:
            pass

        del self.mass_200c, self.P0_Battaglia, self.Battaglia_params
    #---------------------------------------------------------------------------
    @property
    def mass_200c(self):
        """
        The mass contained within `self.radius_200c` 
        [units :math: `M_\odot h^{-1}`]
        """
        try: 
            return self._mass_200c
        except AttributeError:
            
            self._mass_200c = convert_halo_mass(self.virial_mass, self.z, 
                                                'virial', 'critical', 200.,
                                                params=self.cosmo)[0]
            return self._mass_200c

    @mass_200c.deleter
    def mass_200c(self):
        try:
            del self._mass_200c
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def virial_radius(self):
        """
        The virial radius `r_vir` in :math: `Mpc h^{-1}`, given by the 
        concentration times the scale radius.
        """
        return self.concentration * self.scale_radius
    #---------------------------------------------------------------------------
    @property
    def virial_theta(self):
        """
        The virial angular radius in `arcmin`.
        """
        return (self.virial_radius / self.Da) / const.arcminute
    #---------------------------------------------------------------------------
    @property
    def Da(self):
        """
        The angular diameter distance to `self.z` [units: Mpc]
        """
        try:
            return self._Da
        except AttributeError:
            self._Da = Da(self.z, params=self.cosmo)
            return self._Da
    
    @Da.deleter
    def Da(self):
        try:
            del self._Da
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def scale_radius(self):
        """
        The NFW scale radius `r_s` in :math: `Mpc h^{-1}`
        """
        try:
            return self._scale_radius
        except AttributeError:
            self._scale_radius = nfw_rs(self.virial_mass, self.z, 'virial', None, params=self.cosmo)[0]
            return self._scale_radius

    @scale_radius.deleter
    def scale_radius(self):
        try:
            del self._scale_radius
        except AttributeError:
            pass

        del self.scale_density
    #---------------------------------------------------------------------------
    @property
    def scale_density(self):
        """
        The NFW scale density `rho_s` in :math: `h^2 M_\odot / Mpc^3`
        """
        try: 
            return self._scale_density
        except AttributeError:
            self._scale_density = nfw_rhos(self.virial_mass, self.z, 'virial', None, params=self.cosmo)[0]
            return self._scale_density

    @scale_density.deleter
    def scale_density(self):
        try:
            del self._scale_density
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def gamma(self):
        """
        The polytropic index, such that :math: `P_{gas} \propto \rho_{gas}^\gamma`.
        
        Notes
        -----
        Given by Eq. A5 in Komatsu and Seljak 2002.
        """
        try:
            return self._gamma
        except AttributeError:
            
            c = self.concentration
            s_star = -(1. + 3.*c) / (1. + c)
            m_star = np.log(1. + c) - c / (1. + c)
            dms_dx = (-3.*c + 2.*np.log(1. + c)) / (1. + 3.*c)**2
        
            self._gamma = 1. - 1./s_star + (c / m_star) * dms_dx
            return self._gamma
    
    @gamma.deleter
    def gamma(self):
        try:
            del self._gamma
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def _eta_0(self):
        """
        The central mass-temperature normalization :math: `\eta(0)`.
        
        Notes
        -----
        Given by Eq. A2 in Komatsu and Seljak 2002. 
        """
        try:
            return self.__eta_0
        except AttributeError:
            
            m = lambda x: np.log(1+x) - x/(1+x)
            c = self.concentration
            gam = self.gamma
            s_star = -(1. + 3.*c) / (1. + c)
            integral = integ.quad(lambda u: m(u)/u**2, 0, c)[0]
            
            self.__eta_0 = (1./gam) * ( (-3./s_star) + 3*(gam-1.)*c/m(c) * integral)
            return self.__eta_0
    
    @_eta_0.deleter
    def _eta_0(self):
        try:
            del self.__eta_0
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def _B(self):
        """
        Parameter determining dimensionless gas density
        """
        try:
            return self.__B
        except AttributeError:
            
            c = self.concentration
            gam = self.gamma
            
            self.__B = 3./self._eta_0 * (gam - 1.)/gam / (np.log(1. + c)/c - 1./(1. + c))
            return self.__B
    
    @_B.deleter
    def _B(self):
        try:
            del self.__B
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def rho_gas_0(self):
        """
        The central gas density in :math: `h^2 M_\odot / Mpc^3`. This is computed
        from the normalization 
        
        ..math: \rho_{gas}(c) = \rho_{gas}(0) y_{gas}(c) = \Omega_b / \Omega_m \rho_{dm}(c)
        
        Notes
        -----
        Given by Eq. 20/21 in Komatsu and Seljak 2002.
        """
        try:
            return self._rho_gas_0
        except AttributeError:
            
            fb = self.cosmo.omegab / self.cosmo.omegam
            rho_dm = self.dark_matter_density(self.virial_radius)
            y_gas = self.dimensionless_gas_density(self.virial_radius)
            
            self._rho_gas_0 = fb * rho_dm / y_gas
            return self._rho_gas_0

    @rho_gas_0.deleter
    def rho_gas_0(self):
        try:
            del self._rho_gas_0
        except AttributeError:
            pass
        del self.T_gas_0, self.P0_KS
    #---------------------------------------------------------------------------
    @property
    def T_gas_0(self):
        """
        The central temperature [units: keV]
        
        Notes
        -----
        Given by Eq. 19 in Komatsu and Seljak 2002.
        """
        try:
            return self._T_gas_0
        except AttributeError:
            
            # this is in ergs
            norm = self._eta_0 * 4./(3 + 5.*self.X) 
            T0 =  norm * const.G*const.m_p*(self.virial_mass*const.M_sun) / (3.*self.virial_radius*const.Mpc)
            
            # this is in keV
            self._T_gas_0 = T0 / (const.kilo * const.eV)
            return self._T_gas_0
    
    @T_gas_0.deleter
    def T_gas_0(self):
        try:
            del self._T_gas_0
        except AttributeError:
            pass 
    #---------------------------------------------------------------------------
    @property
    def P0_KS(self):
        """
        The central pressure [units :math: h^2 eV / cm^3] for the Komatsu
        and Seljak pressure profile.
        
        Notes
        -----
        Given in Eq. 8 in Komatsu and Seljak 2002.
        """
        try:
            return self._P0_KS
        except AttributeError:
            
            rho_gas_0 = self.rho_gas_0 * (const.M_sun/const.Mpc**3)
            self._P0_KS = 0.25*(3. + 5.*self.X)*(rho_gas_0/const.m_p)*self.T_gas_0*const.kilo
            return self._P0_KS

    @P0_KS.deleter
    def P0_KS(self):
        try:
            del self._P0_KS
        except AttributeError:
            pass       
    #---------------------------------------------------------------------------
    @property
    def P0_Battaglia(self):
        """
        The central pressure P_200c [units :math: h^2 eV / cm^3] for the 
        Battaglia et al. 2012 simulation-derived pressure profile
        """
        try:
            return self._P0_Battaglia
        except AttributeError:
            
            rho_crit = critical_density(self.z, params=self.cosmo) * (const.M_sun/const.Mpc**3)
            fb = self.cosmo.omegab / self.cosmo.omegam
            M = self.mass_200c*const.M_sun
            R = self.radius_200c*const.Mpc
            self._P0_Battaglia = 200*const.G*M*rho_crit*fb / (2.*R) / const.eV     
            
            return self._P0_Battaglia
    
    @P0_Battaglia.deleter
    def P0_Battaglia(self):
        try:
            del self._P0_Battaglia
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property 
    def Battaglia_params(self):
        """
        Return P0, xc, beta for the Battaglia et al. simulation-derived
        pressure profile fit at this mass and redshift
        """
        try:
            return self._P0, self._xc, self._beta
        except AttributeError:
            
            # define the mass/redshift dependence model
            param_fit = lambda A, B, C: A * (self.mass_200c/self.cosmo.h/1e14)**B * (1. + self.z)**C

            # pressure normalization: Eq 24 of Hill and Pajer 2013
            self._P0 = param_fit(18.1, 0.154, -0.758)

            # core radius: Eq 25 of Hill and Pajer 2013
            self._xc = param_fit(0.497, -0.00865, 0.731)

            # outer log slope: Eq 26 of Hill and Pajer 2013
            self._beta = param_fit(4.35, 0.0393, 0.415)
            
            return self._P0, self._xc, self._beta
    
    @Battaglia_params.deleter
    def Battaglia_params(self):
        try:
            del self._P0
            del self._xc
            del self._beta
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    # CALLABLE PROFILES
    ##--------------------------------------------------------------------------
    def dark_matter_density(self, R):
        """
        Compute the dark matter density [units :math: `h^2 M_\odot / Mpc^3`] 
        at 3D radius `R` [units :math: `Mpc h^{-1}`] using the universal 
        NFW profile
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the density at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        rho_dm : {float, array_like}
            The density at `R` [units :math: `h^2 M_\odot / Mpc^3`]
        """
        x = R/self.scale_radius
        rho_dm = self.scale_density / (x * (1. + x)**2)
        
        return rho_dm
    #---------------------------------------------------------------------------
    def dimensionless_gas_density(self, R):
        """
        Compute the dimensionless gas density [units: None] at 3D radius `R` 
        [units :math: `Mpc h^{-1}`].
        
        Notes
        -----
        Given by Eq. 15 in Komatsu and Seljak 2002. 
        """
        x = R/self.scale_radius
        
        return ( 1. - self._B * (1. - np.log(1. + x)/x) )**(1./(self.gamma - 1.))
    #---------------------------------------------------------------------------
    def gas_density(self, R):
        """
        Compute the gas density [units :math: `h^2 M_\odot / Mpc^3`] 
        at 3D radius `R` [units :math: `Mpc h^{-1}`], found by solving 
        hydrostatic equilibrium.
        
        Notes
        -----
        Given by Eq. 12 in Komatsu and Seljak 2002. 
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the density at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        rho_gas : {float, array_like}
            The density at `R` [units :math: `h^2 M_\odot / Mpc^3`]
        """
        rho_gas = self.rho_gas_0 * self.dimensionless_gas_density(R)
        
        return rho_gas
    #---------------------------------------------------------------------------
    def gas_temperature(self, R):   
        """
        Compute the temperature [units: keV] at 3D radius `R` 
        [units :math: `Mpc h^{-1}`], found by assuming a polytropic relationship,
        :math: \T_{gas} \propto \rho_{gas}^{\gamma - 1}
        
        Notes
        -----
        Given by Eq. 13 in Komatsu and Seljak 2002. 
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the temp at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        T_gas : {float, array_like}
            The temperature at `R` [units: keV]
        """
        T_gas = self.T_gas_0 * self.dimensionless_gas_density(R)**(self.gamma-1.)
        
        return T_gas
    #---------------------------------------------------------------------------
    def gas_pressure(self, R):
        """
        Return the thermal gas pressure either using the Komatsu and Seljak 
        analytic profile or the Battaglia et al. 2012 simulation-based profile
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the pressure at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        P : {float, array_like}
            The pressure at `R` [units :math: h^2 eV / cm^3]
        """
        choices = ['KS', 'Battaglia']

        if self.pressure_fit == "KS":
            return self.gas_pressure_KS(R)
        elif self.pressure_fit == "Battaglia":
            return self.gas_pressure_Battaglia(R)
        else:
            raise ValueError("Pressure fit '%s' not recognized. Must be one of %s" \
                            %(self.pressure_fit, choices))
    #---------------------------------------------------------------------------
    def gas_pressure_KS(self, R):
        """
        Compute the thermal gas pressure [units :math: h^2 eV / cm^3] at 3D radius 
        `R` [units :math: `Mpc h^{-1}`], found by assuming a polytropic relationship,
        :math: \P_{gas} \propto \rho_{gas}^\gamma
        
        Notes
        -----
        Given by Eq. 14 in Komatsu and Seljak 2002. 
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the pressure at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        P_gas : {float, array_like}
            The pressure at `R` [units :math: h^2 eV / cm^3]
        """
        P_gas = self.P0_KS * self.dimensionless_gas_density(R)**(self.gamma)
        
        return P_gas
    #---------------------------------------------------------------------------
    def gas_pressure_Battaglia(self, R):
        """
        Compute the thermal gas pressure [units :math: h^2 eV / cm^3] at 3D radius 
        `R` [units :math: `Mpc h^{-1}`], as derived from simulations presented
        in Battaglia et al. 2012.
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute the pressure at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        P_gas : {float, array_like}
            The pressure at `R` [units :math: h^2 eV / cm^3]
        """
        # first compute P200c, in h^2 eV / cm^3
        P_200c = self.P0_Battaglia
        
        P0, xc, beta = self.Battaglia_params
        
        x = R / self.radius_200c
        return (P_200c * P0) * (x / xc)**(-0.3) / (1. + x/xc)**beta
    #---------------------------------------------------------------------------
    def comptonY_3D(self, R):
        """
        Compute the 3D Compton Y profile [units :math: h^2 Mpc^{-1}] at the 3D radius `R`
        [units :math: `Mpc h^{-1}`], using the pressure profile specified 
        by `self.pressure_fit`
        
        Notes
        -----
        The 3D Compton Y profile is given by: 
        
        ..math: y_{3D}(x) = \frac{\sigma_T}{m_e c^2} P_e(x)
        
        and the electron pressure is related to the thermal gas pressure as
        
        ..math: P_e(x) = 2*(1 + X) / (3 + 5X) P_{gas} 
        
        Parameters
        ----------
        R : {float, array_like}
            The 3D radius to compute y at [units :math: `Mpc h^{-1}`]
        
        Returns
        -------
        y3D : {float, array_like}
            The compton Y profile at `R` [units :math: h^2 Mpc^{-1}]
        """
        prefactor = const.sigma_T / (const.m_e * const.c_light**2 / const.eV)
        y3D = prefactor * 2.*(1 + self.X)/(3. + 5.*self.X) * self.gas_pressure(R)
        
        return y3D * const.Mpc
    #---------------------------------------------------------------------------
    def comptonY_2D(self, thetas):
        """
        Compute the 2D Compton Y profile [units: None] as a function of 
        angle `\theta`, which is the angular distance from the halo center 
        in the plane of the sky, by integrating over the line-of-sight through
        the halo.
        
        Notes
        -----
        The 2D Compton Y profile is given by: 
        
        ..math: y_{2D}(\theta) = \int_0^\infty y_{2D}(\sqrt(l^2 + D_A^2 \theta^2)) dl

        Parameters
        ----------
        thetas : {float, array_like}
            The 2D angular distance from the halo center [units: arcminutes]
        
        Returns
        -------
        y2D : {float, array_like}
            The 2D Compton Y profile [units: None]
        """
        this_Da = self.Da
        
        # set up the cylindrical integral for each theta value
        integrand = lambda l, th: 2.*self.comptonY_3D(np.sqrt(l**2 + (this_Da*th)**2))
        cyc_integral = np.vectorize(lambda this_theta: integ.quad(integrand, 0., \
                                        5.*self.virial_radius, args=(this_theta,))[0])
        
        # multiply by h^2 to make it dimensionless
        y2D = cyc_integral(thetas*const.arcminute) * self.cosmo.h**2
        
        return y2D
    #-------------------------------------------------------------------------------
    def optical_depth(self, thetas):    
        """
        Compute the 2D Thomson optical depth profile [units: None] as a function of 
        angle `\theta`, which is the angular distance from the halo center 
        in the plane of the sky, by integrating over the line-of-sight through
        the halo.
        
        Notes
        -----
        The Thomson optical depth profile is given by: 
        
        ..math: \tau(\theta) = \sigma_T \int_0^\infty n_e(\sqrt(l^2 + D_A^2 \theta^2)) dl

        Parameters
        ----------
        thetas : {float, array_like}
            The 2D angular distance from the halo center [units: arcminutes]
        
        Returns
        -------
        tau : {float, array_like}
            The 2D optical depth profile [units: None]
        """ 
        this_Da = self.Da
        
        # define the optical depth function, has units of h^2 / Mpc
        prefactor = const.sigma_T*0.5*(1 + self.X)/const.m_p * (const.M_sun/const.Mpc**2)
        tau3D = lambda r: prefactor * self.gas_density(r)
        
        # set up the cylindrical integral for each theta value
        integrand = lambda l, th: 2.*tau3D(np.sqrt(l**2 + (this_Da*th)**2))
        cyc_integral = np.vectorize(lambda this_theta: integ.quad(integrand, 0., \
                                        5.*self.virial_radius, args=(this_theta,))[0])
        
        # multiply by h^2 so it's dimensionless
        tau = cyc_integral(thetas*const.arcminute) * self.cosmo.h**2
        
        return tau
    #---------------------------------------------------------------------------
    def thermal_sz(self, thetas, freq):
        """
        The thermal Sunyaev-Zel'dovich profile [units :math: `\mu K`] as a 
        function of angle `\theta`. 
        
        Notes
        -----
        The temperature change due to the thermal SZ is given by:
        
        ..math: \delta T_{tSZ} / T_{CMB} = f(\nu) * y_{2D}(\theta)
        
        Parameters
        ----------
        thetas : {float, array_like}
            The 2D angular distance from the halo center [units: arcminutes]
        freq : float
            The frequency to compute the effect at [units: GHz]
        
        Returns
        -------
        dT : {float, array_like}
            The temperature change due to the thermal SZ effect [units :math: `\mu K`]
        """
        freq_factor = tools.f_sz(freq)
        dT = freq_factor * self.comptonY_2D(thetas) * (const.T_cmb / const.micro)
        
        return dT
    #---------------------------------------------------------------------------
    def kinetic_sz(self, thetas, velocity):
        """
        The kinetic Sunyaev-Zel'dovich profile [units :math: `\mu K`] as a 
        function of angle `\theta`. 
        
        Notes
        -----
        The temperature change due to the kinetic SZ is given by:
        
        ..math: \delta T_{kSZ} / T_{CMB} = (v/c) * \tau(\theta)
        
        Parameters
        ----------
        thetas : {float, array_like}
            The 2D angular distance from the halo center [units: arcminutes]
        velocity : float
            The peculiar velocity of the halo, with positive values indicating
            an object receding from the observer [units: km/s]
        
        Returns
        -------
        dT : {float, array_like}
            The temperature change due to the kinetic SZ effect [units :math: `\mu K`]
        """
        v_c = (velocity*const.km/const.second) / const.c_light 
        dT = -self.optical_depth(thetas) * v_c * (const.T_cmb / const.micro) 
        
        return dT
    #---------------------------------------------------------------------------
    def optical_depth_FT(self, ell):    
        """
        Compute the Foruier transform of the 2D Thomson optical depth profile 
        [units: None] as a function of multipole `\ell`, the Fourier analog
        of the angular radius `\theta`.

        Parameters
        ----------
        ell : {float, array_like}
            The Fourier multipoles to compute the transform at [units: None]
        
        Returns
        -------
        tau_ft : array_like
            The Fourier transform of the 2D optical depth profile as function of 
            multipole, including the values for negative frequencies in the 
            style of numpy.fft
        """  
        # first compute the real space profile
        min_theta = 0.
        max_theta = 60.
        dtheta = 0.05

        thetas = np.arange(min_theta, max_theta, dtheta)
        tau = self.optical_depth(thetas) 
        tau0 = self.optical_depth(0.)
        
        return self._fourier_transform(ell, thetas, tau, tau0)
    #---------------------------------------------------------------------------
    def comptonY_2D_FT(self, ell):    
        """
        Compute the 2D Compton Y profile [units: None] as a function of 
        multipole `\ell`, the Fourier analog of the angular radius `\theta`.

        Parameters
        ----------
        ell : {float, array_like}
            The Fourier multipoles to compute the transform at [units: None]
        
        Returns
        -------
        y_ft : array_like
            The Fourier transform of the 2D compton y profile as function of 
            multipole, including the values for negative frequencies in the 
            style of numpy.fft
        """  
        # first compute the real space profile
        min_theta = 0.
        max_theta = 60.
        dtheta = 0.05

        thetas = np.arange(min_theta, max_theta, dtheta)
        y = self.comptonY_2D(thetas) 
        y0 = self.comptonY_2D(0.)
        
        return self._fourier_transform(ell, thetas, y, y0)
    #---------------------------------------------------------------------------
    def _fourier_transform(self, ell, theta, y, y0):
        """
        Fourier transform the real-space profile
        """  
        # make the full real space profile for FFT 
        y_mirror = y[1:][::-1]
        y_full = np.append(y, y_mirror)
        
        # compute the FFT
        Pl = np.fft.fft(y_full).real
        l = 2*np.pi*np.fft.fftfreq(len(y_full), d=(theta[1]-theta[0])*const.arcminute)
        
        # keep only the positive ell values
        inds = np.where(l >= 0)
        Pl = Pl[inds]
        l = l[inds]
        
        # ell value of 0.5 * theta_scale
        ell0 = 2.*np.pi/(0.2*self.scale_radius/self.Da)
        
        logspline = InterpolatedUnivariateSpline(l, np.log(Pl.real))
        slope = derivative(logspline, ell0, dx=1e-3)*ell0

        # now compute the combined model
        model_spline = InterpolatedUnivariateSpline(l, Pl.real) 
        
        # make the final array
        Pl_final = ell.copy()*0.
        
        inds = (ell <= ell0)
        Pl_final[inds] = model_spline(ell[inds])
        Pl_final[~inds] = model_spline(ell0)*(ell[~inds]/ell0)**slope
        
        # set up the mirror for negative frequencies in numpy.fft style
        Pl_mirror = Pl_final[1:][::-1]
        Pl_final = np.append(Pl_final, Pl_mirror)
        
        # now properly normalize so we ifft and recover the correct central value
        # DC value is sum(FT) / len(FT)
        norm = y0/(np.sum(Pl_final)/len(Pl_final))
        Pl_final = norm*Pl_final
        
        return Pl_final
    #---------------------------------------------------------------------------