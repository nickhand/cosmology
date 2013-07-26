"""
 halofit.py
 nonlinear power spectrum from R.E. Smith et al, MNRAS 341:1311 (2003), based
 on the halofit.f code and using updated parameter fits from 
 Takahashi et al 2012
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 06/30/2013
"""
import numpy as np
from scipy import special, optimize
from cosmology import cosmo, linear_growth, parameters, tf_eh

class nonlinear_power(linear_growth.linear_power):
    """
    This class implements the power spectrum from Smith 2003.
    
    Notes
    -----
    Default cosmology is the Planck 2013 parameter set.
    """
    def __init__(self, z = 0.,  
                       tf = 'EH_full', 
                       cosmo_params = None,
                       use_takahashi = True):
        """
        Parameters
        ----------
        z : float, optional
            the redshift to compute the power spectrum at. Default is z = 0
        tf : str, optional
            the available transfer functions are: 
                'BBKS' : approximation from Bardeen et al. 1986
                'EH_full' : full CDM + baryon with wiggles TF from EH 1998
                'EH_no_wiggles' : full CDM + baryon w/o wiggles from EH 1998
                'EH_no_baryons' :  CDM TF from EH 1998
            Default is EH_no_baryons
        cosmo_params : dict or str, optional
            dictionary of cosmological parameters or string defining a
            pre-defined cosmology from parameters module
        use_takahashi : bool, optional
            whether to use the updated halofit parameters from 
            Takahashi et al. (2012)
        """
        self.z_ = z
        self.use_takahashi = use_takahashi
        self.kNL_max = 1e6 # use linear spectrum beyond this k in (1/Mpc)
        self.golinear = False 
        
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
        
        self.update_cosmo()
    #end __init__
    
    #---------------------------------------------------------------------------
    def get_z(self):
        return self.z_

    def set_z(self,z):
        self.z_ = z
        self.calc_spectral_params_()

    z = property(get_z,set_z)
    
    #---------------------------------------------------------------------------
    def update_cosmo(self, **kwargs):
        """
        Update the cosmological parameters
        """
        # update the cosmo dictionary
        for k, v in kwargs.iteritems(): cosmo[k] = v
        
        self.om_m0 = cosmo.omega_m_0
        self.om_v0 = cosmo.omega_l_0
        self.sig8 = cosmo.sigma_8
        
        self.calc_spectral_params_()
    #end update_cosmo
    
    #---------------------------------------------------------------------------    
    def calc_spectral_params_(self):
        """
        Set the redshift of the power spectrum, and compute the
        associated spectral parameters
        """
        self.om_m = self.omega_m_z(self.z_)
        self.om_v = self.omega_l_z(self.z_)
        
        xlogr1 = -10.0
        xlogr2 = 3.5
        logstep = 2.0
        MAX_ITER = 3
        
        #iterate to determine
        #  k_nl  : wavenumber [1/Mpc] where nonlinear effects become important
        #  n_eff : effective spectral index
        #  n_cur : second derivative of the power spectrum at rknl
        
        def objective(r):
            sig, d1, d2 = self._wint(r)
            return sig - 1.0
        try:
            r_final = optimize.brentq(objective, 10**xlogr1, 10**xlogr2)
            sig, d1, d2 = self._wint(r_final)
        except:
            cnt = 0
            while True:
                
                # check if we are in the linear regime
                if (10**(-xlogr1) > self.kNL_max): 
                    self.golinear = True
                    return                    
                xlogr1 -= logstep
                xlogr2 += logstep
                try:
                    r_final = optimize.brentq(target, 10**xlogr1, 10**xlogr2)
                except:
                    if cnt > MAX_ITER:
                        raise
                    cnt += 1

        self.k_nl = 1./r_final
        self.n_eff = -3. - d1
        self.n_cur = -d2
    #end calc_spectral_params_
    
    #---------------------------------------------------------------------------    
    def _wint(self, r):
        """
        The subroutine wint from halofit.f, finds the effective spectral 
        quantities k_nl, n_eff & n_cur. This it does by calculating the radius 
        of the Gaussian filter at which the variance is unity = k_nl.
        n_eff is defined as the first derivative of the variance, calculated 
        at the nonlinear wavenumber and similarly the n_cur is the second
        derivative at the nonlinear wavenumber.
        
        returns (sig, d1, d2)
        """
        nint = 10000.
        t    = ( np.arange(nint)+0.5 )/nint
        k    = 1./t - 1.
        d2   = self.D2_L(k)
        x2   = k*k*r*r
        w1   = np.exp(-x2)
        w2   = 2*x2*w1
        w3   = 4*x2*(1-x2)*w1

        mult = d2/k/t/t
        
        sum1 = np.sum(w1*mult)/nint
        sum2 = np.sum(w2*mult)/nint
        sum3 = np.sum(w3*mult)/nint
        
        sig = np.sqrt(sum1)
        d1  = -sum2/sum1
        d2  = -sum2*sum2/sum1/sum1 - sum3/sum1
        
        return sig, d1, d2
    #end _wint
    
    #---------------------------------------------------------------------------
    def D2_L(self, k):
        """
        Return the dimensionless, linear power spectrum. Uses EH (1999) CDM 
        linear power spectrum by default.
        
        Paremeters
        ----------
        k : numpy.ndarray or float
            the wavenumber in units of 1 / Mpc
        """
        k = np.asarray(k)
        return self.P_k(k, self.z_) * k**3 / (2.*np.pi**2)
    #end D2_L
    
    #---------------------------------------------------------------------------
    def D2_NL(self, k, return_components = False):
        """
        Dimensionless halo model nonlinear fitting formula as described 
        in Appendix C of Smith et al. (2003). Uses EH (1999) CDM+baryon linear 
        power spectrum by default. The linear power spectrum will be returned
        if the nonlinear wavenumber is greater than 1e6 1/Mpc.
        
        Parameters
        ----------
        k : numpy.ndarray or float
            the wavenumber in units of 1 / Mpc
        
        Returns
        -------
        D2_NL : numpy.ndarray or float
            dimensionless, nonlinear matter power spectrum
        """
        if self.golinear and return_components:
            raise ValueError('input redshift is too large to use nonlinear' 
                             ' power spectrum; cannot return components')
        
        if self.golinear:
            return self.D2_L(k)
        
        k = np.asarray(k)
        n    = self.n_eff
        n_cur = self.n_cur
        k_nl  = self.k_nl
        plin  = self.D2_L(k)
        om_m  = self.om_m
        om_v  = self.om_v
        
        if not self.use_takahashi:
            gam = 0.8649 + 0.2989*n + 0.1631*n_cur
            a = 10**(1.4861 + 1.8369*n + 1.6762*n*n + 0.7940*n*n*n + \
                   0.1670*n*n*n*n - 0.6206*n_cur)
            b     = 10**(0.9463 + 0.9466*n + 0.3084*n*n - 0.940*n_cur)
            c     = 10**(-0.2807 + 0.6669*n + 0.3214*n*n - 0.0793*n_cur)
            mu    = 10**(-3.5442 + 0.1908*n)
            nu    = 10**(0.9589 + 1.2857*n)
            alpha = 1.3884 + 0.3700*n - 0.1452*n*n
            beta  = 0.8291 + 0.9854*n + 0.3401*n*n
        
        else:    
            gam = 0.1971 - 0.0843*n + 0.8460*n_cur
            a = 10**(1.5222 + 2.8553*n + 2.3706*n*n + 0.9903*n*n*n + \
                    0.2250*n*n*n*n - 0.6038*n_cur + 0.1749*om_v*(1+cosmo.w0))
            b     = 10**(-0.5642 + 0.5864*n + 0.5716*n*n - 1.5474*n_cur + \
                    0.2279*om_v*(1+cosmo.w0))
            c     = 10**(0.3698 + 2.0404*n + 0.8161*n*n + 0.5869*n_cur)
            mu    = 0.
            nu    = 10**(5.2105 + 3.6902*n)
            alpha = abs(6.0835 + 1.3373*n - 0.1959*n*n - 5.5274*n_cur)
            beta  = 2.0379 - 0.7354*n + 0.3157*n*n + 1.2490*n*n*n + \
                    0.3980*n*n*n*n - 0.1682*n_cur
        
        if abs(1-om_m) > 0.01: #omega evolution
            f1a  = om_m**(-0.0732)
            f2a  = om_m**(-0.1423)
            f3a  = om_m**(0.0725)
            f1b  = om_m**(-0.0307)
            f2b  = om_m**(-0.0585)
            f3b  = om_m**(0.0743)       
            frac = om_v/(1.-om_m) 
            f1   = frac*f1b + (1-frac)*f1a
            f2   = frac*f2b + (1-frac)*f2a
            f3   = frac*f3b + (1-frac)*f3a
        else:         
            f1 = 1.0
            f2 = 1.0
            f3 = 1.0
   
        y = (k/k_nl)
        
        ph = a*y**(3.*f1) / (1 + b*y**(f2) + (f3*c*y)**(3-gam))
        ph /= (1 + mu*y**(-1) + nu*y**(-2))
        pq = plin*(1 + plin)**beta / (1 + plin*alpha) * np.exp(-y/4. - y**2/8.)
        
        pnl = pq + ph

        if return_components:
            return pnl, pq, ph, plin
        else:
            return pnl
    #end D2_NL
    
    #---------------------------------------------------------------------------
    def D2_NL_PD96(self, klin):
        """
        Implement the Peacock & Dodds 1996 power spectrum. Because of the way 
        this is calculated, the user must supply the linear wave number, and a 
        tuple (rk_pd,pnl_pd) is returned. rk_pd is the nonlinear wave number 
        associated with the input linear wave number, and pnl_pd is the 
        nonlinear power spectrum associated with rk_pd.
        
        Notes
        -----
        Uses the EH CDM transfer function
        """
        
        # switch to EH CDM transfer function
        old_tf = self.tf
        tf = tf_eh.tf_eh(cosmo)
        self.tf = tf.no_baryons
        
        plin  = self.D2_L(klin)
        
        n_pd = self._n_cdm(klin)

        pnl_pd = self._f_pd(plin, n_pd)
        
        k_pd = klin * (1. + pnl_pd)**(1./3.)

        self.tf = old_tf
        return k_pd, pnl_pd
    #end D2_NL_PD96
    
    #---------------------------------------------------------------------------
    def _n_cdm(self, k):
        """
        Effective spectral index used in Peacock & Dodds (1996)
        """
        y     = self.D2_L(0.5*k)
        yplus = self.D2_L(0.5*k*1.01)
        return -3. + np.log(yplus/y)*100.5
    #end _n_cdm

    #---------------------------------------------------------------------------
    def _f_pd(self, y, n):
        """
        Peacock & Dodds (1996) fitting formula
        """
        g = 2.5*self.om_m / (self.om_m**(4./7.) - self.om_v + \
             (1. + 0.5*self.om_m) * (1. + self.om_v/70.))
        a   = 0.482*(1. + n/3.)**(-0.947)
        b   = 0.226*(1. + n/3.)**(-1.778)
        alp = 3.310*(1. + n/3.)**(-0.244)
        bet = 0.862*(1. + n/3.)**(-0.287)
        vir = 11.55*(1. + n/3.)**(-0.423)
        return y * ( (1.+ b*y*bet + (a*y)**(alp*bet)) / \
                     (1.+ ((a*y)**alp*g*g*g/vir/y**0.5)**bet ) )**(1./bet)
    #end _f_pd
    #---------------------------------------------------------------------------
if __name__ == '__main__':
    import pylab

    z = 0.
    
    PSpec = nonlinear_power(z=z, cosmo_params='Planck13')

    # calculate S03 power law
    N = 1000
    rk = 10**( -5.0 + 12.0*np.linspace(0,1,N) )

    pnl, pq, ph, plin = PSpec.D2_NL(rk,return_components = True)
    
    pylab.loglog(rk, pnl, '-k', label='nonlinear')
    pylab.loglog(rk, pq, ':k', label='quasi-linear')
    pylab.loglog(rk, ph, ':r',label='halo')
    pylab.loglog(rk, plin, '--r',label='linear')

    #calculate PD96 power law
    rk_lin = 10**( -2.0+4.0*np.linspace(0,1,N) )
    rk_pd, pnl_pd = PSpec.D2_NL_PD96(rk_lin)
       
    pylab.loglog(rk_pd, pnl_pd, '--k', label='PD96')
    
    pylab.ylim(10**-1.5,3E3)
    pylab.xlim(10**-1.5,1E2)
    
    pylab.xlabel(r'$k$ (1/Mpc)')
    pylab.ylabel(r'$\Delta^2(k)$')
    pylab.title('z=%.2f Power Spectrum' % z)
    
    pylab.legend(loc=0)

    pylab.show()

